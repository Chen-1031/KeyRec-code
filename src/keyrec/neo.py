"""NEO-specific extraction and structured injection for KeyRec."""

from __future__ import annotations

import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

import torch

from .memory import TokenPayload
from .readout import ReadoutResult


NativeImageLoader = Callable[..., tuple[torch.Tensor, torch.Tensor]]


@dataclass(frozen=True)
class NeoVisualSpan:
    """Selected tokens from one source frame with original spatial indexes."""

    tokens: torch.Tensor
    positions: torch.Tensor
    source_time: float

    def __post_init__(self) -> None:
        payload = TokenPayload(
            self.tokens,
            torch.full(
                (self.tokens.shape[0],),
                float(self.source_time),
                device=self.tokens.device,
                dtype=torch.float32,
            ),
            self.positions,
        )
        if len(payload) == 0:
            raise ValueError("NEO visual spans cannot be empty")
        if payload.positions is None or payload.positions.shape != (len(payload), 2):
            raise ValueError("NEO span positions must have shape [token, 2]")
        if (payload.positions < 0).any():
            raise ValueError("NEO span positions must be non-negative")
        object.__setattr__(self, "tokens", payload.tokens)
        object.__setattr__(self, "positions", payload.positions)
        object.__setattr__(self, "source_time", float(self.source_time))

    @property
    def token_count(self) -> int:
        return int(self.tokens.shape[0])


@dataclass(frozen=True)
class PreparedNeoInput:
    """Validated NEO embeddings and aligned temporal-spatial indexes."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    visual_tokens: torch.Tensor
    indexes: torch.Tensor
    image_token_id: int
    span_token_counts: tuple[int, ...]
    span_times: tuple[float, ...]
    prompt: str

    def __post_init__(self) -> None:
        if self.input_ids.ndim != 2 or self.input_ids.shape[0] != 1:
            raise ValueError("input_ids must have shape [1, sequence]")
        if self.input_ids.dtype != torch.long:
            raise ValueError("input_ids must use torch.long")
        if self.attention_mask.shape != self.input_ids.shape:
            raise ValueError("attention_mask must match input_ids")
        if self.indexes.shape != (3, self.input_ids.shape[1]):
            raise ValueError("indexes must have shape [3, sequence]")
        if self.indexes.dtype != torch.long:
            raise ValueError("indexes must use torch.long")
        if self.visual_tokens.ndim != 2 or self.visual_tokens.shape[0] == 0:
            raise ValueError("visual_tokens must have shape [token, feature]")
        if not self.visual_tokens.is_floating_point():
            raise ValueError("visual_tokens must use a floating-point dtype")
        if not isinstance(self.prompt, str) or not self.prompt:
            raise ValueError("prompt must be a non-empty string")
        devices = {
            self.input_ids.device,
            self.attention_mask.device,
            self.visual_tokens.device,
            self.indexes.device,
        }
        if len(devices) != 1:
            raise ValueError("prepared NEO tensors must use the same device")
        if len(self.span_token_counts) == 0:
            raise ValueError("prepared NEO input must contain at least one span")
        if len(self.span_token_counts) != len(self.span_times):
            raise ValueError("span counts and times must have equal length")
        if any(count <= 0 for count in self.span_token_counts):
            raise ValueError("NEO span token counts must be positive")
        if sum(self.span_token_counts) != self.visual_tokens.shape[0]:
            raise ValueError("NEO span counts must equal the visual-token count")
        if tuple(self.span_times) != tuple(sorted(self.span_times)):
            raise ValueError("NEO source spans must be in chronological order")

        selected = self.input_ids[0] == self.image_token_id
        placeholder_count = int(selected.sum().item())
        if placeholder_count != self.visual_tokens.shape[0]:
            raise ValueError("NEO placeholder count must equal visual-token count")
        if (self.indexes[1:, selected] < 0).any():
            raise ValueError("NEO spatial indexes must be non-negative")
        temporal = self.indexes[0, selected]
        cursor = 0
        previous = None
        for count in self.span_token_counts:
            values = temporal[cursor : cursor + count]
            if values.numel() != count or not torch.equal(values, values[:1].expand_as(values)):
                raise ValueError("each NEO source span must share one temporal index")
            current = int(values[0].item())
            if previous is not None and current <= previous:
                raise ValueError("NEO source-span temporal indexes must increase")
            previous = current
            cursor += count


class NeoKeyRecAdapter:
    """Preserve NEO's native codec with structured sparse-memory injection."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        load_image_native: NativeImageLoader | None = None,
        patch_size: int = 16,
        downsample_ratio: float = 0.5,
        min_pixels: int = 4 * 32 * 32,
        max_pixels: int = 360 * 640,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.load_image_native = load_image_native
        self.patch_size = _positive_integer(patch_size, "patch_size")
        self.downsample_ratio = float(downsample_ratio)
        self.min_pixels = _positive_integer(min_pixels, "min_pixels")
        self.max_pixels = _positive_integer(max_pixels, "max_pixels")
        if not 0.0 < self.downsample_ratio <= 1.0:
            raise ValueError("downsample_ratio must be in (0, 1]")
        if self.min_pixels > self.max_pixels:
            raise ValueError("min_pixels must not exceed max_pixels")
        self.downsample_factor = _downsample_factor(self.downsample_ratio)
        self._validate_components()
        model_ratio = float(
            getattr(self.model, "downsample_ratio", self.downsample_ratio)
        )
        if abs(model_ratio - self.downsample_ratio) > 1e-6:
            raise ValueError("adapter and NEO model downsample ratios must match")
        self.feature_size = int(self.model.config.llm_config.hidden_size)
        self.last_prepared_input: PreparedNeoInput | None = None
        self.last_spans: tuple[NeoVisualSpan, ...] = ()

    @classmethod
    def from_pretrained(cls, model_path: str) -> "NeoKeyRecAdapter":
        """Load the same released NEO classes and preprocessing as research code."""

        from transformers import AutoModel, AutoTokenizer

        from .neo_preprocess import load_image_native

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=False,
        )
        model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map="auto",
        ).eval()
        return cls(model, tokenizer, load_image_native=load_image_native)

    @property
    def device(self) -> torch.device:
        return torch.device(self.model.device)

    @property
    def dtype(self) -> torch.dtype:
        return self.model.dtype

    @property
    def image_token_id(self) -> int:
        return int(self.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>"))

    @property
    def image_start_token_id(self) -> int:
        return int(self.tokenizer.convert_tokens_to_ids("<img>"))

    @torch.inference_mode()
    def extract_frame(self, frame: Any, timestamp: float) -> TokenPayload:
        """Extract one NEO frame and retain its row-major source coordinates."""

        if self.load_image_native is None:
            raise RuntimeError("NEO native image preprocessing is not configured")
        time_value = float(timestamp)
        if not torch.isfinite(torch.tensor(time_value)):
            raise ValueError("timestamp must be finite")
        if not hasattr(frame, "convert"):
            raise TypeError("NEO frames must expose PIL-compatible convert()")

        with tempfile.TemporaryDirectory(prefix="keyrec_neo_features_") as tmp:
            frame_path = Path(tmp) / "neoimg_0001.jpg"
            frame.convert("RGB").save(frame_path, quality=95)
            pixel_values, grid_hw = self.load_image_native(
                str(frame_path),
                patch_size=self.patch_size,
                downsample_ratio=self.downsample_ratio,
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels,
            )

        pixel_values, grid_hw = self._validate_preprocessed(pixel_values, grid_hw)
        features = self.model.extract_feature(
            pixel_values.to(self.device, dtype=self.dtype),
            grid_hw=grid_hw.to(self.device),
        ).detach()
        if features.ndim == 3 and features.shape[0] == 1:
            features = features.squeeze(0)
        if features.ndim != 2 or features.shape[1] != self.feature_size:
            raise ValueError("NEO extracted features have an invalid shape")

        raw_height = int(grid_hw[0, 0].item())
        raw_width = int(grid_hw[0, 1].item())
        if raw_height % self.downsample_factor or raw_width % self.downsample_factor:
            raise ValueError("NEO grid is not divisible by the downsample factor")
        height = raw_height // self.downsample_factor
        width = raw_width // self.downsample_factor
        rows, columns = torch.meshgrid(
            torch.arange(height, device=features.device, dtype=torch.long),
            torch.arange(width, device=features.device, dtype=torch.long),
            indexing="ij",
        )
        positions = torch.stack([rows.reshape(-1), columns.reshape(-1)], dim=1)
        if positions.shape[0] != features.shape[0]:
            raise RuntimeError(
                "NEO visual feature/position count mismatch: "
                f"{features.shape[0]} features vs {positions.shape[0]} positions"
            )
        times = torch.full(
            (features.shape[0],),
            time_value,
            device=features.device,
            dtype=torch.float32,
        )
        return TokenPayload(features, times, positions)

    @torch.inference_mode()
    def encode_query(self, question: str) -> torch.Tensor:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")
        inputs = self.tokenizer(question, return_tensors="pt")
        input_ids = _batch_tensor(inputs, "input_ids").to(self.device)
        embeddings = self.model.language_model.get_input_embeddings()(input_ids)
        query = embeddings.mean(dim=1).squeeze(0)
        if query.ndim != 1 or query.shape[0] != self.feature_size:
            raise ValueError("NEO query embedding has an invalid feature size")
        return query

    @torch.inference_mode()
    def generate_router_output(self, prompt: str, max_new_tokens: int) -> str:
        _positive_integer(max_new_tokens, "max_new_tokens")
        with self._temporary_image_token_ids():
            query, eos_token_id, separator = self._build_query(prompt, ())
            inputs = self.tokenizer(query, return_tensors="pt")
            input_ids = _batch_tensor(inputs, "input_ids").to(self.device)
            attention_mask = _batch_tensor(inputs, "attention_mask").to(self.device)
            output_ids = self.model.generate(
                pixel_values=None,
                input_ids=input_ids,
                grid_hw=None,
                attention_mask=attention_mask,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                top_p=None,
                eos_token_id=eos_token_id,
            )
        return self._decode(output_ids, separator)

    def prepare_generation(
        self,
        question: str,
        readout: ReadoutResult,
    ) -> PreparedNeoInput:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")
        spans = build_neo_visual_spans(readout)
        lines = ["Selected visual memory segments:"]
        for index, span in enumerate(spans, start=1):
            lines.append(
                f"Memory segment {index} ({span.token_count} visual tokens):"
            )
            lines.append("<image>")
        lines.extend(["Question:", question])
        prompt = "\n".join(lines)
        counts = tuple(span.token_count for span in spans)

        with self._temporary_image_token_ids():
            query, _eos_token_id, _separator = self._build_query(prompt, counts)
            inputs = self.tokenizer(query, return_tensors="pt")
            input_ids = _batch_tensor(inputs, "input_ids").to(self.device)
            attention_mask = _batch_tensor(inputs, "attention_mask").to(self.device)
            placeholder_count = int(
                (input_ids[0] == self.image_token_id).sum().item()
            )
            if placeholder_count != sum(counts):
                raise ValueError(
                    "NEO placeholder count must equal visual-token count"
                )
            grid_hw = torch.cat(
                [self._placeholder_grid(count) for count in counts],
                dim=0,
            )
            indexes = self.model.get_thw_indexes(input_ids[0], grid_hw)

        if not isinstance(indexes, torch.Tensor) or indexes.shape != (
            3,
            input_ids.shape[1],
        ):
            raise ValueError("NEO get_thw_indexes returned an invalid shape")
        indexes = indexes.to(self.device, dtype=torch.long)
        selected = input_ids[0] == self.image_token_id
        positions = torch.cat([span.positions.to(self.device) for span in spans], dim=0)
        if int(selected.sum().item()) != positions.shape[0]:
            raise ValueError("NEO placeholder count must equal position count")
        indexes = indexes.clone()
        indexes[1, selected] = positions[:, 0]
        indexes[2, selected] = positions[:, 1]
        visual_tokens = torch.cat(
            [span.tokens.to(self.device, dtype=torch.bfloat16) for span in spans],
            dim=0,
        )
        prepared = PreparedNeoInput(
            input_ids=input_ids,
            attention_mask=attention_mask,
            visual_tokens=visual_tokens,
            indexes=indexes,
            image_token_id=self.image_token_id,
            span_token_counts=counts,
            span_times=tuple(span.source_time for span in spans),
            prompt=prompt,
        )
        if prepared.visual_tokens.shape[1] != self.feature_size:
            raise ValueError("prepared NEO visual feature size is invalid")
        self.last_spans = spans
        self.last_prepared_input = prepared
        return prepared

    @torch.inference_mode()
    def generate_answer(
        self,
        question: str,
        readout: ReadoutResult,
        max_new_tokens: int,
    ) -> str:
        _positive_integer(max_new_tokens, "max_new_tokens")
        prepared = self.prepare_generation(question, readout)
        _query, eos_token_id, separator = self._build_query("", ())
        input_embeddings = self.model.language_model.get_input_embeddings()(
            prepared.input_ids
        ).clone()
        selected = prepared.input_ids[0] == prepared.image_token_id
        input_embeddings[0, selected] = prepared.visual_tokens.to(
            input_embeddings.dtype
        )
        output_ids = self.model.language_model.generate(
            inputs_embeds=input_embeddings,
            indexes=prepared.indexes,
            attention_mask=prepared.attention_mask,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            top_p=None,
            eos_token_id=eos_token_id,
            use_cache=True,
        )
        return self._decode(output_ids, separator)

    def _placeholder_grid(self, token_count: int) -> torch.Tensor:
        _positive_integer(token_count, "token_count")
        return torch.tensor(
            [[self.downsample_factor, self.downsample_factor * token_count]],
            dtype=torch.long,
            device=self.device,
        )

    def _build_query(
        self,
        question: str,
        span_token_counts: tuple[int, ...],
    ) -> tuple[str, int, str]:
        get_conv_template = _conversation_factory(self.model)
        template = get_conv_template(self.model.template)
        template.system_message = self.model.system_message
        separator = template.sep.strip()
        eos_token_id = int(self.tokenizer.convert_tokens_to_ids(separator))
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()
        if query.count("<image>") != len(span_token_counts):
            raise ValueError("NEO prompt image markers must match source spans")
        for count in span_token_counts:
            image_tokens = "<img>" + "<IMG_CONTEXT>" * count + "</img>"
            query = query.replace("<image>", image_tokens, 1)
        return query, eos_token_id, separator

    def _decode(self, output_ids: Any, separator: str) -> str:
        if not isinstance(output_ids, torch.Tensor) or output_ids.ndim != 2:
            raise ValueError("NEO generation must return rank-2 token ids")
        decoded = self.tokenizer.batch_decode(
            output_ids,
            skip_special_tokens=True,
        )[0]
        return decoded.split(separator)[0].strip()

    def _validate_preprocessed(
        self,
        pixel_values: Any,
        grid_hw: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(pixel_values, torch.Tensor) or pixel_values.ndim != 2:
            raise ValueError("NEO pixel_values must be a rank-2 tensor")
        if not pixel_values.is_floating_point():
            raise ValueError("NEO pixel_values must use a floating-point dtype")
        if not isinstance(grid_hw, torch.Tensor) or grid_hw.shape != (1, 2):
            raise ValueError("NEO grid_hw must have shape [1, 2]")
        if grid_hw.is_floating_point() or grid_hw.is_complex():
            raise ValueError("NEO grid_hw must use an integer dtype")
        if (grid_hw <= 0).any():
            raise ValueError("NEO grid dimensions must be positive")
        return pixel_values, grid_hw.to(dtype=torch.long)

    def _validate_components(self) -> None:
        required_model = (
            "chat",
            "config",
            "device",
            "dtype",
            "extract_feature",
            "generate",
            "get_thw_indexes",
            "language_model",
            "system_message",
            "template",
        )
        missing = [name for name in required_model if not hasattr(self.model, name)]
        if missing:
            raise TypeError(f"model is missing required attributes: {missing}")
        required_tokenizer = (
            "batch_decode",
            "convert_tokens_to_ids",
        )
        missing = [
            name for name in required_tokenizer if not hasattr(self.tokenizer, name)
        ]
        if missing or not callable(self.tokenizer):
            raise TypeError(f"tokenizer is missing required behavior: {missing}")

    @contextmanager
    def _temporary_image_token_ids(self) -> Iterator[None]:
        with _temporary_attributes(
            self.model,
            {
                "img_context_token_id": self.image_token_id,
                "img_start_token_id": self.image_start_token_id,
            },
        ):
            yield


def build_neo_visual_spans(readout: ReadoutResult) -> tuple[NeoVisualSpan, ...]:
    """Group selected tokens into globally ordered sparse source-frame spans."""

    if not isinstance(readout, ReadoutResult):
        raise TypeError("readout must be a ReadoutResult")
    rows: list[tuple[float, int, int, torch.Tensor, torch.Tensor]] = []
    for segment in readout.segments:
        payload = segment.payload
        if payload.positions is None or payload.positions.shape != (len(payload), 2):
            raise ValueError("structured NEO memory requires [token, 2] positions")
        if (payload.positions < 0).any():
            raise ValueError("structured NEO memory contains negative positions")
        for token_index in range(len(payload)):
            rows.append(
                (
                    float(payload.times[token_index].item()),
                    segment.ordinal,
                    token_index,
                    payload.tokens[token_index : token_index + 1],
                    payload.positions[token_index : token_index + 1],
                )
            )
    if not rows:
        raise ValueError("readout must contain structured NEO visual tokens")
    rows.sort(key=lambda row: (row[0], row[1], row[2]))

    spans: list[NeoVisualSpan] = []
    current_time = rows[0][0]
    token_parts: list[torch.Tensor] = []
    position_parts: list[torch.Tensor] = []
    for source_time, _ordinal, _index, token, position in rows:
        if source_time != current_time:
            spans.append(
                NeoVisualSpan(
                    torch.cat(token_parts, dim=0),
                    torch.cat(position_parts, dim=0),
                    current_time,
                )
            )
            current_time = source_time
            token_parts = []
            position_parts = []
        token_parts.append(token)
        position_parts.append(position)
    spans.append(
        NeoVisualSpan(
            torch.cat(token_parts, dim=0),
            torch.cat(position_parts, dim=0),
            current_time,
        )
    )
    return tuple(spans)


@contextmanager
def _temporary_attributes(
    obj: Any,
    replacements: dict[str, Any],
) -> Iterator[None]:
    previous = {
        name: (hasattr(obj, name), getattr(obj, name, None))
        for name in replacements
    }
    for name, value in replacements.items():
        setattr(obj, name, value)
    try:
        yield
    finally:
        for name, (existed, value) in previous.items():
            if existed:
                setattr(obj, name, value)
            else:
                delattr(obj, name)


def _conversation_factory(model: Any) -> Callable[[str], Any]:
    try:
        factory = model.chat.__globals__["get_conv_template"]
    except (AttributeError, KeyError, TypeError) as exc:
        raise TypeError("NEO model.chat must expose get_conv_template") from exc
    if not callable(factory):
        raise TypeError("NEO get_conv_template must be callable")
    return factory


def _batch_tensor(batch: Any, name: str) -> torch.Tensor:
    try:
        value = batch[name]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"tokenizer output must contain {name}") from exc
    if not isinstance(value, torch.Tensor) or value.ndim != 2:
        raise ValueError(f"tokenizer {name} must be a rank-2 tensor")
    return value


def _downsample_factor(ratio: float) -> int:
    factor = round(1.0 / ratio)
    if factor <= 0 or abs(factor * ratio - 1.0) > 1e-6:
        raise ValueError("downsample_ratio must have an integer reciprocal")
    return factor


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value
