"""Gemma-specific extraction and injection for the KeyRec pipeline."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

import torch

from .memory import TokenPayload
from .readout import ReadoutResult


class _StopVisionForward(Exception):
    """Private control flow used to stop after Gemma's projected vision tokens."""


class _VideoFeatureOutput:
    def __init__(self, pooler_output: torch.Tensor) -> None:
        self.pooler_output = pooler_output


@dataclass(frozen=True)
class PreparedGemmaInput:
    """Validated Gemma token ids and their replacement visual embeddings."""

    input_ids: torch.Tensor
    visual_tokens: torch.Tensor
    video_token_id: int
    prompt: str

    def __post_init__(self) -> None:
        if isinstance(self.video_token_id, bool) or not isinstance(
            self.video_token_id, int
        ):
            raise TypeError("video_token_id must be an integer")
        if not isinstance(self.prompt, str) or not self.prompt:
            raise ValueError("prompt must be a non-empty string")
        if self.input_ids.ndim != 2 or self.input_ids.shape[0] != 1:
            raise ValueError("input_ids must have shape [1, sequence]")
        if self.input_ids.dtype != torch.long:
            raise ValueError("input_ids must use torch.long")
        if self.visual_tokens.ndim != 2 or self.visual_tokens.shape[0] == 0:
            raise ValueError("visual_tokens must have shape [token, feature]")
        if not self.visual_tokens.is_floating_point():
            raise ValueError("visual_tokens must use a floating-point dtype")
        if self.input_ids.device != self.visual_tokens.device:
            raise ValueError("input_ids and visual_tokens must use the same device")
        placeholder_count = int((self.input_ids == self.video_token_id).sum().item())
        if placeholder_count != self.visual_tokens.shape[0]:
            raise ValueError(
                "Gemma video-placeholder count must equal visual-token count"
            )


class GemmaKeyRecAdapter:
    """Preserve the research Gemma codec behind a narrow, safe interface."""

    def __init__(
        self,
        model: Any,
        processor: Any,
        *,
        max_soft_tokens: int = 70,
    ) -> None:
        if isinstance(max_soft_tokens, bool) or not isinstance(max_soft_tokens, int):
            raise ValueError("max_soft_tokens must be a positive integer")
        if max_soft_tokens <= 0:
            raise ValueError("max_soft_tokens must be a positive integer")
        self.model = model
        self.processor = processor
        self.max_soft_tokens = max_soft_tokens
        self._validate_components()
        self.processor.video_processor.max_soft_tokens = max_soft_tokens
        self.feature_size = int(self.model.config.text_config.hidden_size)
        self.last_prepared_input: PreparedGemmaInput | None = None

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        *,
        max_soft_tokens: int = 70,
    ) -> "GemmaKeyRecAdapter":
        """Load the same frozen Gemma classes used by the research wrapper."""

        from transformers import AutoModelForMultimodalLM, AutoProcessor

        processor = AutoProcessor.from_pretrained(model_path)
        model = AutoModelForMultimodalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        ).eval()
        return cls(model, processor, max_soft_tokens=max_soft_tokens)

    @property
    def device(self) -> torch.device:
        return torch.device(self.model.device)

    @property
    def dtype(self) -> torch.dtype:
        return self.model.dtype

    @property
    def video_token_id(self) -> int:
        return int(self.model.config.video_token_id)

    @torch.inference_mode()
    def extract_frame(self, frame: Any, timestamp: float) -> TokenPayload:
        """Return one frame's model-facing tokens using the original workaround.

        Gemma's video processor expects more than one frame, so the research
        implementation duplicated a single frame, ran ``embed_vision``, and
        retained the first half of the projected output. This behavior is kept
        exactly, while processor state and the forward hook are now scoped.
        """

        time_value = float(timestamp)
        if not torch.isfinite(torch.tensor(time_value)):
            raise ValueError("timestamp must be finite")
        duplicated = [frame, frame]
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": duplicated},
                    {"type": "text", "text": ""},
                ],
            }
        ]
        projected: list[torch.Tensor] = []

        def stop_after_projection(_module, _inputs, output) -> None:
            if not isinstance(output, torch.Tensor):
                raise TypeError("Gemma embed_vision output must be a tensor")
            projected.append(output.detach())
            raise _StopVisionForward

        with _temporary_attribute(
            self.processor.video_processor,
            "num_frames",
            len(duplicated),
        ):
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = _move_inputs(inputs, self.device, floating_dtype=torch.bfloat16)
            handle = self.model.model.embed_vision.register_forward_hook(
                stop_after_projection
            )
            try:
                self.model.forward(**inputs)
            except _StopVisionForward:
                pass
            finally:
                handle.remove()

        if not projected:
            raise RuntimeError("Gemma embed_vision hook was not triggered")
        features = projected[0]
        if features.ndim == 3:
            features = features.reshape(-1, features.shape[-1])
        if features.ndim != 2:
            raise ValueError("Gemma projected features must have rank two or three")
        if features.shape[0] < 2 or features.shape[0] % 2 != 0:
            raise ValueError("duplicated-frame output must contain two equal token halves")
        features = features[: features.shape[0] // 2]
        if features.shape[1] != self.feature_size:
            raise ValueError("Gemma projected feature size disagrees with text hidden size")
        times = torch.full(
            (features.shape[0],),
            time_value,
            device=features.device,
            dtype=torch.float32,
        )
        return TokenPayload(features, times)

    @torch.inference_mode()
    def encode_query(self, question: str) -> torch.Tensor:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")
        inputs = self.processor(text=question, return_tensors="pt")
        try:
            input_ids = inputs["input_ids"].to(self.device)
        except (KeyError, TypeError, AttributeError) as exc:
            raise ValueError("processor must return tensor input_ids") from exc
        embeddings = self.model.model.language_model.embed_tokens(input_ids)
        query = embeddings.mean(dim=1).squeeze(0)
        if query.ndim != 1 or query.shape[0] != self.feature_size:
            raise ValueError("Gemma query embedding has an invalid feature size")
        return query

    @torch.inference_mode()
    def generate_router_output(self, prompt: str, max_new_tokens: int) -> str:
        """Run the same Gemma model as the original text-only router."""

        _positive_integer(max_new_tokens, "max_new_tokens")
        message = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        inputs = self._text_chat_inputs(message)
        input_length = inputs["input_ids"].shape[-1]
        outputs = self.model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            eos_token_id=self.processor.tokenizer.eos_token_id,
        )
        generated = outputs[0][input_length:]
        return self.processor.decode(generated, skip_special_tokens=True)

    def prepare_generation(
        self,
        question: str,
        readout: ReadoutResult,
    ) -> PreparedGemmaInput:
        """Preserve the existing neutral interleaved-segment serialization."""

        if not isinstance(readout, ReadoutResult):
            raise TypeError("readout must be a ReadoutResult")
        if not readout.segments:
            raise ValueError("readout must contain at least one memory segment")

        markers = [
            f"<<<KEYREC_VISUAL_SEGMENT_{index}>>>"
            for index in range(len(readout.segments))
        ]
        lines = ["Selected visual memory segments:"]
        for marker, segment in zip(markers, readout.segments):
            lines.append(f"{segment.label} ({segment.token_count} visual tokens):")
            lines.append(marker)
        lines.extend(["Question:", question])
        prompt = "\n".join(lines)
        message = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        text_inputs = self._text_chat_inputs(message)
        input_ids = text_inputs["input_ids"][0].tolist()

        visual_parts: list[torch.Tensor] = []
        cursor = 0
        for marker, segment in zip(markers, readout.segments):
            payload = segment.payload
            if payload.feature_size != self.feature_size:
                raise ValueError("memory segment feature size is incompatible with Gemma")
            if payload.positions is not None:
                raise ValueError("Gemma memory segments must not carry spatial positions")
            marker_ids = self.processor.tokenizer(
                marker,
                add_special_tokens=False,
            ).input_ids
            marker_position = _find_subsequence(input_ids, marker_ids, cursor)
            if marker_position < 0:
                raise RuntimeError(
                    f"could not locate Gemma segment marker {marker!r}"
                )
            replacement = [self.video_token_id] * segment.token_count
            input_ids = (
                input_ids[:marker_position]
                + replacement
                + input_ids[marker_position + len(marker_ids) :]
            )
            cursor = marker_position + segment.token_count
            visual_parts.append(payload.tokens.to(self.device))

        visual_tokens = torch.cat(visual_parts, dim=0)
        prepared = PreparedGemmaInput(
            input_ids=torch.tensor(
                [input_ids],
                dtype=torch.long,
                device=self.device,
            ),
            visual_tokens=visual_tokens,
            video_token_id=self.video_token_id,
            prompt=prompt,
        )
        if prepared.visual_tokens.shape[1] != self.feature_size:
            raise ValueError("prepared Gemma visual feature size is invalid")
        self.last_prepared_input = prepared
        return prepared

    @torch.inference_mode()
    def generate_answer(
        self,
        question: str,
        readout: ReadoutResult,
        max_new_tokens: int,
    ) -> str:
        """Inject selected features through Gemma's existing video interface."""

        _positive_integer(max_new_tokens, "max_new_tokens")
        prepared = self.prepare_generation(question, readout)
        visual_tokens = prepared.visual_tokens.to(self.device)

        def replacement_get_video_features(*_args, **_kwargs) -> _VideoFeatureOutput:
            return _VideoFeatureOutput(pooler_output=visual_tokens.unsqueeze(0))

        vision_owner = self.model.model
        with _temporary_attribute(
            vision_owner,
            "get_video_features",
            replacement_get_video_features,
        ):
            dummy_videos = torch.zeros(
                1,
                1,
                3,
                224,
                224,
                dtype=self.dtype,
                device=self.device,
            )
            outputs = self.model.generate(
                input_ids=prepared.input_ids,
                pixel_values_videos=dummy_videos,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
            )
        if not isinstance(outputs, torch.Tensor) or outputs.ndim != 2:
            raise ValueError("Gemma generate must return rank-2 token ids")
        generated = outputs[0][prepared.input_ids.shape[-1] :]
        return self.processor.decode(generated, skip_special_tokens=True)

    def _text_chat_inputs(self, message: list[dict[str, Any]]) -> Any:
        kwargs = dict(
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        )
        try:
            inputs = self.processor.apply_chat_template(message, **kwargs)
        except TypeError:
            kwargs.pop("enable_thinking")
            inputs = self.processor.apply_chat_template(message, **kwargs)
        inputs = _move_inputs(inputs, self.device)
        try:
            input_ids = inputs["input_ids"]
        except (KeyError, TypeError) as exc:
            raise ValueError("chat template must return input_ids") from exc
        if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
            raise ValueError("chat-template input_ids must be a rank-2 tensor")
        return inputs

    def _validate_components(self) -> None:
        required_model = ("model", "config", "generate", "forward", "device", "dtype")
        missing_model = [name for name in required_model if not hasattr(self.model, name)]
        if missing_model:
            raise TypeError(f"model is missing required attributes: {missing_model}")
        if not hasattr(self.model.model, "embed_vision"):
            raise TypeError("model.model must expose embed_vision")
        if not hasattr(self.model.model, "get_video_features"):
            raise TypeError("model.model must expose get_video_features")
        if not hasattr(self.model.model, "language_model"):
            raise TypeError("model.model must expose language_model")
        required_processor = ("video_processor", "tokenizer", "apply_chat_template")
        missing_processor = [
            name for name in required_processor if not hasattr(self.processor, name)
        ]
        if missing_processor:
            raise TypeError(
                f"processor is missing required attributes: {missing_processor}"
            )


@contextmanager
def _temporary_attribute(obj: Any, name: str, value: Any) -> Iterator[None]:
    """Temporarily replace one attribute and restore exact prior state."""

    existed = hasattr(obj, name)
    previous = getattr(obj, name, None)
    setattr(obj, name, value)
    try:
        yield
    finally:
        if existed:
            setattr(obj, name, previous)
        else:
            delattr(obj, name)


def _move_inputs(
    inputs: Any,
    device: torch.device,
    *,
    floating_dtype: torch.dtype | None = None,
) -> Any:
    if hasattr(inputs, "to"):
        inputs = inputs.to(device)
    elif isinstance(inputs, dict):
        inputs = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in inputs.items()
        }
    else:
        raise TypeError("processor inputs must be a mapping or expose to(device)")
    if floating_dtype is not None:
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor) and value.is_floating_point():
                inputs[key] = value.to(dtype=floating_dtype)
    return inputs


def _find_subsequence(sequence: list[int], pattern: list[int], start: int) -> int:
    if not pattern:
        return -1
    final = len(sequence) - len(pattern)
    for index in range(start, final + 1):
        if sequence[index : index + len(pattern)] == pattern:
            return index
    return -1


def _positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
