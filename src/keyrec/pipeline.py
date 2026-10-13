"""Architecture-neutral orchestration for bounded-memory KeyRec.

The pipeline owns algorithm sequencing, while a model adapter owns extraction,
question embedding, routing inference, and final visual-token injection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import torch

from .encoder import KeyRecMemoryWriter, MemoryWriteConfig
from .memory import MemoryUpdate, TokenPayload
from .readout import (
    RELEVANCE_STATES,
    QueryAdaptiveReadout,
    ReadoutConfig,
    ReadoutResult,
    RouterDecision,
    RouterState,
    build_router_prompt,
    parse_router_output,
)


class KeyRecModelAdapter(Protocol):
    """Narrow model-facing operations required by :class:`KeyRecPipeline`."""

    def extract_frame(self, frame: Any, timestamp: float) -> TokenPayload: ...

    def generate_router_output(self, prompt: str, max_new_tokens: int) -> str: ...

    def encode_query(self, question: str) -> torch.Tensor: ...

    def generate_answer(
        self,
        question: str,
        readout: ReadoutResult,
        max_new_tokens: int,
    ) -> str: ...


@dataclass(frozen=True)
class PipelineConfig:
    """One internally consistent write/read configuration."""

    profile_name: str
    memory_write: MemoryWriteConfig
    readout: ReadoutConfig
    router_max_new_tokens: int = 64
    answer_max_new_tokens: int = 512

    def __post_init__(self) -> None:
        _positive_integer(self.router_max_new_tokens, "router_max_new_tokens")
        _positive_integer(self.answer_max_new_tokens, "answer_max_new_tokens")
        if self.memory_write.top_m != self.readout.tokens_per_event:
            raise ValueError("write and readout event sizes must match")
        if (
            self.memory_write.recent_cache_capacity
            != self.readout.total_budget
        ):
            raise ValueError("write recent capacity and readout budget must match")
        if (
            self.memory_write.event_memory.capacity
            != self.readout.event_capacity
        ):
            raise ValueError("write and readout event capacities must match")

    @classmethod
    def from_main_result(
        cls,
        config: Any,
        profile_name: str,
        *,
        router_max_new_tokens: int = 64,
        answer_max_new_tokens: int = 512,
    ) -> "PipelineConfig":
        return cls(
            profile_name=profile_name,
            memory_write=MemoryWriteConfig.from_main_result(config, profile_name),
            readout=ReadoutConfig.from_main_result(config, profile_name),
            router_max_new_tokens=router_max_new_tokens,
            answer_max_new_tokens=answer_max_new_tokens,
        )


@dataclass(frozen=True)
class PipelineResult:
    """Response plus inspectable, deterministic intermediate artifacts."""

    response: str
    decision: RouterDecision
    readout: ReadoutResult
    frame_updates: tuple[MemoryUpdate, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.response, str):
            raise TypeError("response must be a string")


class KeyRecPipeline:
    """Run one video's frame writes and one query without model coupling."""

    def __init__(self, adapter: KeyRecModelAdapter, config: PipelineConfig) -> None:
        if not isinstance(config, PipelineConfig):
            raise TypeError("config must be a PipelineConfig")
        self.adapter = adapter
        self.config = config
        self.writer = KeyRecMemoryWriter(config.memory_write)
        self.readout = QueryAdaptiveReadout(config.readout)
        self._frame_updates: list[MemoryUpdate] = []

    @property
    def frame_updates(self) -> tuple[MemoryUpdate, ...]:
        return tuple(self._frame_updates)

    def process_frames(
        self,
        frames: Sequence[Any],
        timestamps: Sequence[float],
    ) -> tuple[MemoryUpdate, ...]:
        if len(frames) == 0:
            raise ValueError("frames must not be empty")
        if len(frames) != len(timestamps):
            raise ValueError("frames and timestamps must have equal length")
        parsed_times = [float(value) for value in timestamps]
        if not torch.isfinite(torch.tensor(parsed_times)).all():
            raise ValueError("timestamps must be finite")
        if any(right < left for left, right in zip(parsed_times, parsed_times[1:])):
            raise ValueError("timestamps must be chronological")

        for frame, timestamp in zip(frames, parsed_times):
            payload = self.adapter.extract_frame(frame, timestamp)
            if not isinstance(payload, TokenPayload):
                raise TypeError("extract_frame must return a TokenPayload")
            _candidate, update = self.writer.write_frame(
                payload.tokens,
                payload.times,
                payload.positions,
            )
            self._frame_updates.append(update)
        return self.frame_updates

    def answer(
        self,
        question: str,
        *,
        router_output: str | None = None,
        fixed_state: RouterState | None = None,
    ) -> PipelineResult:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")
        if not self._frame_updates:
            raise RuntimeError("process at least one frame before answering")
        if router_output is not None and fixed_state is not None:
            raise ValueError("provide router_output or fixed_state, not both")

        if fixed_state is not None:
            if not isinstance(fixed_state, RouterState):
                raise TypeError("fixed_state must be a RouterState")
            decision = RouterDecision(state=fixed_state, raw_output="")
        else:
            if router_output is None:
                prompt = build_router_prompt(question, self.config.readout.regime)
                router_output = self.adapter.generate_router_output(
                    prompt,
                    self.config.router_max_new_tokens,
                )
            decision = parse_router_output(
                router_output,
                self.config.readout.fallback_state,
            )

        query_route = None
        if (
            decision.state in RELEVANCE_STATES
            and self.config.readout.event_slots(decision.state) > 0
            and len(self.writer.event_bank) > 0
        ):
            query_route = self.adapter.encode_query(question)
        selected = self.readout.read(
            self.writer.event_bank,
            self.writer.recent_cache,
            decision,
            query_route,
        )
        if selected.token_count == 0:
            raise RuntimeError("KeyRec readout returned no visual tokens")
        response = self.adapter.generate_answer(
            question,
            selected,
            self.config.answer_max_new_tokens,
        )
        return PipelineResult(
            response=response,
            decision=decision,
            readout=selected,
            frame_updates=self.frame_updates,
        )


def _positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
