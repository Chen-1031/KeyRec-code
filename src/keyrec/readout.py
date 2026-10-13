"""Query-adaptive fixed-budget readout for bounded-memory KeyRec memory.

This module chooses complete stored-event payloads and a recent-token tail without
mutating either memory source. Model-specific query embedding and visual-token
injection belong to adapter layers.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal

import torch
import torch.nn.functional as F

from .memory import BoundedEventBank, RecentCache, StoredEvent, TokenPayload


class RouterState(str, Enum):
    """The five ordered memory-allocation states for recent/event allocation."""

    ALL_RECENT = "all_recent"
    HEAVY_RECENT = "heavy_recent"
    BALANCED = "balanced"
    HEAVY_KEY_EVENT = "heavy_key_event"
    ALL_KEY_EVENT = "all_key_event"


ROUTER_STATES = tuple(RouterState)
RELEVANCE_STATES = frozenset(
    {RouterState.HEAVY_RECENT, RouterState.BALANCED}
)
COVERAGE_STATES = frozenset(
    {RouterState.HEAVY_KEY_EVENT, RouterState.ALL_KEY_EVENT}
)


@dataclass(frozen=True)
class RouterDecision:
    """A strict router parse result with one explicit fallback path."""

    state: RouterState
    raw_output: str
    reason: str = ""
    used_fallback: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.state, RouterState):
            raise TypeError("state must be a RouterState")
        if not isinstance(self.raw_output, str) or not isinstance(self.reason, str):
            raise TypeError("raw_output and reason must be strings")
        if not isinstance(self.used_fallback, bool):
            raise TypeError("used_fallback must be a bool")


def parse_router_output(
    output: str,
    fallback: RouterState = RouterState.BALANCED,
) -> RouterDecision:
    """Parse strict JSON router output, otherwise use ``fallback``.

    A surrounding Markdown JSON fence is tolerated, but state aliases and the
    unsupported spellings are intentionally not.
    """

    if not isinstance(fallback, RouterState):
        raise TypeError("fallback must be a RouterState")
    raw = output if isinstance(output, str) else ""
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        parsed = json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        parsed = None

    if isinstance(parsed, dict):
        value = parsed.get("state", parsed.get("category"))
        try:
            state = RouterState(value)
        except (ValueError, TypeError):
            state = None
        if state is not None:
            reason = parsed.get("reason", "")
            return RouterDecision(
                state=state,
                raw_output=raw,
                reason=reason.strip() if isinstance(reason, str) else "",
            )
    return RouterDecision(
        state=fallback,
        raw_output=raw,
        used_fallback=True,
    )


def build_router_prompt(question: str, regime: str) -> str:
    """Build the architecture-neutral, question-only router prompt."""

    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    if regime == "streaming":
        recent_description = "the detailed latest observations before the query"
    elif regime == "long_video":
        recent_description = "the detailed ending portion of the complete video"
    else:
        raise ValueError("regime must be 'streaming' or 'long_video'")
    return f"""You route a question between two bounded visual-memory sources.
Recent memory contains {recent_description}.
Event memory contains compact historical events distributed over time.

Choose exactly one state:
- all_recent: only detailed recent evidence is needed
- heavy_recent: mostly recent evidence with a few relevant events
- balanced: both recent evidence and relevant historical events
- heavy_key_event: broad historical coverage with a small recent tail
- all_key_event: complete budget devoted to temporal event coverage

Return only JSON: {{"state": "<state>", "reason": "<short reason>"}}
Question: {question.strip()}"""


@dataclass(frozen=True)
class ReadoutConfig:
    """Budget and allocation codebook for one canonical main-result profile."""

    profile_name: str
    regime: str
    total_budget: int
    tokens_per_event: int
    event_capacity: int
    event_allocations: tuple[tuple[RouterState, int], ...]
    fallback_state: RouterState = RouterState.BALANCED

    def __post_init__(self) -> None:
        _positive_integer(self.total_budget, "total_budget")
        _positive_integer(self.tokens_per_event, "tokens_per_event")
        _positive_integer(self.event_capacity, "event_capacity")
        if self.regime not in {"streaming", "long_video"}:
            raise ValueError("regime must be 'streaming' or 'long_video'")
        if not isinstance(self.fallback_state, RouterState):
            raise TypeError("fallback_state must be a RouterState")
        states = tuple(state for state, _ in self.event_allocations)
        if states != ROUTER_STATES:
            raise ValueError("event_allocations must contain all canonical states in order")
        for state, count in self.event_allocations:
            if not isinstance(state, RouterState):
                raise TypeError("allocation keys must be RouterState values")
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError("event allocations must be non-negative integers")
            if count > self.event_capacity:
                raise ValueError("event allocation exceeds event capacity")
            if count * self.tokens_per_event > self.total_budget:
                raise ValueError("event allocation exceeds the readout budget")

    def event_slots(self, state: RouterState) -> int:
        if not isinstance(state, RouterState):
            raise TypeError("state must be a RouterState")
        return dict(self.event_allocations)[state]

    @classmethod
    def from_main_result(cls, config: Any, profile_name: str) -> "ReadoutConfig":
        """Consume a validated ``MainResultConfig`` without alternate defaults."""

        validate = getattr(config, "validate", None)
        if not callable(validate):
            raise TypeError("config must be a validated MainResultConfig")
        validate()
        try:
            profile = config.profiles[profile_name]
        except (AttributeError, KeyError, TypeError) as exc:
            raise ValueError(f"unknown main-result profile {profile_name!r}") from exc
        try:
            allocations = tuple(
                (RouterState(state), count)
                for state, count in profile.event_allocations.items()
            )
        except (ValueError, TypeError) as exc:
            raise ValueError("profile contains a non-canonical router state") from exc
        return cls(
            profile_name=profile_name,
            regime=profile.regime,
            total_budget=profile.readout_budget_tokens,
            tokens_per_event=profile.event_payload_tokens,
            event_capacity=profile.event_capacity,
            event_allocations=allocations,
        )


@dataclass(frozen=True)
class MemorySegment:
    """One complete, neutrally labelled memory block for model presentation."""

    ordinal: int
    source: Literal["event", "recent"]
    payload: TokenPayload
    event_id: int | None = None

    def __post_init__(self) -> None:
        _positive_integer(self.ordinal, "ordinal")
        if not isinstance(self.payload, TokenPayload):
            raise TypeError("payload must be a TokenPayload")
        if self.source not in {"event", "recent"}:
            raise ValueError("source must be 'event' or 'recent'")
        if len(self.payload) == 0:
            raise ValueError("memory segments cannot be empty")
        if self.source == "event" and self.event_id is None:
            raise ValueError("event segments require an event_id")
        if self.event_id is not None and (
            isinstance(self.event_id, bool)
            or not isinstance(self.event_id, int)
            or self.event_id < 0
        ):
            raise ValueError("event_id must be a non-negative integer")
        if self.source == "recent" and self.event_id is not None:
            raise ValueError("recent segments cannot have an event_id")

    @property
    def label(self) -> str:
        return f"Memory segment {self.ordinal}"

    @property
    def token_count(self) -> int:
        return len(self.payload)


@dataclass(frozen=True)
class ReadoutResult:
    """Immutable evidence selection and explicit budget accounting."""

    decision: RouterDecision
    requested_event_slots: int
    segments: tuple[MemorySegment, ...]
    total_budget: int

    def __post_init__(self) -> None:
        if not isinstance(self.decision, RouterDecision):
            raise TypeError("decision must be a RouterDecision")
        if (
            isinstance(self.requested_event_slots, bool)
            or not isinstance(self.requested_event_slots, int)
            or self.requested_event_slots < 0
        ):
            raise ValueError("requested_event_slots must be a non-negative integer")
        _positive_integer(self.total_budget, "total_budget")
        if not isinstance(self.segments, tuple) or not all(
            isinstance(segment, MemorySegment) for segment in self.segments
        ):
            raise TypeError("segments must be a tuple of MemorySegment values")
        if tuple(segment.ordinal for segment in self.segments) != tuple(
            range(1, len(self.segments) + 1)
        ):
            raise ValueError("segment ordinals must be contiguous from one")
        if self.token_count > self.total_budget:
            raise ValueError("segments exceed total_budget")

    @property
    def token_count(self) -> int:
        return sum(segment.token_count for segment in self.segments)

    @property
    def unused_budget(self) -> int:
        return self.total_budget - self.token_count

    @property
    def selected_event_ids(self) -> tuple[int, ...]:
        return tuple(
            segment.event_id
            for segment in self.segments
            if segment.source == "event" and segment.event_id is not None
        )


class QueryAdaptiveReadout:
    """Select complete event blocks and fill the residual budget from recent."""

    def __init__(self, config: ReadoutConfig) -> None:
        if not isinstance(config, ReadoutConfig):
            raise TypeError("config must be a ReadoutConfig")
        self.config = config

    def read_from_router_output(
        self,
        event_bank: BoundedEventBank,
        recent_cache: RecentCache,
        router_output: str,
        query_route: torch.Tensor | None = None,
    ) -> ReadoutResult:
        decision = parse_router_output(router_output, self.config.fallback_state)
        return self.read(event_bank, recent_cache, decision, query_route)

    def read(
        self,
        event_bank: BoundedEventBank,
        recent_cache: RecentCache,
        decision: RouterDecision,
        query_route: torch.Tensor | None = None,
    ) -> ReadoutResult:
        """Read without mutation, filling unused event blocks from recent.

        Allocations specify a maximum number of complete event blocks. If the
        bank is sparse or a complete payload contains fewer than ``m`` tokens,
        newest recent tokens may fill the residual budget. If both sources are
        insufficient, ``ReadoutResult.unused_budget`` remains positive.
        """

        self._validate_memory(event_bank, recent_cache)
        if not isinstance(decision, RouterDecision):
            raise TypeError("decision must be a RouterDecision")
        slots = self.config.event_slots(decision.state)
        selected = self._select_events(
            event_bank.events,
            slots,
            decision.state,
            query_route,
        )
        event_tokens = sum(len(event.payload) for event in selected)
        if event_tokens > self.config.total_budget:
            raise ValueError("selected event payloads exceed the readout budget")

        segments: list[MemorySegment] = []
        for event in selected:
            segments.append(
                MemorySegment(
                    ordinal=len(segments) + 1,
                    source="event",
                    payload=event.payload,
                    event_id=event.event_id,
                )
            )

        recent_limit = self.config.total_budget - event_tokens
        recent = recent_cache.payload
        if recent is not None and recent_limit > 0 and len(recent) > 0:
            count = min(recent_limit, len(recent))
            indexes = torch.arange(
                len(recent) - count,
                len(recent),
                device=recent.tokens.device,
            )
            segments.append(
                MemorySegment(
                    ordinal=len(segments) + 1,
                    source="recent",
                    payload=recent.select(indexes),
                )
            )

        result = ReadoutResult(
            decision=decision,
            requested_event_slots=slots,
            segments=tuple(segments),
            total_budget=self.config.total_budget,
        )
        if result.token_count > self.config.total_budget:
            raise AssertionError("readout exceeded its validated token budget")
        return result

    def _validate_memory(
        self,
        event_bank: BoundedEventBank,
        recent_cache: RecentCache,
    ) -> None:
        if not isinstance(event_bank, BoundedEventBank):
            raise TypeError("event_bank must be a BoundedEventBank")
        if not isinstance(recent_cache, RecentCache):
            raise TypeError("recent_cache must be a RecentCache")
        if event_bank.config.tokens_per_event != self.config.tokens_per_event:
            raise ValueError("event bank and readout tokens_per_event must match")
        if event_bank.config.capacity != self.config.event_capacity:
            raise ValueError("event bank and readout capacities must match")
        if recent_cache.capacity != self.config.total_budget:
            raise ValueError("canonical readout requires recent-cache capacity N_C=B")

    def _select_events(
        self,
        events: tuple[StoredEvent, ...],
        requested: int,
        state: RouterState,
        query_route: torch.Tensor | None,
    ) -> list[StoredEvent]:
        if requested == 0 or not events:
            return []
        count = min(requested, len(events))
        if state in RELEVANCE_STATES:
            selected = _select_relevant(events, count, query_route)
        elif state in COVERAGE_STATES:
            selected = _select_coverage(events, count)
        else:
            raise ValueError(f"state {state.value!r} cannot request event slots")
        return sorted(selected, key=lambda event: (event.center_time, event.event_id))


def _select_relevant(
    events: tuple[StoredEvent, ...],
    count: int,
    query_route: torch.Tensor | None,
) -> list[StoredEvent]:
    if query_route is None:
        raise ValueError("relevance-oriented readout requires query_route")
    if not isinstance(query_route, torch.Tensor) or query_route.ndim != 1:
        raise ValueError("query_route must be a rank-1 tensor")
    if not query_route.is_floating_point() or not torch.isfinite(query_route).all():
        raise ValueError("query_route must contain finite floating-point values")
    if query_route.shape[0] != events[0].route.shape[0]:
        raise ValueError("query_route and event routes must have the same feature size")
    if float(torch.linalg.vector_norm(query_route.float()).item()) == 0.0:
        raise ValueError("query_route must have non-zero norm")
    routes = torch.stack(
        [event.route.to(device=query_route.device) for event in events], dim=0
    )
    query = F.normalize(query_route.float(), p=2, dim=0)
    scores = F.normalize(routes.float(), p=2, dim=1) @ query
    ranking = sorted(
        range(len(events)),
        key=lambda index: (-float(scores[index].item()), events[index].event_id),
    )
    return [events[index] for index in ranking[:count]]


def _select_coverage(
    events: tuple[StoredEvent, ...],
    count: int,
) -> list[StoredEvent]:
    ordered = sorted(events, key=lambda event: (event.center_time, event.event_id))
    if count >= len(ordered):
        return ordered
    indexes = _uniform_indexes(len(ordered), count)
    return [ordered[index] for index in indexes]


def _uniform_indexes(length: int, count: int) -> list[int]:
    _positive_integer(length, "length")
    _positive_integer(count, "count")
    if count >= length:
        return list(range(length))
    if count == 1:
        return [0]
    denominator = count - 1
    return [
        (index * (length - 1) + denominator // 2) // denominator
        for index in range(count)
    ]


def _positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
