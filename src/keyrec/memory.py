"""Deterministic, bounded storage for bounded-memory KeyRec events.

This module owns query-agnostic storage, independently of model APIs,
prompts, questions, and readout policies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F


def _require_positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _require_finite_number(value: float, name: str) -> float:
    result = float(value)
    if not torch.isfinite(torch.tensor(result)):
        raise ValueError(f"{name} must be finite")
    return result


def _normalized_route(route: torch.Tensor, feature_size: int) -> torch.Tensor:
    if not isinstance(route, torch.Tensor) or route.ndim != 1:
        raise ValueError("route must be a rank-1 tensor")
    if route.shape[0] != feature_size:
        raise ValueError("route and payload feature dimensions must match")
    if not route.is_floating_point():
        raise ValueError("route must use a floating-point dtype")
    if not torch.isfinite(route).all():
        raise ValueError("route must contain only finite values")
    if float(torch.linalg.vector_norm(route.float()).item()) == 0.0:
        raise ValueError("route must have non-zero norm")
    return F.normalize(route.detach(), p=2, dim=0)


@dataclass(frozen=True)
class TokenPayload:
    """Visual tokens and metadata aligned along their leading dimension."""

    tokens: torch.Tensor
    times: torch.Tensor
    positions: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.tokens, torch.Tensor) or self.tokens.ndim != 2:
            raise ValueError("tokens must be a rank-2 tensor [token, feature]")
        if not self.tokens.is_floating_point():
            raise ValueError("tokens must use a floating-point dtype")
        if not torch.isfinite(self.tokens).all():
            raise ValueError("tokens must contain only finite values")
        if not isinstance(self.times, torch.Tensor) or self.times.ndim != 1:
            raise ValueError("times must be a rank-1 tensor")
        if self.times.shape[0] != self.tokens.shape[0]:
            raise ValueError("times must have one entry per token")
        if not self.times.is_floating_point():
            raise ValueError("times must use a floating-point dtype")
        if not torch.isfinite(self.times).all():
            raise ValueError("times must contain only finite values")
        if self.positions is not None:
            if not isinstance(self.positions, torch.Tensor) or self.positions.ndim < 1:
                raise ValueError("positions must have a token dimension")
            if self.positions.shape[0] != self.tokens.shape[0]:
                raise ValueError("positions must have one entry per token")
            if self.positions.is_floating_point() or self.positions.is_complex():
                raise ValueError("positions must use an integer dtype")

        tokens = self.tokens.detach()
        times = self.times.detach().to(device=tokens.device, dtype=torch.float32)
        positions = self.positions
        if positions is not None:
            positions = positions.detach().to(device=tokens.device, dtype=torch.long)
        object.__setattr__(self, "tokens", tokens)
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "positions", positions)

    def __len__(self) -> int:
        return int(self.tokens.shape[0])

    @property
    def feature_size(self) -> int:
        return int(self.tokens.shape[1])

    @property
    def interval(self) -> tuple[float, float]:
        if len(self) == 0:
            raise ValueError("an empty payload has no temporal interval")
        return float(self.times.min().item()), float(self.times.max().item())

    def select(self, indices: torch.Tensor) -> "TokenPayload":
        if indices.ndim != 1 or indices.dtype != torch.long:
            raise ValueError("indices must be a rank-1 torch.long tensor")
        indices = indices.to(self.tokens.device)
        positions = self.positions[indices] if self.positions is not None else None
        return TokenPayload(self.tokens[indices], self.times[indices], positions)


@dataclass(frozen=True)
class EventCandidate:
    """One frame's top-novelty event proposal from the most novel tokens."""

    route: torch.Tensor
    payload: TokenPayload
    utility: float

    def __post_init__(self) -> None:
        if len(self.payload) == 0:
            raise ValueError("event payload must contain at least one token")
        route = _normalized_route(self.route, self.payload.feature_size)
        if route.device != self.payload.tokens.device:
            raise ValueError("route and payload must be on the same device")
        object.__setattr__(self, "route", route)
        object.__setattr__(self, "utility", _require_finite_number(self.utility, "utility"))

    @property
    def interval(self) -> tuple[float, float]:
        return self.payload.interval


@dataclass(frozen=True)
class StoredEvent:
    """An immutable event record retained by :class:`BoundedEventBank`."""

    event_id: int
    route: torch.Tensor
    payload: TokenPayload
    utility: float

    def __post_init__(self) -> None:
        if isinstance(self.event_id, bool) or not isinstance(self.event_id, int):
            raise ValueError("event_id must be an integer")
        if self.event_id < 0:
            raise ValueError("event_id must be non-negative")
        if len(self.payload) == 0:
            raise ValueError("event payload must contain at least one token")
        route = _normalized_route(self.route, self.payload.feature_size)
        if route.device != self.payload.tokens.device:
            raise ValueError("route and payload must be on the same device")
        object.__setattr__(self, "route", route)
        object.__setattr__(self, "utility", _require_finite_number(self.utility, "utility"))

    @property
    def interval(self) -> tuple[float, float]:
        return self.payload.interval

    @property
    def center_time(self) -> float:
        start, end = self.interval
        return 0.5 * (start + end)


@dataclass(frozen=True)
class EventMemoryConfig:
    """Explicit event-bank parameters from the selected main-result profile."""

    capacity: int
    tokens_per_event: int
    merge_similarity: float
    max_merge_gap_seconds: float
    route_update_weight: float
    temporal_coverage_weight: float

    def __post_init__(self) -> None:
        _require_positive_int(self.capacity, "capacity")
        _require_positive_int(self.tokens_per_event, "tokens_per_event")
        similarity = _require_finite_number(self.merge_similarity, "merge_similarity")
        max_gap = _require_finite_number(
            self.max_merge_gap_seconds, "max_merge_gap_seconds"
        )
        route_weight = _require_finite_number(
            self.route_update_weight, "route_update_weight"
        )
        coverage_weight = _require_finite_number(
            self.temporal_coverage_weight, "temporal_coverage_weight"
        )
        if not -1.0 <= similarity <= 1.0:
            raise ValueError("merge_similarity must be in [-1, 1]")
        if max_gap < 0.0:
            raise ValueError("max_merge_gap_seconds must be non-negative")
        if not 0.0 <= route_weight <= 1.0:
            raise ValueError("route_update_weight must be in [0, 1]")
        if coverage_weight < 0.0:
            raise ValueError("temporal_coverage_weight must be non-negative")


@dataclass(frozen=True)
class MemoryUpdate:
    """Observable result of one deterministic event-bank update."""

    action: Literal["inserted", "merged"]
    target_event_id: int
    evicted_event_id: int | None = None


class RecentCache:
    """Fixed-capacity FIFO tail over model-facing visual tokens."""

    def __init__(self, capacity: int) -> None:
        _require_positive_int(capacity, "capacity")
        self.capacity = capacity
        self._payload: TokenPayload | None = None

    def __len__(self) -> int:
        return 0 if self._payload is None else len(self._payload)

    @property
    def payload(self) -> TokenPayload | None:
        return self._payload

    def validate_append(self, payload: TokenPayload) -> None:
        if not isinstance(payload, TokenPayload):
            raise TypeError("payload must be a TokenPayload")
        if self._payload is None or len(payload) == 0:
            return
        current = self._payload
        if payload.feature_size != current.feature_size:
            raise ValueError("recent-cache feature dimensions must remain constant")
        if payload.tokens.device != current.tokens.device:
            raise ValueError("recent-cache payloads must remain on one device")
        if payload.tokens.dtype != current.tokens.dtype:
            raise ValueError("recent-cache token dtypes must remain constant")
        if (payload.positions is None) != (current.positions is None):
            raise ValueError("recent-cache position metadata cannot change schema")
        if payload.positions is not None and (
            payload.positions.shape[1:] != current.positions.shape[1:]
        ):
            raise ValueError("recent-cache position shapes must remain constant")

    def append(self, payload: TokenPayload) -> None:
        self.validate_append(payload)
        if len(payload) == 0:
            return
        if self._payload is None:
            combined = payload
        else:
            current = self._payload
            positions = None
            if current.positions is not None:
                positions = torch.cat([current.positions, payload.positions], dim=0)
            combined = TokenPayload(
                torch.cat([current.tokens, payload.tokens], dim=0),
                torch.cat([current.times, payload.times], dim=0),
                positions,
            )
        if len(combined) > self.capacity:
            start = len(combined) - self.capacity
            indices = torch.arange(start, len(combined), device=combined.tokens.device)
            combined = combined.select(indices)
        self._payload = combined

    def clear(self) -> None:
        self._payload = None


class BoundedEventBank:
    """Online add--merge--evict event memory with fixed storage bounds."""

    def __init__(self, config: EventMemoryConfig) -> None:
        if not isinstance(config, EventMemoryConfig):
            raise TypeError("config must be an EventMemoryConfig")
        self.config = config
        self._events: list[StoredEvent] = []
        self._next_event_id = 0

    def __len__(self) -> int:
        return len(self._events)

    @property
    def events(self) -> tuple[StoredEvent, ...]:
        return tuple(self._events)

    def route_matrix(self, *, device: torch.device | str | None = None) -> torch.Tensor | None:
        if not self._events:
            return None
        routes = [event.route for event in self._events]
        if device is not None:
            routes = [route.to(device=device) for route in routes]
        return torch.stack(routes, dim=0)

    def update(self, candidate: EventCandidate) -> MemoryUpdate:
        self._validate_candidate(candidate)
        merge_index = self._merge_target(candidate)
        if merge_index is not None:
            event_id = self._events[merge_index].event_id
            self._events[merge_index] = self._merge(self._events[merge_index], candidate)
            return MemoryUpdate(action="merged", target_event_id=event_id)

        event_id = self._next_event_id
        self._next_event_id += 1
        self._events.append(
            StoredEvent(event_id, candidate.route, candidate.payload, candidate.utility)
        )
        evicted_id = self._evict_if_needed()
        return MemoryUpdate(
            action="inserted",
            target_event_id=event_id,
            evicted_event_id=evicted_id,
        )

    def _validate_candidate(self, candidate: EventCandidate) -> None:
        if not isinstance(candidate, EventCandidate):
            raise TypeError("candidate must be an EventCandidate")
        if len(candidate.payload) > self.config.tokens_per_event:
            raise ValueError("candidate payload exceeds tokens_per_event")
        if not self._events:
            return
        first = self._events[0]
        if candidate.payload.feature_size != first.payload.feature_size:
            raise ValueError("event feature dimensions must remain constant")
        if candidate.payload.tokens.device != first.payload.tokens.device:
            raise ValueError("event payloads must remain on one device")
        if candidate.payload.tokens.dtype != first.payload.tokens.dtype:
            raise ValueError("event token dtypes must remain constant")
        if (candidate.payload.positions is None) != (first.payload.positions is None):
            raise ValueError("event position metadata cannot change schema")
        if candidate.payload.positions is not None and (
            candidate.payload.positions.shape[1:] != first.payload.positions.shape[1:]
        ):
            raise ValueError("event position shapes must remain constant")

    def _merge_target(self, candidate: EventCandidate) -> int | None:
        if not self._events:
            return None
        routes = torch.stack(
            [event.route.to(candidate.route.device) for event in self._events], dim=0
        )
        similarities = F.cosine_similarity(candidate.route.float().unsqueeze(0), routes.float())
        # torch.argmax returns the first maximum, so equal similarities resolve
        # to the oldest still-retained event.
        index = int(torch.argmax(similarities).item())
        matched = self._events[index]
        candidate_start, _ = candidate.interval
        _, matched_end = matched.interval
        forward_gap = candidate_start - matched_end
        if forward_gap < 0.0:
            return None
        if float(similarities[index].item()) < self.config.merge_similarity:
            return None
        if forward_gap > self.config.max_merge_gap_seconds:
            return None
        return index

    def _merge(self, stored: StoredEvent, candidate: EventCandidate) -> StoredEvent:
        beta = self.config.route_update_weight
        route = F.normalize(
            (1.0 - beta) * stored.route + beta * candidate.route,
            p=2,
            dim=0,
        )
        payload = _merge_payloads(
            stored.payload,
            candidate.payload,
            self.config.tokens_per_event,
        )
        return StoredEvent(
            event_id=stored.event_id,
            route=route,
            payload=payload,
            utility=max(stored.utility, candidate.utility),
        )

    def _evict_if_needed(self) -> int | None:
        if len(self._events) <= self.config.capacity:
            return None
        scores = [self.retention_score(index) for index in range(len(self._events))]
        # The lower event id (earlier insertion) is evicted on an exact score
        # tie.  This makes repeated streams independent of backend sort order.
        index = min(
            range(len(self._events)),
            key=lambda item: (scores[item], self._events[item].event_id),
        )
        return self._events.pop(index).event_id

    def retention_score(self, index: int) -> float:
        if not 0 <= index < len(self._events):
            raise IndexError("event index out of range")
        event = self._events[index]
        if len(self._events) == 1:
            return event.utility
        nearest_distance = min(
            abs(event.center_time - other.center_time)
            for other_index, other in enumerate(self._events)
            if other_index != index
        )
        return event.utility + self.config.temporal_coverage_weight * nearest_distance


def _merge_payloads(
    first: TokenPayload,
    second: TokenPayload,
    limit: int,
) -> TokenPayload:
    positions = None
    if first.positions is not None:
        positions = torch.cat([first.positions, second.positions], dim=0)
    combined = TokenPayload(
        torch.cat([first.tokens, second.tokens], dim=0),
        torch.cat([first.times, second.times], dim=0),
        positions,
    )
    order = torch.argsort(combined.times, stable=True)
    ordered = combined.select(order)
    if len(ordered) <= limit:
        return ordered
    return ordered.select(_uniform_indices(len(ordered), limit, ordered.tokens.device))


def _uniform_indices(
    length: int,
    count: int,
    device: torch.device,
) -> torch.Tensor:
    """Return endpoint-preserving, integer-defined uniform sample indexes."""

    _require_positive_int(length, "length")
    _require_positive_int(count, "count")
    if count >= length:
        return torch.arange(length, device=device)
    if count == 1:
        return torch.zeros(1, device=device, dtype=torch.long)
    denominator = count - 1
    indexes = [
        (item * (length - 1) + denominator // 2) // denominator
        for item in range(count)
    ]
    return torch.tensor(indexes, device=device, dtype=torch.long)
