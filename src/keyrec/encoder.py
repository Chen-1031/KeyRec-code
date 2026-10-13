"""Per-frame candidate construction for the bounded-memory memory core."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from .memory import (
    BoundedEventBank,
    EventCandidate,
    EventMemoryConfig,
    MemoryUpdate,
    RecentCache,
    TokenPayload,
)


def novelty_against_routes(
    tokens: torch.Tensor,
    stored_routes: torch.Tensor | None,
) -> torch.Tensor:
    """Compute novelty against every route retained before this frame."""

    _validate_frame_tokens(tokens)
    if stored_routes is None or stored_routes.numel() == 0:
        return torch.ones(tokens.shape[0], device=tokens.device, dtype=torch.float32)
    if stored_routes.ndim != 2:
        raise ValueError("stored_routes must be rank 2 [event, feature]")
    if stored_routes.shape[1] != tokens.shape[1]:
        raise ValueError("stored routes and tokens must have the same feature size")
    if not stored_routes.is_floating_point():
        raise ValueError("stored_routes must use a floating-point dtype")
    if not torch.isfinite(stored_routes).all():
        raise ValueError("stored_routes must contain only finite values")
    token_features = F.normalize(tokens.float(), p=2, dim=1)
    route_features = F.normalize(
        stored_routes.to(device=tokens.device).float(), p=2, dim=1
    )
    similarities = token_features @ route_features.T
    return 1.0 - similarities.max(dim=1).values


def build_event_candidate(
    tokens: torch.Tensor,
    times: torch.Tensor,
    stored_routes: torch.Tensor | None,
    top_m: int,
    positions: torch.Tensor | None = None,
) -> EventCandidate:
    """Build one event directly from the frame's deterministic top-``m`` set."""

    if isinstance(top_m, bool) or not isinstance(top_m, int) or top_m <= 0:
        raise ValueError("top_m must be a positive integer")
    frame = TokenPayload(tokens, times, positions)
    if len(frame) == 0:
        raise ValueError("a frame must contain at least one visual token")
    novelty = novelty_against_routes(frame.tokens, stored_routes)
    selected_count = min(top_m, len(frame))
    # Stable sorting gives lower source indexes precedence when novelty ties.
    novelty_order = torch.argsort(novelty, descending=True, stable=True)
    selected = novelty_order[:selected_count]
    selected_novelty = novelty[selected]

    weights = torch.softmax(selected_novelty, dim=0)
    route_sum = (weights.unsqueeze(1) * frame.tokens[selected].float()).sum(dim=0)
    if float(torch.linalg.vector_norm(route_sum).item()) == 0.0:
        raise ValueError("top-m tokens produce a zero-norm candidate route")
    route = F.normalize(route_sum, p=2, dim=0)

    # Payloads preserve source order; the very same index set supplies route,
    # payload, and mean-novelty utility for all candidate statistics.
    source_order = selected.sort().values
    payload = frame.select(source_order)
    return EventCandidate(
        route=route,
        payload=payload,
        utility=float(selected_novelty.mean().item()),
    )


@dataclass(frozen=True)
class MemoryWriteConfig:
    """Complete write-side settings derived from one validated main profile."""

    top_m: int
    recent_cache_capacity: int
    event_memory: EventMemoryConfig

    def __post_init__(self) -> None:
        if isinstance(self.top_m, bool) or not isinstance(self.top_m, int) or self.top_m <= 0:
            raise ValueError("top_m must be a positive integer")
        if (
            isinstance(self.recent_cache_capacity, bool)
            or not isinstance(self.recent_cache_capacity, int)
            or self.recent_cache_capacity <= 0
        ):
            raise ValueError("recent_cache_capacity must be a positive integer")
        if self.top_m != self.event_memory.tokens_per_event:
            raise ValueError("top_m must equal event-memory tokens_per_event")

    @classmethod
    def from_main_result(cls, config: Any, profile_name: str) -> "MemoryWriteConfig":
        """Consume values from ``MainResultConfig`` without parallel defaults."""

        validate = getattr(config, "validate", None)
        if not callable(validate):
            raise TypeError("config must be a validated MainResultConfig")
        validate()
        try:
            profile = config.profiles[profile_name]
            event_update = config.event_update
        except (AttributeError, KeyError, TypeError) as exc:
            raise ValueError(f"unknown main-result profile {profile_name!r}") from exc
        if profile.chunk_size != 1:
            raise ValueError("bounded event memory requires one frame per update")
        if profile.top_m_novel_tokens != profile.event_payload_tokens:
            raise ValueError("main-result profile must construct candidates directly at m")
        if profile.recent_cache_tokens != profile.readout_budget_tokens:
            raise ValueError("main-result profile must use N_C=B")
        return cls(
            top_m=profile.top_m_novel_tokens,
            recent_cache_capacity=profile.recent_cache_tokens,
            event_memory=EventMemoryConfig(
                capacity=profile.event_capacity,
                tokens_per_event=profile.event_payload_tokens,
                merge_similarity=event_update["merge_similarity"],
                max_merge_gap_seconds=event_update["merge_max_gap_seconds"],
                route_update_weight=event_update["route_moving_average_weight"],
                temporal_coverage_weight=profile.temporal_coverage_weight,
            ),
        )


class KeyRecMemoryWriter:
    """Own the recent cache and event bank updated once for each frame."""

    def __init__(self, config: MemoryWriteConfig) -> None:
        if not isinstance(config, MemoryWriteConfig):
            raise TypeError("config must be a MemoryWriteConfig")
        self.config = config
        self.recent_cache = RecentCache(config.recent_cache_capacity)
        self.event_bank = BoundedEventBank(config.event_memory)

    def write_frame(
        self,
        tokens: torch.Tensor,
        times: torch.Tensor,
        positions: torch.Tensor | None = None,
    ) -> tuple[EventCandidate, MemoryUpdate]:
        """Append dense recent tokens and incorporate exactly one event candidate."""

        frame = TokenPayload(tokens, times, positions)
        if len(frame) == 0:
            raise ValueError("a frame must contain at least one visual token")
        self.recent_cache.validate_append(frame)
        candidate = build_event_candidate(
            frame.tokens,
            frame.times,
            self.event_bank.route_matrix(device=frame.tokens.device),
            self.config.top_m,
            frame.positions,
        )
        update = self.event_bank.update(candidate)
        self.recent_cache.append(frame)
        return candidate, update


def _validate_frame_tokens(tokens: torch.Tensor) -> None:
    if not isinstance(tokens, torch.Tensor) or tokens.ndim != 2:
        raise ValueError("tokens must be rank 2 [token, feature]")
    if tokens.shape[0] == 0:
        raise ValueError("tokens must contain at least one visual token")
    if not tokens.is_floating_point():
        raise ValueError("tokens must use a floating-point dtype")
    if not torch.isfinite(tokens).all():
        raise ValueError("tokens must contain only finite values")
