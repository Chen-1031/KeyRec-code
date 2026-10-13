"""Validated configuration contract for the supported evaluation profiles."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROUTER_STATES = (
    "all_recent",
    "heavy_recent",
    "balanced",
    "heavy_key_event",
    "all_key_event",
)


class MainResultConfigError(ValueError):
    """Raised when a main-result configuration violates the supported configuration."""


@dataclass(frozen=True)
class MainResultProfile:
    """One architecture/regime profile for video evaluation."""

    name: str
    family: str
    regime: str
    sampled_frames: int
    nominal_tokens_per_frame: int
    chunk_size: int
    top_m_novel_tokens: int
    event_payload_tokens: int
    event_capacity: int
    recent_cache_tokens: int
    readout_budget_tokens: int
    temporal_coverage_weight: float
    event_allocations: dict[str, int]

    @property
    def maximum_read_events(self) -> int:
        return self.readout_budget_tokens // self.event_payload_tokens


@dataclass(frozen=True)
class MainResultConfig:
    """Supported methods, models, benchmarks, and memory profiles."""

    methods: tuple[str, ...]
    models: dict[str, dict[str, Any]]
    benchmarks: dict[str, dict[str, Any]]
    event_update: dict[str, Any]
    profiles: dict[str, MainResultProfile]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "MainResultConfig":
        config_path = Path(path)
        raw = yaml.safe_load(config_path.read_text())
        if not isinstance(raw, dict):
            raise MainResultConfigError("configuration root must be a mapping")
        if raw.get("schema_version") != 1:
            raise MainResultConfigError("schema_version must be 1")

        models = _mapping(raw, "models")
        benchmarks = _mapping(raw, "benchmarks")
        event_update = _mapping(raw, "event_update")
        profile_values = _mapping(raw, "profiles")
        methods = tuple(raw.get("methods", ()))

        profiles: dict[str, MainResultProfile] = {}
        for name, value in profile_values.items():
            if not isinstance(value, dict):
                raise MainResultConfigError(f"profile {name!r} must be a mapping")
            try:
                profile = MainResultProfile(name=name, **value)
            except TypeError as exc:
                raise MainResultConfigError(f"invalid profile {name!r}: {exc}") from exc
            profiles[name] = profile

        config = cls(
            methods=methods,
            models=models,
            benchmarks=benchmarks,
            event_update=event_update,
            profiles=profiles,
        )
        config.validate()
        return config

    def profile(self, family: str, regime: str) -> MainResultProfile:
        name = f"{family}_{regime}"
        try:
            return self.profiles[name]
        except KeyError as exc:
            raise MainResultConfigError(f"unknown profile {name!r}") from exc

    def validate(self) -> None:
        if self.methods != ("vanilla", "keyrec"):
            raise MainResultConfigError("methods must be exactly vanilla and keyrec")

        expected_model_families = {
            "gemma4_e2b": "gemma",
            "gemma4_e4b": "gemma",
            "neo_ov_2b": "neo",
        }
        if set(self.models) != set(expected_model_families):
            raise MainResultConfigError(
                "models must match the three supported backbones"
            )
        for name, family in expected_model_families.items():
            if self.models[name].get("family") != family:
                raise MainResultConfigError(f"model {name!r} has the wrong family")

        expected_benchmark_regimes = {
            "ovo_bench": "streaming",
            "streamingbench": "streaming",
            "longvideobench": "long_video",
            "video_mme_v2": "long_video",
        }
        if set(self.benchmarks) != set(expected_benchmark_regimes):
            raise MainResultConfigError(
                "benchmarks must match the four supported datasets"
            )
        for name, regime in expected_benchmark_regimes.items():
            if self.benchmarks[name].get("regime") != regime:
                raise MainResultConfigError(f"benchmark {name!r} has the wrong regime")

        expected_update = {
            "merge_similarity": 0.75,
            "merge_max_gap_seconds": 2.0,
            "route_moving_average_weight": 0.2,
        }
        if self.event_update != expected_update:
            raise MainResultConfigError(
                "event update parameters do not match the supported profiles"
            )

        expected_profiles = {
            "gemma_streaming": ("gemma", "streaming", 32, 70, 16, 16, 224, 0.2),
            "neo_streaming": ("neo", "streaming", 32, 210, 48, 16, 672, 0.2),
            "gemma_long_video": ("gemma", "long_video", 64, 70, 16, 32, 448, 0.0),
            "neo_long_video": ("neo", "long_video", 64, 210, 48, 32, 1344, 0.0),
        }
        if set(self.profiles) != set(expected_profiles):
            raise MainResultConfigError("exactly four canonical profiles are required")

        expected_allocations = {
            "streaming": (0, 2, 4, 12, 14),
            "long_video": (0, 8, 14, 24, 28),
        }
        for name, expected in expected_profiles.items():
            profile = self.profiles[name]
            actual = (
                profile.family,
                profile.regime,
                profile.sampled_frames,
                profile.nominal_tokens_per_frame,
                profile.event_payload_tokens,
                profile.event_capacity,
                profile.readout_budget_tokens,
                profile.temporal_coverage_weight,
            )
            if actual != expected:
                raise MainResultConfigError(
                    f"profile {name!r} does not match the supported profile"
                )
            if profile.chunk_size != 1:
                raise MainResultConfigError(f"profile {name!r} must use chunk_size=1")
            if profile.top_m_novel_tokens != profile.event_payload_tokens:
                raise MainResultConfigError(f"profile {name!r} must use keep_tokens=m")
            if profile.recent_cache_tokens != profile.readout_budget_tokens:
                raise MainResultConfigError(f"profile {name!r} must use N_C=B")
            if tuple(profile.event_allocations) != ROUTER_STATES:
                raise MainResultConfigError(
                    f"profile {name!r} has invalid router states"
                )
            if (
                tuple(profile.event_allocations.values())
                != expected_allocations[profile.regime]
            ):
                raise MainResultConfigError(f"profile {name!r} has invalid allocations")
            if (
                profile.event_allocations["all_key_event"]
                != profile.maximum_read_events
            ):
                raise MainResultConfigError(
                    f"profile {name!r} has an invalid event-only readout"
                )
            if max(profile.event_allocations.values()) > profile.event_capacity:
                raise MainResultConfigError(
                    f"profile {name!r} reads beyond event capacity"
                )


def _mapping(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise MainResultConfigError(f"{key} must be a mapping")
    return value
