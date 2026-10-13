"""Canonical NEO adapter backed by the bounded-memory KeyRec pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from gemma4_video_eval.main_result_config import MainResultConfig
from src.keyrec.neo import NeoKeyRecAdapter
from src.keyrec.pipeline import KeyRecPipeline, PipelineConfig
from src.keyrec.readout import RouterState

from .base import VideoModel
from .video_utils import get_video_frames


class KeyRecNEOModel(VideoModel):
    """NEO-ov wrapper with structured one-frame writes and sparse readout."""

    def __init__(
        self,
        model_path: str,
        num_frames: int = 32,
        sample_fps: float = 1.0,
        max_new_tokens: int = 128,
        *,
        profile_name: str | None = None,
        router_max_new_tokens: int = 64,
        main_result_config: MainResultConfig | None = None,
        adapter: NeoKeyRecAdapter | None = None,
    ) -> None:
        if isinstance(num_frames, bool) or not isinstance(num_frames, int):
            raise ValueError("num_frames must be a positive integer")
        if num_frames <= 0:
            raise ValueError("num_frames must be a positive integer")
        if sample_fps <= 0:
            raise ValueError("sample_fps must be positive")

        config = main_result_config or MainResultConfig.from_yaml(
            Path(__file__).resolve().parents[1] / "configs/main_results.yaml"
        )
        config.validate()
        selected_profile = profile_name or (
            "neo_long_video" if num_frames > 32 else "neo_streaming"
        )
        try:
            profile = config.profiles[selected_profile]
        except KeyError as exc:
            raise ValueError(f"unknown NEO profile {selected_profile!r}") from exc
        if profile.family != "neo":
            raise ValueError("KeyRecNEOModel requires a NEO profile")
        if num_frames > profile.sampled_frames:
            raise ValueError("num_frames exceeds the selected profile frame limit")

        if adapter is None:
            adapter = NeoKeyRecAdapter.from_pretrained(model_path)

        self.model_path = model_path
        self.num_frames = num_frames
        self.sample_fps = float(sample_fps)
        self.max_new_tokens = max_new_tokens
        self.profile_name = selected_profile
        self.sampled_frame_limit = profile.sampled_frames
        self.main_result_config = config
        self.adapter = adapter
        self.model = adapter.model
        self.tokenizer = adapter.tokenizer
        self.pipeline_config = PipelineConfig.from_main_result(
            config,
            selected_profile,
            router_max_new_tokens=router_max_new_tokens,
            answer_max_new_tokens=max_new_tokens,
        )

        self.last_pipeline = None
        self.last_pipeline_result = None
        self.last_router_decision = None
        self.last_frame_timestamps = None
        self.last_frame_count = None
        self.last_memory_events = None
        self.last_memory_segments = None
        self.last_read_policy = None
        self.last_visual_token_count = None
        self.last_neo_span_count = None
        self.last_interleaved_prompt = None
        self.last_interleaved_segments = None
        self.last_video_memory_reused = False

    def generate(self, video_path: str, prompt: str, **kwargs) -> str:
        frames, timestamps = get_video_frames(
            video_path,
            max_frames=kwargs.get("num_frames", self.num_frames),
            fps=kwargs.get("sample_fps", self.sample_fps),
            sampling=kwargs.get("sampling", "step"),
            add_timestamp_overlay=kwargs.get("add_timestamp_overlay", False),
            start_time=kwargs.get("video_start_time"),
            end_time=kwargs.get("video_end_time"),
            return_timestamps=True,
        )
        if not frames:
            raise RuntimeError("video sampling returned no frames")
        if len(frames) > self.sampled_frame_limit:
            raise ValueError("sampled frame count exceeds the selected profile")

        pipeline_config = PipelineConfig(
            profile_name=self.pipeline_config.profile_name,
            memory_write=self.pipeline_config.memory_write,
            readout=self.pipeline_config.readout,
            router_max_new_tokens=kwargs.get(
                "router_max_new_tokens",
                self.pipeline_config.router_max_new_tokens,
            ),
            answer_max_new_tokens=kwargs.get(
                "max_new_tokens",
                self.pipeline_config.answer_max_new_tokens,
            ),
        )
        pipeline = KeyRecPipeline(self.adapter, pipeline_config)
        pipeline.process_frames(frames, timestamps)

        policy_name = kwargs.get("read_policy", "llm_router")
        result = pipeline.answer(
            prompt,
            router_output=kwargs.get("router_output"),
            fixed_state=self._fixed_state(policy_name),
        )

        self.last_pipeline = pipeline
        self.last_pipeline_result = result
        self.last_router_decision = result.decision
        self.last_frame_timestamps = list(timestamps)
        self.last_frame_count = len(frames)
        self.last_memory_events = list(pipeline.writer.event_bank.events)
        self.last_memory_segments = list(result.readout.segments)
        self.last_read_policy = pipeline_config.readout
        self.last_visual_token_count = result.readout.token_count
        self.last_neo_span_count = len(getattr(self.adapter, "last_spans", ()))
        self.last_interleaved_prompt = getattr(
            getattr(self.adapter, "last_prepared_input", None),
            "prompt",
            None,
        )
        self.last_interleaved_segments = list(result.readout.segments)
        self.last_video_memory_reused = False
        return result.response

    @staticmethod
    def _fixed_state(policy_name: str) -> RouterState | None:
        if policy_name == "llm_router":
            return None
        try:
            return RouterState(policy_name)
        except ValueError as exc:
            raise ValueError(
                f"unsupported read policy {policy_name!r}; use llm_router "
                "or one of the five router states"
            ) from exc
