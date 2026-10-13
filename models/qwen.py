"""Placeholder for future Qwen video adapters."""

from __future__ import annotations

from .base import VideoModel


class QwenVideoModel(VideoModel):
    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError("QwenVideoModel is not implemented yet.")

    def generate(self, video_path: str, prompt: str, **kwargs) -> str:
        raise NotImplementedError("QwenVideoModel is not implemented yet.")
