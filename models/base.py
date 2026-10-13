"""Minimal model interface shared by benchmark wrappers."""

from __future__ import annotations

from abc import ABC, abstractmethod


class VideoModel(ABC):
    """A benchmark-agnostic video QA/generation interface."""

    @abstractmethod
    def generate(self, video_path: str, prompt: str, **kwargs) -> str:
        """Generate a text response for one video and one prompt."""
