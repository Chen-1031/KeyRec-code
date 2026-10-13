"""Factory for the supported video model adapters."""

from __future__ import annotations

from .base import VideoModel
from .gemma import GemmaVideoModel
from .keyrec_gemma import KeyRecGemmaModel
from .keyrec_neo import KeyRecNEOModel


def build_video_model(
    backend: str,
    model_path: str,
    *,
    num_frames: int = 32,
    sample_fps: float = 1.0,
    max_soft_tokens: int = 70,
    max_new_tokens: int = 512,
) -> VideoModel:
    """Build Gemma or NEO-ov, with dense input or KeyRec memory."""
    common = dict(
        model_path=model_path,
        num_frames=num_frames,
        sample_fps=sample_fps,
        max_new_tokens=max_new_tokens,
    )
    if backend == "gemma":
        return GemmaVideoModel(**common, max_soft_tokens=max_soft_tokens)
    if backend == "keyrec":
        return KeyRecGemmaModel(**common, max_soft_tokens=max_soft_tokens)
    if backend == "keyrec_neo":
        return KeyRecNEOModel(**common)
    if backend == "neo_ov":
        from .neo_ov import NeoOVVideoModel

        return NeoOVVideoModel(**common)
    raise ValueError(f"Unknown model backend: {backend}")


def ovo_score_model_name(backend: str) -> str:
    """Return the scorer identifier for a supported backend."""
    if backend in {"gemma", "neo_ov"}:
        return "Gemma4"
    if backend in {"keyrec", "keyrec_neo"}:
        return "KeyRecGemma"
    raise ValueError(f"OVO scoring model name is not configured for backend: {backend}")
