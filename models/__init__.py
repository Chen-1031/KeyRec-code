"""Video model adapters for KeyRec evaluation."""

from .base import VideoModel
from .builder import build_video_model, ovo_score_model_name
from .gemma import GemmaVideoModel
from .keyrec_gemma import KeyRecGemmaModel
from .keyrec_neo import KeyRecNEOModel

__all__ = [
    "VideoModel",
    "GemmaVideoModel",
    "KeyRecGemmaModel",
    "KeyRecNEOModel",
    "build_video_model",
    "ovo_score_model_name",
]
