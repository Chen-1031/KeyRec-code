"""NEO-ov video adapter.

This adapter keeps the NEO integration separate from the Gemma paths because
NEO treats video as a sequence of image frames processed by its own native
patch preprocessor.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import tempfile
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

from src.keyrec.neo_preprocess import load_image_native

from .base import VideoModel
from .video_utils import get_video_frames


class NeoOVVideoModel(VideoModel):
    """NEO-ov 2B video inference using its native image-frame preprocessor."""

    def __init__(
        self,
        model_path: str,
        num_frames: int = 32,
        sample_fps: float = 1.0,
        max_new_tokens: int = 128,
        patch_size: int = 16,
        downsample_ratio: float = 0.5,
        min_pixels: int = 4 * 32 * 32,
        max_pixels: int = 360 * 640,
    ) -> None:
        self.model_path = model_path
        self.num_frames = num_frames
        self.sample_fps = sample_fps
        self.max_new_tokens = max_new_tokens
        self.patch_size = patch_size
        self.downsample_ratio = downsample_ratio
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.load_image_native = load_image_native

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=False,
        )
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map="auto",
        ).eval()
        self.device = "cuda"

    def _save_temp_frames(self, frames, tmpdir: Path) -> list[str]:
        frame_paths: list[str] = []
        for idx, frame in enumerate(frames, start=1):
            path = tmpdir / f"neoimg_{idx:04d}.jpg"
            frame.convert("RGB").save(path, quality=95)
            frame_paths.append(str(path))
        return frame_paths

    def _encode_frame_files(self, frame_paths: list[str]):
        grid_hw_list = []
        pixel_values_list = []
        quiet = os.environ.get("STREAMING_VU_NEO_VERBOSE_PREPROCESS") != "1"
        stream = open(os.devnull, "w") if quiet else None
        try:
            redirect = (
                contextlib.redirect_stdout(stream)
                if stream is not None
                else contextlib.nullcontext()
            )
            with redirect:
                for frame_path in frame_paths:
                    pixel_values, grid_hw = self.load_image_native(
                        frame_path,
                        patch_size=self.patch_size,
                        downsample_ratio=self.downsample_ratio,
                        min_pixels=self.min_pixels,
                        max_pixels=self.max_pixels,
                    )
                    grid_hw_list.append(grid_hw.to(self.device))
                    pixel_values_list.append(
                        pixel_values.to(self.device).to(torch.bfloat16)
                    )
        finally:
            if stream is not None:
                stream.close()

        if not pixel_values_list:
            return None, None
        return torch.cat(pixel_values_list, dim=0), torch.cat(grid_hw_list, dim=0)

    def generate(self, video_path: str, prompt: str, **kwargs) -> str:
        self.last_frame_count = None
        num_frames = kwargs.get("num_frames", self.num_frames)
        sample_fps = kwargs.get("sample_fps", self.sample_fps)
        sampling = kwargs.get("sampling", "uniform")
        video_start_time = kwargs.get("video_start_time")
        video_end_time = kwargs.get("video_end_time")

        frames = get_video_frames(
            video_path,
            max_frames=num_frames,
            fps=sample_fps,
            sampling=sampling,
            add_timestamp_overlay=kwargs.get("add_timestamp_overlay", False),
            start_time=video_start_time,
            end_time=video_end_time,
        )
        if len(frames) == 1:
            frames = [frames[0], frames[0]]
        self.last_frame_count = len(frames)

        digest = hashlib.md5(str(video_path).encode("utf-8")).hexdigest()[:10]
        with tempfile.TemporaryDirectory(prefix=f"streaming_vu_neo_{digest}_") as tmp:
            frame_paths = self._save_temp_frames(frames, Path(tmp))
            pixel_values, grid_hw = self._encode_frame_files(frame_paths)

        question = ("<image>\n" * len(frames)) + prompt
        generation_config = {
            "do_sample": False,
            "max_new_tokens": kwargs.get("max_new_tokens", self.max_new_tokens),
            "top_p": None,
        }

        with torch.inference_mode():
            return self.model.chat(
                self.tokenizer,
                pixel_values=pixel_values,
                grid_hw=grid_hw,
                question=question,
                generation_config=generation_config,
                verbose=False,
            )
