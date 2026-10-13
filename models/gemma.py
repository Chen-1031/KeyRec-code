"""Plain Gemma 4 video adapter."""

from __future__ import annotations

from .base import VideoModel
from .video_utils import (
    build_transformers_video_metadata,
    get_video_frames,
    patch_transformers_video_backend,
)


class GemmaVideoModel(VideoModel):
    """Naive Gemma 4 video inference through the Hugging Face processor."""

    def __init__(
        self,
        model_path: str,
        num_frames: int = 32,
        sample_fps: float = 1.0,
        max_soft_tokens: int = 70,
        max_new_tokens: int = 512,
    ) -> None:
        patch_transformers_video_backend()

        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        self.model_path = model_path
        self.num_frames = num_frames
        self.sample_fps = sample_fps
        self.max_new_tokens = max_new_tokens
        self.dtype = torch.bfloat16
        self.model = AutoModelForMultimodalLM.from_pretrained(
            model_path,
            torch_dtype=self.dtype,
            device_map="auto",
        ).eval()
        self.device = self.model.device
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.processor.image_processor.max_soft_tokens = max_soft_tokens
        self.processor.video_processor.max_soft_tokens = max_soft_tokens
        self.processor.video_processor.num_frames = num_frames

    def generate(self, video_path: str, prompt: str, **kwargs) -> str:
        import torch

        num_frames = kwargs.get("num_frames", self.num_frames)
        sample_fps = kwargs.get("sample_fps", self.sample_fps)
        sampling = kwargs.get("sampling", "step")
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
        old_num_frames = getattr(self.processor.video_processor, "num_frames", None)
        self.processor.video_processor.num_frames = len(frames)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": frames},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        template_kwargs = dict(
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        )
        if hasattr(self.processor, "video_processor"):
            metadata = build_transformers_video_metadata(
                video_path,
                max_frames=num_frames,
                fps=sample_fps,
                sampling=sampling,
                start_time=video_start_time,
                end_time=video_end_time,
            )
            if metadata is not None:
                template_kwargs["processor_kwargs"] = {
                    "video_metadata": [metadata],
                    "do_sample_frames": False,
                }
        try:
            try:
                inputs = self.processor.apply_chat_template(messages, **template_kwargs)
            except TypeError:
                template_kwargs.pop("enable_thinking", None)
                try:
                    inputs = self.processor.apply_chat_template(
                        messages, **template_kwargs
                    )
                except TypeError:
                    template_kwargs.pop("processor_kwargs", None)
                    inputs = self.processor.apply_chat_template(
                        messages, **template_kwargs
                    )
        finally:
            if old_num_frames is not None:
                self.processor.video_processor.num_frames = old_num_frames

        inputs = inputs.to(self.device, dtype=self.dtype)
        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=kwargs.get("max_new_tokens", self.max_new_tokens),
            )
        generated = output_ids[0][input_len:]
        return self.processor.decode(generated, skip_special_tokens=True)
