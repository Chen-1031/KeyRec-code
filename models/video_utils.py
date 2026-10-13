"""Video loading helpers shared by model adapters."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterator


def _floor_by_factor(value: float, factor: int) -> int:
    return int(value) // factor * factor


def _linspace_indices(start: int, end: int, count: int) -> list[int]:
    if count <= 1:
        return [start]
    return [round(start + (end - start) * i / (count - 1)) for i in range(count)]


def select_video_frame_indices(
    total_frames: int,
    video_fps: float,
    fps: float,
    max_frames: int | None = None,
    sampling: str = "step",
    start_time: float | None = None,
    end_time: float | None = None,
) -> list[int]:
    """Select frame indices for a local video.

    ``sampling="linspace"`` mirrors qwen_vl_utils/TimeLens: compute the
    target frame count from requested FPS, clamp it, round down to an even
    frame count, then sample uniformly across the whole video.
    ``sampling="uniform"`` ignores FPS and samples exactly ``max_frames``
    frames uniformly across the requested interval. ``step`` keeps the previous
    KeyRec/OVO behavior. ``tail`` keeps only the most recent FPS-sampled frames,
    matching sliding-window streaming baselines.
    """
    if total_frames <= 0:
        return []
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    if video_fps <= 0:
        raise ValueError(f"video_fps must be positive, got {video_fps}")

    start_idx = 0 if start_time is None else max(0, int(start_time * video_fps))
    end_idx = (
        total_frames - 1
        if end_time is None
        else min(total_frames - 1, int(end_time * video_fps))
    )
    if end_idx < start_idx:
        return []
    segment_frames = end_idx - start_idx + 1

    if sampling == "step":
        step = max(1, int(video_fps / fps))
        indices = list(range(start_idx, end_idx + 1, step))
        return indices[:max_frames] if max_frames else indices

    if sampling in {"tail", "recent"}:
        step = max(1, int(video_fps / fps))
        indices = list(range(start_idx, end_idx + 1, step))
        return indices[-max_frames:] if max_frames else indices

    if sampling == "uniform":
        nframes = min(max_frames or segment_frames, segment_frames)
        return _linspace_indices(start_idx, end_idx, int(max(nframes, 1)))

    if sampling != "linspace":
        raise ValueError(f"Unsupported video sampling mode: {sampling}")

    nframes = segment_frames / video_fps * fps
    if max_frames:
        nframes = min(nframes, max_frames)
    nframes = min(max(nframes, 1), segment_frames)
    if nframes >= 2:
        nframes = max(2, _floor_by_factor(nframes, 2))
    else:
        nframes = 1
    return _linspace_indices(start_idx, end_idx, int(nframes))


def draw_timestamp_overlay(frame, timestamp_sec: float):
    """Render a timestamp label onto a PIL frame."""
    from PIL import ImageDraw, ImageFont

    image = frame.convert("RGB").copy()
    draw = ImageDraw.Draw(image)
    text = f"{timestamp_sec:.1f}s"

    width, height = image.size
    font_size = max(18, int(min(width, height) * 0.055))
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()

    margin = max(8, int(min(width, height) * 0.02))
    pad_x = max(6, int(font_size * 0.35))
    pad_y = max(4, int(font_size * 0.2))
    bbox = draw.textbbox((margin, margin), text, font=font)
    rect = [
        bbox[0] - pad_x,
        bbox[1] - pad_y,
        bbox[2] + pad_x,
        bbox[3] + pad_y,
    ]
    draw.rectangle(rect, fill=(0, 0, 0))
    draw.text((margin, margin), text, fill=(255, 255, 255), font=font)
    return image


logger = logging.getLogger(__name__)


def _yield_pyav_frames(
    video_path: str | Path,
    indices: list[int],
    video_fps: float,
    add_timestamp_overlay: bool = False,
    return_timestamps: bool = False,
) -> Iterator:
    """Yield selected frames with PyAV as a fallback for decord seek failures."""
    from PIL import Image
    import av

    if not indices:
        return

    wanted = set(indices)
    max_index = max(wanted)

    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        for frame_index, frame in enumerate(container.decode(stream)):
            if frame_index > max_index:
                break
            if frame_index not in wanted:
                continue

            timestamp_sec = frame_index / video_fps
            image = frame.to_image()
            if add_timestamp_overlay:
                image = draw_timestamp_overlay(image, timestamp_sec)
            if return_timestamps:
                yield image, timestamp_sec
            else:
                yield image


def patch_transformers_video_backend() -> None:
    """Force Transformers video processors to use pyav for local videos."""
    import transformers.video_processing_utils as video_processing_utils
    from transformers.video_utils import load_video

    def fetch_videos_pyav(self, video_url_or_urls, sample_indices_fn=None):
        if isinstance(video_url_or_urls, list):
            return list(
                zip(
                    *[
                        self.fetch_videos(x, sample_indices_fn=sample_indices_fn)
                        for x in video_url_or_urls
                    ]
                )
            )
        return load_video(
            video_url_or_urls, backend="pyav", sample_indices_fn=sample_indices_fn
        )

    video_processing_utils.BaseVideoProcessor.fetch_videos = fetch_videos_pyav


def iter_video_frames(
    video_path: str | Path,
    max_frames: int | None = None,
    fps: float = 1.0,
    sampling: str = "step",
    add_timestamp_overlay: bool = False,
    return_timestamps: bool = False,
    start_time: float | None = None,
    end_time: float | None = None,
) -> Iterator:
    """Yield PIL frames from a video at the requested FPS.

    When ``return_timestamps`` is set, each yielded item is
    ``(frame, timestamp_sec)``. Keeping this optional preserves the existing
    model paths while letting memory modules track where selected tokens came
    from in time.
    """
    import decord
    from PIL import Image

    decord.bridge.set_bridge("torch")
    vr = decord.VideoReader(str(video_path))
    video_fps = vr.get_avg_fps()
    indices = select_video_frame_indices(
        total_frames=len(vr),
        video_fps=video_fps,
        fps=fps,
        max_frames=max_frames,
        sampling=sampling,
        start_time=start_time,
        end_time=end_time,
    )

    logger.info(
        "Extracting %d frames from %s at %.3g FPS with %s sampling%s",
        len(indices),
        video_path,
        fps,
        sampling,
        " and timestamp overlay" if add_timestamp_overlay else "",
    )

    try:
        for idx in indices:
            timestamp_sec = idx / video_fps
            frame = vr[idx]
            if hasattr(frame, "numpy"):
                frame = frame.numpy()
            elif hasattr(frame, "asnumpy"):
                frame = frame.asnumpy()
            image = Image.fromarray(frame)
            if add_timestamp_overlay:
                image = draw_timestamp_overlay(image, timestamp_sec)
            if return_timestamps:
                yield image, timestamp_sec
            else:
                yield image
    except Exception as exc:
        if os.environ.get("STREAMING_VU_USE_PYAV_FALLBACK") == "1":
            logger.warning(
                "Decord failed while reading %s (%s); retrying selected frames with PyAV.",
                video_path,
                exc,
            )
            yield from _yield_pyav_frames(
                video_path,
                indices,
                video_fps,
                add_timestamp_overlay=add_timestamp_overlay,
                return_timestamps=return_timestamps,
            )
            return
        logger.warning(
            "Decord failed while reading %s (%s); set STREAMING_VU_USE_PYAV_FALLBACK=1 to retry with PyAV.",
            video_path,
            exc,
        )
        raise


def get_video_frames(
    video_path: str | Path,
    max_frames: int | None = 32,
    fps: float = 1.0,
    sampling: str = "step",
    add_timestamp_overlay: bool = False,
    return_timestamps: bool = False,
    start_time: float | None = None,
    end_time: float | None = None,
):
    """Decode a local video into PIL frames with FPS-based sampling."""
    items = list(
        iter_video_frames(
            video_path,
            max_frames=max_frames,
            fps=fps,
            sampling=sampling,
            add_timestamp_overlay=add_timestamp_overlay,
            return_timestamps=return_timestamps,
            start_time=start_time,
            end_time=end_time,
        )
    )
    if return_timestamps:
        frames = [frame for frame, _ in items]
        timestamps = [timestamp for _, timestamp in items]
    else:
        frames = items
    if not frames:
        raise ValueError(f"No frames selected from video: {video_path}")
    if return_timestamps:
        return frames, timestamps
    return frames


def build_transformers_video_metadata(
    video_path: str | Path,
    max_frames: int | None = 32,
    fps: float = 1.0,
    sampling: str = "step",
    start_time: float | None = None,
    end_time: float | None = None,
):
    """Build HF video metadata for pre-sampled frames when supported."""
    try:
        import decord
        from transformers.video_utils import VideoMetadata
    except Exception:
        return None

    vr = decord.VideoReader(str(video_path))
    video_fps = float(vr.get_avg_fps())
    total_frames = len(vr)
    indices = select_video_frame_indices(
        total_frames=total_frames,
        video_fps=video_fps,
        fps=fps,
        max_frames=max_frames,
        sampling=sampling,
        start_time=start_time,
        end_time=end_time,
    )
    if not indices:
        return None
    try:
        height, width = vr[0].shape[:2]
    except Exception:
        height = width = None
    duration = total_frames / video_fps if video_fps > 0 else None
    return VideoMetadata(
        total_num_frames=total_frames,
        fps=video_fps,
        width=width,
        height=height,
        duration=duration,
        frames_indices=indices,
    )
