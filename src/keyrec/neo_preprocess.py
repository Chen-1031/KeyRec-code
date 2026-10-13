"""Minimal NEO native-resolution preprocessing without VLMEvalKit imports."""

from __future__ import annotations

import math

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _round_by_factor(number: int, factor: int) -> int:
    return round(number / factor) * factor


def _ceil_by_factor(number: int, factor: int) -> int:
    return math.ceil(number / factor) * factor


def _floor_by_factor(number: int, factor: int) -> int:
    return math.floor(number / factor) * factor


def _smart_resize(
    height: int,
    width: int,
    factor: int,
    min_pixels: int,
    max_pixels: int,
) -> tuple[int, int]:
    if max(height, width) / min(height, width) > 200:
        raise ValueError("absolute aspect ratio must be smaller than 200")
    resized_height = max(factor, _round_by_factor(height, factor))
    resized_width = max(factor, _round_by_factor(width, factor))
    if resized_height * resized_width > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        resized_height = max(factor, _floor_by_factor(height / beta, factor))
        resized_width = max(factor, _floor_by_factor(width / beta, factor))
    elif resized_height * resized_width < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        resized_height = _ceil_by_factor(height * beta, factor)
        resized_width = _ceil_by_factor(width * beta, factor)
    return resized_height, resized_width


def _preprocess_pixel_values(pixel_values: torch.Tensor, patch_size: int):
    channels, height, width = pixel_values.shape
    grid_height = height // patch_size
    grid_width = width // patch_size
    flattened = (
        pixel_values.view(
            channels, grid_height, patch_size, grid_width, patch_size
        )
        .permute(1, 3, 0, 2, 4)
        .reshape(grid_height * grid_width, channels * patch_size**2)
    )
    grid_hw = torch.tensor([[grid_height, grid_width]], device=flattened.device)
    return flattened, grid_hw


def _contrasting_background(image: Image.Image):
    image_array = np.array(image)
    if not (image_array[:, :, 3] == 0).any():
        return None
    foreground = image_array[:, :, :3][image_array[:, :, 3] > 0]
    if foreground.size == 0:
        return None
    return (0, 0, 0) if foreground.mean() > 382.5 else (255, 255, 255)


def load_image_native(
    image_file: str,
    patch_size: int = 16,
    downsample_ratio: float = 0.5,
    min_pixels: int = 65536,
    max_pixels: int = 4194304,
    upscale: bool = False,
):
    """Match NEO's released native-resolution image preprocessing."""
    image = Image.open(image_file)
    if image.mode == "RGBA":
        background_color = _contrasting_background(image)
        if background_color:
            background = Image.new("RGB", image.size, background_color)
            background.paste(image, mask=image.split()[3])
            image = background.convert("RGB")
        else:
            image = image.convert("RGB")
    else:
        image = image.convert("RGB")

    if upscale:
        image = image.resize((image.width * 2, image.height * 2), Image.BILINEAR)

    resized_height, resized_width = _smart_resize(
        image.height,
        image.width,
        factor=int(patch_size // downsample_ratio),
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    image = image.resize((resized_width, resized_height))
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return _preprocess_pixel_values(transform(image).to(torch.float32), patch_size)
