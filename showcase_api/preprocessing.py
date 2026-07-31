from __future__ import annotations

import io
from typing import Sequence

import numpy as np
from PIL import Image, ImageOps


DEFAULT_EVAL_CROP_PCT = 224 / 256


def eval_resize_shorter_side(input_size: int, crop_pct: float = DEFAULT_EVAL_CROP_PCT) -> int:
    if input_size <= 0:
        raise ValueError("input_size must be positive")
    if crop_pct <= 0:
        raise ValueError("crop_pct must be positive")
    return int(input_size / crop_pct)


def resize_for_formal_evaluation(
    image: Image.Image,
    *,
    input_size: int,
    crop_pct: float = DEFAULT_EVAL_CROP_PCT,
) -> Image.Image:
    """Mirror training_core.datasets.build_transform(is_train=False)."""
    resample = getattr(Image, "Resampling", Image).BICUBIC
    if input_size >= 384:
        return image.resize((input_size, input_size), resample)

    shorter_side = eval_resize_shorter_side(input_size, crop_pct)
    width, height = image.size
    if width <= height:
        resized_width = shorter_side
        resized_height = int(shorter_side * height / width)
    else:
        resized_height = shorter_side
        resized_width = int(shorter_side * width / height)

    resized = image.resize((resized_width, resized_height), resample)
    left = int(round((resized_width - input_size) / 2.0))
    top = int(round((resized_height - input_size) / 2.0))
    return resized.crop((left, top, left + input_size, top + input_size))


def decode_rgb_for_formal_evaluation(
    image_bytes: bytes,
    *,
    input_size: int,
    crop_pct: float = DEFAULT_EVAL_CROP_PCT,
    max_image_dimension: int,
) -> tuple[np.ndarray, tuple[int, int]]:
    with Image.open(io.BytesIO(image_bytes)) as image:
        image = ImageOps.exif_transpose(image)
        width, height = image.size
        if width <= 0 or height <= 0 or width > max_image_dimension or height > max_image_dimension:
            raise ValueError("image dimensions are invalid or too large")
        image = resize_for_formal_evaluation(
            image.convert("RGB"),
            input_size=input_size,
            crop_pct=crop_pct,
        )
        rgb_hwc = np.asarray(image, dtype=np.float32) / 255.0
    return rgb_hwc, (width, height)


def normalize_rgb_hwc(
    rgb_hwc: np.ndarray,
    *,
    mean: Sequence[float],
    std: Sequence[float],
    clip_abs: float = 0.0,
) -> np.ndarray:
    if rgb_hwc.ndim != 3 or rgb_hwc.shape[2] != 3:
        raise ValueError("rgb_hwc must have shape [height, width, 3]")
    rgb = np.transpose(rgb_hwc, (2, 0, 1))[None, ...].astype(np.float32)
    mean_array = np.asarray(mean, dtype=np.float32).reshape(1, 3, 1, 1)
    std_array = np.asarray(std, dtype=np.float32).reshape(1, 3, 1, 1)
    if np.any(std_array <= 0):
        raise ValueError("normalization std must be positive")
    normalized = ((rgb - mean_array) / std_array).astype(np.float32)
    if clip_abs > 0:
        normalized = np.clip(normalized, -clip_abs, clip_abs).astype(np.float32)
    return normalized
