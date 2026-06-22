"""Crop node patches from screenshots."""
from __future__ import annotations

from PIL import Image

MIN_PATCH_SIZE = 32


def pad_patch_for_embedding(patch: Image.Image, min_size: int = MIN_PATCH_SIZE) -> Image.Image:
    w, h = patch.size
    if w >= min_size and h >= min_size:
        return patch
    canvas = Image.new("RGB", (max(w, min_size), max(h, min_size)), (255, 255, 255))
    canvas.paste(patch, (0, 0))
    return canvas


def crop_node_patch(screenshot: Image.Image, bounds: list[int]) -> Image.Image | None:
    if len(bounds) != 4:
        return None
    left, top, right, bottom = bounds
    width, height = screenshot.size
    left = max(0, min(left, width - 1))
    top = max(0, min(top, height - 1))
    right = max(left + 1, min(right, width))
    bottom = max(top + 1, min(bottom, height))
    if right - left < 1 or bottom - top < 1:
        return None
    patch = screenshot.crop((left, top, right, bottom))
    return pad_patch_for_embedding(patch)
