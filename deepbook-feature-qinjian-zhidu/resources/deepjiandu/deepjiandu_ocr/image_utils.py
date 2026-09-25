from __future__ import annotations

from PIL import Image
import numpy as np
import torch


def letterbox_grayscale(image: Image.Image, size: int, fill: int = 0) -> Image.Image:
    if image.mode != "L":
        image = image.convert("L")

    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError("Image has invalid size.")

    scale = min(size / width, size / height)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resized = image.resize((new_width, new_height), Image.Resampling.BILINEAR)

    canvas = Image.new("L", (size, size), color=fill)
    paste_x = (size - new_width) // 2
    paste_y = (size - new_height) // 2
    canvas.paste(resized, (paste_x, paste_y))
    return canvas


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    # Normalize to approximately zero-centered values for stable optimization.
    array = (array - 0.5) / 0.5
    return torch.from_numpy(array).unsqueeze(0)

