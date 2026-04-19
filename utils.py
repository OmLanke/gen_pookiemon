"""
utils.py — Image I/O and visualisation utilities (Pillow-based)

Replaces the original scipy.misc.imread / imsave which were removed in SciPy 1.6.
All image I/O is done through Pillow (PIL) for modern Python compatibility.
Normalisation contract is preserved: pixels ↔ [-1, 1] ↔ tanh output range.
"""

from __future__ import annotations

import math
import datetime
from pathlib import Path

import numpy as np
from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
# Low-level I/O
# ─────────────────────────────────────────────────────────────────────────────


def imread(path: str | Path, grayscale: bool = False) -> np.ndarray:
    """
    Read an image from disk as a float64 numpy array (range [0, 255]).
    Drop-in replacement for scipy.misc.imread.
    """
    img = Image.open(path)
    if grayscale:
        img = img.convert("L")
    else:
        img = img.convert("RGB")
    return np.array(img, dtype=np.float32)


def imsave(images: np.ndarray, size: list[int], path: str | Path) -> None:
    """
    Arrange *images* into a grid of shape *size* = [rows, cols] and save to *path*.
    Expects images in [0, 1] float range; converts to uint8 for saving.
    """
    grid = merge(images, size)
    grid_uint8 = (np.squeeze(grid) * 255.0).clip(0, 255).astype(np.uint8)
    Image.fromarray(grid_uint8).save(path)


# ─────────────────────────────────────────────────────────────────────────────
# Normalisation
# ─────────────────────────────────────────────────────────────────────────────


def normalize(image: np.ndarray) -> np.ndarray:
    """[0, 255] → [-1, 1]  (matches tanh output range)."""
    return np.array(image, dtype=np.float32) / 127.5 - 1.0


def inverse_transform(images: np.ndarray) -> np.ndarray:
    """[-1, 1] → [0, 1]  (applied before saving to disk)."""
    return (images + 1.0) / 2.0


# ─────────────────────────────────────────────────────────────────────────────
# Crop / resize
# ─────────────────────────────────────────────────────────────────────────────


def center_crop(
    x: np.ndarray, crop_h: int, crop_w: int, resize_h: int = 64, resize_w: int = 64
) -> np.ndarray:
    """Crop the centre of *x* to [crop_h × crop_w] then resize to [resize_h × resize_w]."""
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h) / 2.0))
    i = int(round((w - crop_w) / 2.0))
    cropped = x[j : j + crop_h, i : i + crop_w]
    img = Image.fromarray(cropped.astype(np.uint8))
    img = img.resize((resize_w, resize_h), Image.LANCZOS)
    return np.array(img, dtype=np.float32)


def transform(
    image: np.ndarray,
    input_height: int,
    input_width: int,
    resize_height: int = 64,
    resize_width: int = 64,
    crop: bool = True,
) -> np.ndarray:
    """
    Crop/resize then normalise to [-1, 1].
    crop=True  → centre-crop to input_height × input_width, then resize
    crop=False → direct resize to resize_height × resize_width
    """
    if crop:
        out = center_crop(image, input_height, input_width, resize_height, resize_width)
    else:
        img = Image.fromarray(image.astype(np.uint8))
        img = img.resize((resize_width, resize_height), Image.LANCZOS)
        out = np.array(img, dtype=np.float32)
    return normalize(out)


def get_image(
    image_path: str | Path,
    input_height: int,
    input_width: int,
    resize_height: int = 64,
    resize_width: int = 64,
    crop: bool = True,
    grayscale: bool = False,
) -> np.ndarray:
    """Full pipeline: read → transform → normalise. Returns float32 in [-1, 1]."""
    image = imread(image_path, grayscale)
    return transform(
        image, input_height, input_width, resize_height, resize_width, crop
    )


# ─────────────────────────────────────────────────────────────────────────────
# Grid assembly
# ─────────────────────────────────────────────────────────────────────────────


def merge(images: np.ndarray, size: list[int]) -> np.ndarray:
    """
    Pack *images* (shape [N, H, W, C]) into a grid of *size* = [rows, cols].
    Returns array of shape [rows*H, cols*W, C].
    """
    h, w = images.shape[1], images.shape[2]
    c = images.shape[3]

    if c in (1, 3, 4):
        grid = np.zeros((h * size[0], w * size[1], c), dtype=images.dtype)
        for idx, image in enumerate(images):
            col = idx % size[1]
            row = idx // size[1]
            grid[row * h : row * h + h, col * w : col * w + w, :] = image
        return grid
    raise ValueError(
        f"merge: images must have 1, 3, or 4 channels; got shape {images.shape}"
    )


def save_images(images: np.ndarray, size: list[int], image_path: str | Path) -> None:
    """Denormalise [-1,1]→[0,1], assemble grid, save PNG."""
    imsave(inverse_transform(images), size, image_path)


# ─────────────────────────────────────────────────────────────────────────────
# Grid size helper
# ─────────────────────────────────────────────────────────────────────────────


def image_manifold_size(num_images: int) -> tuple[int, int]:
    """
    Return (rows, cols) for a near-square grid.
    For batch_size=64 → (8, 8).
    Asserts rows * cols == num_images to catch mismatches early.
    """
    rows = int(math.floor(math.sqrt(num_images)))
    cols = int(math.ceil(math.sqrt(num_images)))
    assert rows * cols == num_images, (
        f"image_manifold_size: {num_images} images cannot fill a {rows}×{cols} grid"
    )
    return rows, cols


# ─────────────────────────────────────────────────────────────────────────────
# Inference helper
# ─────────────────────────────────────────────────────────────────────────────


def visualize(model: object, config: object, option: int = 0) -> None:
    """
    Run the trained generator and save an output image grid.
    option=0: sample z ~ N(-1, 1), generate batch, save timestamped PNG.
    """
    import torch

    image_frame_dim = int(math.ceil(config.batch_size**0.5))
    samples_dir = Path(config.sample_dir)
    samples_dir.mkdir(parents=True, exist_ok=True)

    if option == 0:
        z_sample = (
            torch.randn(config.batch_size, model.z_dim, device=model.device) * 1.0
        )
        # Remap to N(-1,1) range used during training
        z_sample = z_sample.clamp(-1.0, 1.0)

        model.netG.eval()
        with torch.no_grad():
            samples = model.netG(z_sample)

        # [B, C, H, W] → [B, H, W, C] numpy for save_images
        samples_np = samples.cpu().permute(0, 2, 3, 1).numpy()
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        out_path = samples_dir / f"test_{timestamp}.png"
        save_images(samples_np, [image_frame_dim, image_frame_dim], out_path)
        print(f"[*] Saved inference grid → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Variable count helper (replaces tf.contrib.slim.model_analyzer)
# ─────────────────────────────────────────────────────────────────────────────


def show_all_variables(model: object) -> None:
    """Print all trainable parameter names and shapes."""
    import torch.nn as nn

    assert isinstance(model, nn.Module)
    total = 0
    print(f"\n{'Name':<60} {'Shape':<30} {'Params':>10}")
    print("-" * 103)
    for name, param in model.named_parameters():
        if param.requires_grad:
            n = param.numel()
            total += n
            print(f"{name:<60} {str(list(param.shape)):<30} {n:>10,}")
    print("-" * 103)
    print(f"{'Total trainable parameters':<60} {'':30} {total:>10,}\n")
