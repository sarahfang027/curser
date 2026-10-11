"""RGGB Bayer packing and mosaic utilities."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

PATTERN_RGGB = "rggb"


def rgb_to_bayer(rgb: Tensor, pattern: str = PATTERN_RGGB) -> Tensor:
    """Convert RGB float tensor (B,3,H,W) in [0,1] to single-channel Bayer (B,1,H,W)."""
    if pattern != PATTERN_RGGB:
        raise ValueError(f"Only {PATTERN_RGGB!r} is supported, got {pattern!r}")
    b, _, h, w = rgb.shape
    if h % 2 or w % 2:
        raise ValueError(f"Height and width must be even, got {h}x{w}")

    r, g, b_ch = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    mosaic = torch.zeros(b, h, w, device=rgb.device, dtype=rgb.dtype)
    mosaic[:, 0::2, 0::2] = r[:, 0::2, 0::2]
    mosaic[:, 1::2, 0::2] = g[:, 1::2, 0::2]
    mosaic[:, 0::2, 1::2] = g[:, 0::2, 1::2]
    mosaic[:, 1::2, 1::2] = b_ch[:, 1::2, 1::2]
    return mosaic.unsqueeze(1)


def bayer_mosaic_to_packed(bayer: Tensor) -> Tensor:
    """Single-channel Bayer (B,1,H,W) → 4-channel RGGB planes (B,4,H/2,W/2)."""
    if bayer.dim() != 4 or bayer.shape[1] != 1:
        raise ValueError("Expected bayer shape (B,1,H,W)")
    _, _, h, w = bayer.shape
    if h % 2 or w % 2:
        raise ValueError(f"Height and width must be even, got {h}x{w}")
    x = bayer[:, 0]
    r = x[:, 0::2, 0::2]
    g1 = x[:, 1::2, 0::2]
    g2 = x[:, 0::2, 1::2]
    b = x[:, 1::2, 1::2]
    return torch.stack([r, g1, g2, b], dim=1)


def rggb_mosaic_to_display_rgb(mosaic_hw: Tensor) -> Tensor:
    """Map RGGB mosaic (H, W) to (3, H, W) false-color RGB for PNG preview."""
    if mosaic_hw.dim() != 2:
        raise ValueError(f"Expected (H, W) mosaic, got shape {tuple(mosaic_hw.shape)}")
    h, w = mosaic_hw.shape
    if h % 2 or w % 2:
        raise ValueError(f"Height and width must be even, got {h}x{w}")
    rgb = torch.zeros(3, h, w, dtype=mosaic_hw.dtype, device=mosaic_hw.device)
    rgb[0, 0::2, 0::2] = mosaic_hw[0::2, 0::2]
    rgb[1, 1::2, 0::2] = mosaic_hw[1::2, 0::2]
    rgb[1, 0::2, 1::2] = mosaic_hw[0::2, 1::2]
    rgb[2, 1::2, 1::2] = mosaic_hw[1::2, 1::2]
    return rgb.clamp(0.0, 1.0)


def packed_to_bayer_mosaic(packed: Tensor) -> Tensor:
    """4-channel RGGB (B,4,H/2,W/2) → single-channel Bayer (B,1,H,W)."""
    if packed.shape[1] != 4:
        raise ValueError("Expected packed shape (B,4,H,W)")
    b, _, h2, w2 = packed.shape
    h, w = h2 * 2, w2 * 2
    mosaic = torch.zeros(b, h, w, device=packed.device, dtype=packed.dtype)
    mosaic[:, 0::2, 0::2] = packed[:, 0]
    mosaic[:, 1::2, 0::2] = packed[:, 1]
    mosaic[:, 0::2, 1::2] = packed[:, 2]
    mosaic[:, 1::2, 1::2] = packed[:, 3]
    return mosaic.unsqueeze(1)


def load_bayer_from_array(arr: np.ndarray) -> Tensor:
    """Load Bayer from HxW or HxWx1 numpy array, values in [0,1] or 0–255."""
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D Bayer array, got shape {arr.shape}")
    x = arr.astype(np.float32)
    if x.max() > 1.0:
        x = x / 255.0
    return torch.from_numpy(x).unsqueeze(0).unsqueeze(0)
