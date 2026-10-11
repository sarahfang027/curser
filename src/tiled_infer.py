"""Sliding-window Bayer mosaic inference (packed RGGB U-Net)."""

from __future__ import annotations

import numpy as np
import torch

from bayer_denoiser.sort_bayer import bayer_mosaic_to_packed, packed_to_bayer_mosaic


def _even_floor(x: int) -> int:
    return x - (x % 2)


def tile_axis_starts(length: int, tile_size: int, overlap: int) -> list[int]:
    """Start indices along one axis; last window is full ``tile_size`` when possible."""
    if tile_size <= 0:
        raise ValueError(f"tile_size must be positive, got {tile_size}")
    if overlap < 0 or overlap >= tile_size:
        raise ValueError(f"overlap must be in [0, tile_size), got {overlap}")
    if length <= tile_size:
        return [0]
    stride = tile_size - overlap
    starts: list[int] = []
    pos = 0
    while True:
        pos = _even_floor(pos)
        if pos + tile_size > length:
            pos = _even_floor(max(0, length - tile_size))
        if starts and starts[-1] == pos:
            break
        starts.append(pos)
        if pos + tile_size >= length:
            break
        pos += stride
    return starts


def _tile_blend_weights(
    tile_size: int,
    overlap: int,
    top: int,
    left: int,
    mosaic_h: int,
    mosaic_w: int,
) -> np.ndarray:
    """2D blend weights: 1.0 at mosaic edges; linear ramps only on interior overlaps."""
    wy = np.ones(tile_size, dtype=np.float32)
    wx = np.ones(tile_size, dtype=np.float32)
    if overlap <= 0:
        return wy[:, None] * wx[None, :]
    ramp = np.linspace(0.0, 1.0, overlap, dtype=np.float32)
    if top > 0:
        wy[:overlap] = ramp
    if top + tile_size < mosaic_h:
        wy[-overlap:] = ramp[::-1]
    if left > 0:
        wx[:overlap] = ramp
    if left + tile_size < mosaic_w:
        wx[-overlap:] = ramp[::-1]
    return wy[:, None] * wx[None, :]


def run_bayer_tiled_infer(
    model: torch.nn.Module,
    mosaic_hw: np.ndarray,
    *,
    device: torch.device,
    tile_size: int,
    overlap: int,
) -> np.ndarray:
    """
    Denoise a full H×W mosaic by tiling. Returns float32 mosaic in [0, 1].

    Tile origins are even (RGGB phase). Overlap > 0 uses separable linear ramps.
    """
    mosaic = np.asarray(mosaic_hw, dtype=np.float32)
    if mosaic.ndim != 2:
        raise ValueError(f"Expected (H, W) mosaic, got {mosaic.shape}")
    h, w = mosaic.shape
    if tile_size % 2:
        raise ValueError(f"tile_size must be even, got {tile_size}")
    if h % 2 or w % 2:
        raise ValueError(f"Mosaic height and width must be even, got {h}x{w}")

    if h <= tile_size and w <= tile_size:
        packed = bayer_mosaic_to_packed(
            torch.from_numpy(mosaic).unsqueeze(0).unsqueeze(0).to(device)
        )
        with torch.no_grad():
            den = model(packed)
        out = packed_to_bayer_mosaic(den).squeeze(0).squeeze(0).cpu().numpy()
        return out.astype(np.float32)

    acc = np.zeros((h, w), dtype=np.float64)
    weight = np.zeros((h, w), dtype=np.float64)

    tops = tile_axis_starts(h, tile_size, overlap)
    lefts = tile_axis_starts(w, tile_size, overlap)
    model.eval()

    for top in tops:
        for left in lefts:
            patch = mosaic[top : top + tile_size, left : left + tile_size]
            packed = bayer_mosaic_to_packed(
                torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(device)
            )
            with torch.no_grad():
                den = model(packed)
            den_hw = (
                packed_to_bayer_mosaic(den).squeeze(0).squeeze(0).cpu().numpy().astype(np.float64)
            )
            tw = _tile_blend_weights(tile_size, overlap, top, left, h, w)
            acc[top : top + tile_size, left : left + tile_size] += den_hw * tw
            weight[top : top + tile_size, left : left + tile_size] += tw

    if np.any(weight <= 0):
        raise RuntimeError("tiled infer: zero blend weight — check tile_size/overlap")
    return (acc / weight).astype(np.float32)
