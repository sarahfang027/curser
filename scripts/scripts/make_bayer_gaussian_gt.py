#!/usr/bin/env python3
"""Bayer-safe Gaussian smooth on packed RGGB planes.

Input/output: float16 H×W mosaic in [0, 1] on disk (float32 math inside), same contract as bayer_denoiser infer.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from bayer_denoiser.sort_bayer import mosaic_to_packed_np, packed_to_mosaic_np


def _gaussian_filter_plane_scipy(plane: np.ndarray, sigma: float) -> np.ndarray:
    from scipy.ndimage import gaussian_filter

    return gaussian_filter(plane, sigma=sigma, mode="reflect")


def _gaussian_filter_plane_torch(plane: np.ndarray, sigma: float) -> np.ndarray:
    import torch
    import torch.nn.functional as F

    if sigma <= 0:
        return np.asarray(plane, dtype=np.float32)
    radius = max(1, int(round(4.0 * sigma)))
    ksize = 2 * radius + 1
    x = torch.arange(-radius, radius + 1, dtype=torch.float32)
    k1d = torch.exp(-0.5 * (x / sigma) ** 2)
    k1d = k1d / k1d.sum()
    kh = k1d.view(1, 1, 1, ksize)
    kv = k1d.view(1, 1, ksize, 1)

    t = torch.from_numpy(np.asarray(plane, dtype=np.float32)).unsqueeze(0).unsqueeze(0)
    pad = radius
    t = F.pad(t, (pad, pad, pad, pad), mode="reflect")
    t = F.conv2d(t, kh)
    t = F.conv2d(t, kv)
    return t.squeeze(0).squeeze(0).numpy()


def gaussian_filter_plane(plane: np.ndarray, sigma: float, backend: str) -> np.ndarray:
    if backend == "scipy":
        return _gaussian_filter_plane_scipy(plane, sigma)
    if backend == "torch":
        return _gaussian_filter_plane_torch(plane, sigma)
    # auto
    try:
        return _gaussian_filter_plane_scipy(plane, sigma)
    except ImportError:
        return _gaussian_filter_plane_torch(plane, sigma)

def gaussian_clean_mosaic(mosaic: np.ndarray, sigma: float, backend: str = "auto") -> np.ndarray:
    packed = mosaic_to_packed_np(mosaic)
    for c in range(4):
        print(f"  plane {c}/3 sigma={sigma} backend={backend} shape={packed[c].shape}")
        packed[c] = gaussian_filter_plane(packed[c], sigma, backend)
    out = packed_to_mosaic_np(packed)
    return np.clip(out, 0.0, 1.0).astype(np.float16)


def center_crop(mosaic: np.ndarray, tile: int) -> np.ndarray:
    h, w = mosaic.shape
    top = (h - tile) // 2
    top -= top % 2
    left = (w - tile) // 2
    left -= left % 2
    return mosaic[top : top + tile, left : left + tile].astype(np.float16)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True, help="Noisy/source .npy mosaic")
    p.add_argument("--output", type=Path, required=True, help="Gaussian GT .npy")
    p.add_argument(
        "--sigma",
        type=float,
        default=1.0,
        help="Gaussian sigma per RGGB plane (default 1.0)",
    )
    p.add_argument(
        "--tile-size",
        type=int,
        default=0,
        help="If >0, RGGB-aligned center crop before smoothing",
    )
    p.add_argument(
        "--backend",
        choices=("auto", "scipy", "torch"),
        default="auto",
        help="Gaussian implementation (default auto: scipy if installed else torch)",
    )
    args = p.parse_args()

    mosaic = np.load(args.input)
    print(f"loaded {args.input} shape={mosaic.shape} dtype={mosaic.dtype}")
    if args.tile_size > 0:
        mosaic = center_crop(mosaic, args.tile_size)
        print(f"center crop tile-size={args.tile_size} -> {mosaic.shape}")

    clean = gaussian_clean_mosaic(mosaic, args.sigma, args.backend)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, clean)
    print(f"wrote {args.output} shape={clean.shape} sigma={args.sigma}")


if __name__ == "__main__":
    main()
