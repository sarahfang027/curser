#!/usr/bin/env python3
"""Export a single-channel Bayer .npy from an RGB reference (for testing infer)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF

from bayer_denoiser.sort_bayer import rgb_to_bayer


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rgb", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--size", type=int, default=0, help="Optional square resize (even)")
    args = p.parse_args()

    img = Image.open(args.rgb).convert("RGB")
    rgb = TF.to_tensor(img)
    if args.size:
        if args.size % 2:
            raise SystemExit("--size must be even")
        rgb = TF.resize(rgb, [args.size, args.size], antialias=True)
    h, w = rgb.shape[1], rgb.shape[2]
    if h % 2 or w % 2:
        raise SystemExit(f"Image size must be even after load; got {h}x{w}")

    bayer = rgb_to_bayer(rgb.unsqueeze(0)).squeeze().numpy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, bayer)
    print(f"Wrote Bayer mosaic {bayer.shape} → {args.output}")


if __name__ == "__main__":
    main()
