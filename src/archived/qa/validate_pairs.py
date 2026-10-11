#!/usr/bin/env python3
"""List paired Bayer/RGB files and verify sizes before training."""

from __future__ import annotations

import argparse
from pathlib import Path

from bayer_denoiser.archived.io.decode import load_bayer
from bayer_denoiser.archived.qa.pairs import discover_pairs, load_pairs_csv
from bayer_denoiser.archived.io.raw_io import BayerLoadOptions
from PIL import Image


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bayer-dir", type=Path)
    p.add_argument("--rgb-dir", type=Path)
    p.add_argument("--pairs-csv", type=Path)
    p.add_argument("--black-level", type=float, default=0.0)
    p.add_argument("--white-level", type=float, default=None)
    p.add_argument("--raw-width", type=int, default=None)
    p.add_argument("--raw-height", type=int, default=None)
    p.add_argument("--raw-dtype", type=str, default="uint16")
    args = p.parse_args()

    if args.pairs_csv:
        pairs = load_pairs_csv(args.pairs_csv)
    elif args.bayer_dir and args.rgb_dir:
        pairs = discover_pairs(args.bayer_dir, args.rgb_dir)
    else:
        raise SystemExit("Provide --pairs-csv or both --bayer-dir and --rgb-dir")

    opts = BayerLoadOptions(
        black_level=args.black_level,
        white_level=args.white_level,
        raw_width=args.raw_width,
        raw_height=args.raw_height,
        raw_dtype=args.raw_dtype,
    )

    ok = 0
    for pair in pairs:
        loaded = load_bayer(pair.bayer, options=opts, pack=False)
        bh, bw = loaded.mosaic.shape
        with Image.open(pair.rgb) as img:
            w, h = img.size
        match = (bh, bw) == (h, w)
        status = "OK" if match else "MISMATCH"
        print(f"[{status}] {pair.bayer.name} {bh}x{bw}  ↔  {pair.rgb.name} {h}x{w}")
        if match:
            ok += 1

    print(f"{ok}/{len(pairs)} pairs ready")
    if ok != len(pairs):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
