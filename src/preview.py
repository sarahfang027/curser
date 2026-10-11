#!/usr/bin/env python3
"""Preview decoded Bayer (no ML): decode stats and optional mosaic PNG.

PNG output (``.raw`` and ``.npy`` alike) is controlled only by ``--write-mosaic``.
``--tile-size`` / ``--tile-top`` / ``--tile-left`` / ``--center`` crop the mosaic
before stats and PNG; they never write a file by themselves.

Tile sizing
-----------
``--tile-size 0`` (default) decodes the **full** mosaic. When set, ``size`` must be a
**positive even** integer (RGGB pack). The maximum is ``min(mosaic height, mosaic width)``
for the loaded frame; larger values raise a decode error. For ~18K plates, **512** is a
good spot-check (also try 256 or 1024); there is no separate upper cap beyond the frame.

Tile position
-------------
By default ``--tile-top 0`` and ``--tile-left 0`` — the crop is anchored at the **top-left**.
Pass ``--center`` with ``--tile-size`` to place the square at the **image center**; origins
are rounded down to **even** row/column indices for RGGB phase. ``--center`` cannot be
combined with non-zero ``--tile-top`` / ``--tile-left``.

Example — center crop of a full plate ``.npy`` (17712×18400), PNG beside the input::

    python -m bayer_denoiser.preview data/samples/A001C004_0101_4E490B.npy --tile-size 512 --center

For ``.npy`` inputs, if ``--write-mosaic`` is omitted, an 8-bit **grayscale** mosaic PNG is
written next to the input (same stem, ``.png``). Pass ``--mosaic-false-color`` for RGGB
site tinting (red/green/blue). For ``.raw`` (and other formats), omitting
``--write-mosaic`` prints stats only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torchvision.transforms import functional as TF

from bayer_denoiser.sort_bayer import rggb_mosaic_to_display_rgb
from bayer_denoiser.archived.io.decode import BayerTile, load_bayer, pack_mosaic
from bayer_denoiser.model import bilinear_demosaic
from bayer_denoiser.archived.io.raw_io import BayerLoadOptions, load_bayer_array, load_bayer_sidecar


def _mosaic_hw(path: Path, opts: BayerLoadOptions) -> tuple[int, int]:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        shape = np.load(path, mmap_mode="r").shape
        return int(shape[0]), int(shape[1])
    if suffix in (".raw", ".bin"):
        sidecar = load_bayer_sidecar(path)
        if sidecar and sidecar.width and sidecar.height:
            return sidecar.height, sidecar.width
        if opts.raw_width and opts.raw_height:
            return opts.raw_height, opts.raw_width
        raise ValueError(f"{path}: need sidecar or --raw-width/--raw-height for --center")
    arr = load_bayer_array(path, opts)
    return int(arr.shape[0]), int(arr.shape[1])


def _center_tile_origin(h: int, w: int, size: int) -> tuple[int, int]:
    if size > h or size > w:
        raise ValueError(f"tile size {size} exceeds mosaic {h}x{w}")
    top = (h - size) // 2
    left = (w - size) // 2
    if top % 2:
        top -= 1
    if left % 2:
        left -= 1
    return top, left


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path, help="Bayer file (.raw + .raw.json, or .npy, …)")
    p.add_argument(
        "--write-mosaic",
        type=Path,
        default=None,
        help="Save normalized mosaic as 8-bit PNG. Default output is single-channel "
        "grayscale; use --mosaic-false-color for RGGB site tinting. For .npy, default "
        "path is INPUT.png when this flag is omitted; for .raw, PNG is written only when set.",
    )
    p.add_argument(
        "--mosaic-grayscale",
        action="store_true",
        help="Write mosaic PNG as grayscale (default; explicit for scripts)",
    )
    p.add_argument(
        "--mosaic-false-color",
        action="store_true",
        help="Tint R/G/B Bayer sites red/green/blue in the mosaic PNG (not demosaic)",
    )
    p.add_argument(
        "--write-rgb",
        type=Path,
        default=None,
        help="Save bilinear-demosaiced RGB PNG.",
    )
    p.add_argument(
        "--tile-size",
        type=int,
        default=0,
        help="Optional even crop size before stats/PNG (default 0 = full mosaic)",
    )
    p.add_argument("--tile-top", type=int, default=0, help="Crop origin row (default 0 = top)")
    p.add_argument("--tile-left", type=int, default=0, help="Crop origin column (default 0 = left)")
    p.add_argument(
        "--center",
        action="store_true",
        help="With --tile-size, center the crop (even top/left); ignores default 0,0 anchor",
    )
    p.add_argument("--black-level", type=float, default=0.0)
    p.add_argument("--white-level", type=float, default=None)
    p.add_argument("--raw-width", type=int, default=None)
    p.add_argument("--raw-height", type=int, default=None)
    p.add_argument("--raw-dtype", type=str, default="uint16")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.center and not args.tile_size:
        raise SystemExit("--center requires --tile-size")
    if args.center and (args.tile_top or args.tile_left):
        raise SystemExit("--center cannot be used with non-zero --tile-top/--tile-left")
    if args.mosaic_grayscale and args.mosaic_false_color:
        raise SystemExit("Use only one of --mosaic-grayscale and --mosaic-false-color")
    opts = BayerLoadOptions(
        black_level=args.black_level,
        white_level=args.white_level,
        raw_width=args.raw_width,
        raw_height=args.raw_height,
        raw_dtype=args.raw_dtype,
    )
    tile = None
    if args.tile_size:
        top, left = args.tile_top, args.tile_left
        if args.center:
            h, w = _mosaic_hw(args.input, opts)
            top, left = _center_tile_origin(h, w, args.tile_size)
        tile = BayerTile(top=top, left=left, size=args.tile_size)

    loaded = load_bayer(args.input, options=opts, pack=True, tile=tile)
    m = loaded.mosaic
    p = loaded.packed
    assert p is not None

    print(f"path:     {loaded.path}")
    print(f"mosaic:   {tuple(m.shape)} float32  min={m.min():.6f} max={m.max():.6f} mean={m.mean():.6f}")
    print(f"packed:   {tuple(p.shape)} float32  min={p.min():.6f} max={p.max():.6f}")

    if args.write_mosaic is not None:
        mosaic_png = args.write_mosaic
    elif args.input.suffix.lower() == ".npy":
        mosaic_png = args.input.with_suffix(".png")
    else:
        mosaic_png = None

    if mosaic_png is not None:
        if args.mosaic_false_color:
            out = rggb_mosaic_to_display_rgb(m)
            mode = "RGGB false-color"
        else:
            out = m.unsqueeze(0)
            mode = "grayscale CFA"
        mosaic_png.parent.mkdir(parents=True, exist_ok=True)
        TF.to_pil_image(out).save(mosaic_png)
        print(f"wrote mosaic preview ({mode}): {mosaic_png} ({m.shape[1]}x{m.shape[0]} px)")

    if args.write_rgb is not None:
        packed_rgb = pack_mosaic(m)
        rgb = bilinear_demosaic(packed_rgb.unsqueeze(0)).squeeze(0)
        args.write_rgb.parent.mkdir(parents=True, exist_ok=True)
        TF.to_pil_image(rgb).save(args.write_rgb)
        print(f"wrote RGB preview: {args.write_rgb} ({rgb.shape[2]}x{rgb.shape[1]} px)")
    elif args.input.suffix.lower() in (".raw", ".bin"):
        suggested = args.input.with_suffix(".mosaic.png")
        print(
            "No PNG written (stats only). Pass --write-mosaic PATH.png to save an 8-bit mosaic, e.g.\n"
            f"  --write-mosaic {suggested}"
        )


if __name__ == "__main__":
    main()
