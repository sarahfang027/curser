#!/usr/bin/env python3
"""Convert ``.sky`` (csky) frames to float16 Bayer ``.npy`` mosaics in [0, 1]."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

_SKY_IO_PATH = Path(__file__).resolve().parent / "io.py"
_SKY_IO_NAME = "bayer_denoiser_scripts_sky_io"
_spec = importlib.util.spec_from_file_location(_SKY_IO_NAME, _SKY_IO_PATH)
_sky_io = importlib.util.module_from_spec(_spec)
sys.modules[_SKY_IO_NAME] = _sky_io
_spec.loader.exec_module(_sky_io)

SkyDecodeOptions = _sky_io.SkyDecodeOptions
decode_sky_mosaic = _sky_io.decode_sky_mosaic
parse_sky_file = _sky_io.parse_sky_file

def _load_sky_io():
    path = Path(__file__).resolve().parent / "io.py"
    name = "bayer_denoiser_scripts_sky_io"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod   # ← must be BEFORE exec_module
    spec.loader.exec_module(mod)
    return mod

_sky_io = _load_sky_io()
SkyDecodeOptions = _sky_io.SkyDecodeOptions
decode_sky_mosaic = _sky_io.decode_sky_mosaic
parse_sky_file = _sky_io.parse_sky_file


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True, help="Path to .sky file")
    p.add_argument("--output", type=Path, required=True, help="Output .npy path")
    p.add_argument("--black-level", type=float, default=0.0)
    p.add_argument(
        "--white-level",
        type=float,
        default=None,
        help="Linear white (default: 2^bit_depth - 1 after --value-shift)",
    )
    p.add_argument(
        "--value-shift",
        type=int,
        default=None,
        help="Right-shift uint16 samples before normalize (default: 2 for 12-bit)",
    )
    p.add_argument(
        "--write-json",
        type=Path,
        default=None,
        help="Optional sidecar with width/height/normalization metadata",
    )
    p.add_argument(
        "--dump-clip-json",
        type=Path,
        default=None,
        help="Write embedded clip JSON for inspection",
    )
    p.add_argument(
        "--no-row-deinterleave",
        action="store_true",
        help="Keep contiguous row-bank block order from the file (shows 4× vertical tiles)",
    )
    args = p.parse_args()

    info = parse_sky_file(args.input)
    if args.dump_clip_json:
        args.dump_clip_json.parent.mkdir(parents=True, exist_ok=True)
        args.dump_clip_json.write_text(json.dumps(info.clip_json, indent=2))

    opts = SkyDecodeOptions(
        black_level=args.black_level,
        white_level=args.white_level,
        value_shift=args.value_shift,
        row_deinterleave=not args.no_row_deinterleave,
    )
    mosaic = decode_sky_mosaic(args.input, info=info, options=opts)
    args.output.parent.mkdir(parents=True, exist_ok=True)	
    np.save(args.output, mosaic.astype(np.float16))

    if args.write_json:
        sidecar = {
            "width": int(mosaic.shape[1]),
            "height": int(mosaic.shape[0]),
            "dtype": "float16",
            "black_level": opts.black_level,
            "white_level": opts.white_level
            if opts.white_level is not None
            else (1 << info.bit_depth) - 1,
            "value_shift": opts.value_shift
            if opts.value_shift is not None
            else (2 if info.bit_depth == 12 else max(0, 16 - info.bit_depth)),
            "cfa_pattern": opts.cfa_pattern,
            "source_sky": str(args.input),
            "bit_depth": info.bit_depth,
            "row_stride_bytes": info.row_stride_bytes,
        }
        args.write_json.write_text(json.dumps(sidecar, indent=2) + "\n")

    print(
        f"Wrote {mosaic.shape} float16 mosaic "
        f"(min={mosaic.min():.4f} max={mosaic.max():.4f}) → {args.output}"
    )


if __name__ == "__main__":
    main()
