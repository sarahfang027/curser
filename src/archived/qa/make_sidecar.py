#!/usr/bin/env python3
"""Write canonical plate.raw.json from the project ingest contract."""

from __future__ import annotations

import argparse
from pathlib import Path

from bayer_denoiser.archived.io.raw_spec import PROJECT_RAW_SPEC, expected_raw_bytes, write_canonical_sidecar


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("raw_file", type=Path, help="Path to .raw (sidecar name derived from this)")
    p.add_argument("--width", type=int, default=PROJECT_RAW_SPEC.width)
    p.add_argument("--height", type=int, default=PROJECT_RAW_SPEC.height)
    p.add_argument("--dtype", type=str, default=PROJECT_RAW_SPEC.dtype)
    p.add_argument("--black-level", type=float, default=PROJECT_RAW_SPEC.black_level)
    p.add_argument("--white-level", type=float, default=PROJECT_RAW_SPEC.white_level)
    p.add_argument("--endian", choices=("little", "big"), default=PROJECT_RAW_SPEC.endian)
    args = p.parse_args()

    from bayer_denoiser.archived.io.raw_spec import RawIngestSpec

    spec = RawIngestSpec(
        width=args.width,
        height=args.height,
        dtype=args.dtype,
        black_level=args.black_level,
        white_level=args.white_level,
        endian=args.endian,
    )
    out = write_canonical_sidecar(args.raw_file, spec)
    nbytes = expected_raw_bytes(spec.width, spec.height, spec.dtype)
    print(f"Wrote {out}")
    print(f"Expected .raw size: {nbytes:,} bytes ({nbytes / (1024**2):.1f} MiB)")


if __name__ == "__main__":
    main()
