#!/usr/bin/env python3
"""Fail-fast validation of headerless Bayer .raw files against the ingest contract."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bayer_denoiser.archived.io.raw_spec import (
    PROJECT_RAW_SPEC,
    RawIngestError,
    RawIngestSpec,
    expected_raw_bytes,
    validate_raw_file,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("paths", nargs="*", type=Path, help=".raw file(s)")
    p.add_argument("--dir", type=Path, help="Validate all *.raw in directory (non-recursive)")
    p.add_argument(
        "--recursive",
        action="store_true",
        help="With --dir, include subdirectories",
    )
    p.add_argument(
        "--no-strict",
        action="store_true",
        help="Only require sidecar, even geometry, and exact byte size (ignore project W/H/dtype defaults)",
    )
    p.add_argument(
        "--no-stem-check",
        action="store_true",
        help="Allow any filename stem",
    )
    return p.parse_args()


def collect_raw_paths(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = list(args.paths)
    if args.dir:
        d = Path(args.dir)
        if not d.is_dir():
            raise SystemExit(f"Not a directory: {d}")
        glob = d.rglob("*.raw") if args.recursive else d.glob("*.raw")
        paths.extend(sorted(glob))
    if not paths:
        raise SystemExit("Provide .raw path(s) or --dir")
    return paths


def main() -> None:
    args = parse_args()
    spec: RawIngestSpec = PROJECT_RAW_SPEC
    paths = collect_raw_paths(args)
    ok = 0
    for raw_path in paths:
        try:
            meta = validate_raw_file(
                raw_path,
                spec=spec,
                strict_spec=not args.no_strict,
                check_stem=not args.no_stem_check,
            )
            nbytes = expected_raw_bytes(meta.width, meta.height, meta.dtype)
            print(
                f"[OK] {raw_path.name}  {meta.width}×{meta.height}  "
                f"{meta.dtype}  {nbytes:,} bytes"
            )
            ok += 1
        except RawIngestError as exc:
            print(f"[FAIL] {exc}", file=sys.stderr)

    print(f"{ok}/{len(paths)} passed")
    if ok != len(paths):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
