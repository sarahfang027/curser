#!/usr/bin/env python3
"""Write aligned noisy/clean crop pairs + manifest.json under train/crops/.

Also updates legacy train/noisy_input_crop.npy and train/gaussian_GT_crop.npy from crop 0.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

try:
    from bayer_denoiser.crop_pairs import (  # noqa: E402
        resolve_origins,
        write_crop_pair,
        write_manifest,
    )
except ModuleNotFoundError:
    from crop_pairs import (  # noqa: E402
        resolve_origins,
        write_crop_pair,
        write_manifest,
    )


def parse_grid(s: str) -> tuple[int, int]:
    parts = s.lower().replace("x", " ").split()
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("grid must look like 2x2 or '2 2'")
    return int(parts[0]), int(parts[1])


def build_clean_from_noisy(noisy: np.ndarray, sigma: float, backend: str) -> np.ndarray:
    from make_bayer_gaussian_gt import gaussian_clean_mosaic

    return gaussian_clean_mosaic(noisy, sigma, backend)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--noisy", type=Path, required=True, help="Full-plate or tile noisy .npy")
    p.add_argument("--clean", type=Path, default=None, help="Optional aligned clean/GT mosaic")
    p.add_argument("--sigma", type=float, default=1.0, help="Gaussian GT sigma when --clean omitted")
    p.add_argument("--backend", choices=("auto", "scipy", "torch"), default="auto")
    p.add_argument("--train-dir", type=Path, required=True, help="Run train/ directory")
    p.add_argument("--tile-size", type=int, default=512)
    p.add_argument("--num-crops", type=int, default=1)
    p.add_argument("--grid", type=parse_grid, default=None, help="e.g. 2x2 (overrides num-crops count check)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--crops-subdir", type=str, default="crops", help="Subfolder under train-dir")
    args = p.parse_args()

    if args.tile_size % 2 != 0:
        p.error("--tile-size must be even")
    if args.num_crops < 1:
        p.error("--num-crops must be >= 1")

    noisy = np.load(args.noisy)
    if noisy.ndim != 2:
        p.error(f"--noisy must be H×W, got {noisy.shape}")
    h, w = noisy.shape
    if args.clean is not None:
        clean = np.load(args.clean)
        if clean.shape != noisy.shape:
            p.error(f"clean shape {clean.shape} != noisy {noisy.shape}")
    else:
        print(f"building Gaussian GT on full mosaic (sigma={args.sigma}) …")
        clean = build_clean_from_noisy(noisy, args.sigma, args.backend)

    origins = resolve_origins(h, w, args.tile_size, args.num_crops, args.grid, args.seed)
    mode = "grid" if args.grid else ("center" if len(origins) == 1 else "random")

    crops_dir = args.train_dir / args.crops_subdir
    crops_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for i, (y, x) in enumerate(origins):
        entries.append(
            write_crop_pair(noisy, clean, crops_dir, i, args.tile_size, y, x)
        )
        print(f"crop {i}: ({y}, {x}) size={args.tile_size}")

    manifest_path = crops_dir / "manifest.json"
    write_manifest(
        manifest_path,
        tile=args.tile_size,
        source_shape=(h, w),
        mode=mode,
        seed=args.seed,
        crops=entries,
        grid=args.grid,
    )
    print(f"wrote {manifest_path} ({len(entries)} crops)")

    # Legacy single-tile paths (crop 0)
    legacy_noisy = args.train_dir / "noisy_input_crop.npy"
    legacy_clean = args.train_dir / "gaussian_GT_crop.npy"
    np.save(legacy_noisy, np.load(crops_dir / entries[0]["noisy"]))
    np.save(legacy_clean, np.load(crops_dir / entries[0]["clean"]))
    print(f"legacy: {legacy_noisy.name}, {legacy_clean.name} (crop 0)")


if __name__ == "__main__":
    main()
