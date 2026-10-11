#!/usr/bin/env python3
"""Static noisy sequence: one clean Bayer mosaic + independent AWGN per frame.

Every frame shows the same scene (the clean mosaic); only the noise changes.
Disk format is float16 H×W mosaic in [0, 1]; noise math is float32 inside.

Clean source (pick one):
  --clean gaussian_GT_crop.npy                 existing Gaussian GT (train/ root)
  --input noisy.npy --sigma 1.0                build GT with make_bayer_gaussian_gt.py logic

Writes:
  <output-dir>/clean_mosaic.npy   (float16, once)
  <output-dir>/frame_0000.npy ... (float16, one per frame)
  <output-dir>/manifest.json      (with --write-manifest)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


def build_clean_from_input(input_path: Path, sigma: float, tile_size: int, backend: str) -> np.ndarray:
    from make_bayer_gaussian_gt import center_crop, gaussian_clean_mosaic

    mosaic = np.load(input_path)
    print(f"loaded {input_path} shape={mosaic.shape} dtype={mosaic.dtype}")
    if tile_size > 0:
        mosaic = center_crop(mosaic, tile_size)
        print(f"center crop tile-size={tile_size} -> {mosaic.shape}")
    return gaussian_clean_mosaic(mosaic, sigma, backend)


def noisy_frame(clean_f32: np.ndarray, noise_std: float, rng: np.random.Generator) -> np.ndarray:
    noise = rng.standard_normal(clean_f32.shape, dtype=np.float32) * np.float32(noise_std)
    return np.clip(clean_f32 + noise, 0.0, 1.0).astype(np.float16)


def psnr_f32(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
    return 99.0 if mse <= 0 else 10.0 * float(np.log10(1.0 / mse))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--clean", type=Path, help="Clean H×W mosaic .npy (e.g. Gaussian GT)")
    src.add_argument("--input", type=Path, help="Noisy/source .npy; Gaussian GT is built with --sigma")
    p.add_argument("--sigma", type=float, default=1.0, help="Gaussian sigma when using --input (default 1.0)")
    p.add_argument("--tile-size", type=int, default=0, help="With --input: RGGB-aligned center crop (0 = full)")
    p.add_argument("--backend", choices=("auto", "scipy", "torch"), default="auto")
    p.add_argument("--output-dir", type=Path, required=True, help="Directory for clean + frames")
    p.add_argument("--frames", type=int, default=20, help="Number of frames (default 20)")
    p.add_argument("--noise-std", type=float, default=0.02, help="AWGN std on mosaic in [0,1] (default 0.02)")
    p.add_argument("--seed", type=int, default=0, help="Base RNG seed (default 0); frame i gets its own stream")
    p.add_argument("--write-manifest", action="store_true", help="Write manifest.json in output-dir")
    args = p.parse_args()

    if args.frames < 1:
        p.error("--frames must be >= 1")
    if args.noise_std < 0:
        p.error("--noise-std must be >= 0")

    if args.clean is not None:
        clean = np.load(args.clean)
        print(f"loaded clean {args.clean} shape={clean.shape} dtype={clean.dtype}")
    else:
        clean = build_clean_from_input(args.input, args.sigma, args.tile_size, args.backend)

    if clean.ndim != 2 or clean.shape[0] % 2 or clean.shape[1] % 2:
        p.error(f"clean mosaic must be 2-D with even H and W, got {clean.shape}")
    clean = np.clip(clean.astype(np.float32), 0.0, 1.0).astype(np.float16)

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    clean_path = out / "clean_mosaic.npy"
    np.save(clean_path, clean)
    print(f"wrote {clean_path} shape={clean.shape} dtype={clean.dtype}")

    clean_f32 = clean.astype(np.float32)
    width = max(4, len(str(args.frames - 1)))
    child_seeds = np.random.SeedSequence(args.seed).spawn(args.frames)
    frame_paths: list[str] = []
    psnrs: list[float] = []
    for i, ss in enumerate(child_seeds):
        frame = noisy_frame(clean_f32, args.noise_std, np.random.default_rng(ss))
        path = out / f"frame_{i:0{width}d}.npy"
        np.save(path, frame)
        frame_paths.append(path.name)
        psnrs.append(psnr_f32(frame, clean))
        print(f"wrote {path.name} PSNR vs clean {psnrs[-1]:.2f} dB")

    print(f"{args.frames} frames, noise_std={args.noise_std}, mean PSNR vs clean {np.mean(psnrs):.2f} dB (float32 metric)")

    if args.write_manifest:
        manifest = {
            "type": "static_noisy_sequence",
            "output_dir": str(out.resolve()),
            "clean": clean_path.name,
            "frames": frame_paths,
            "num_frames": args.frames,
            "shape": list(clean.shape),
            "dtype": "float16",
            "noise_std": args.noise_std,
            "seed": args.seed,
            "clean_source": str(args.clean) if args.clean else str(args.input),
            "sigma": None if args.clean else args.sigma,
            "mean_psnr_db": round(float(np.mean(psnrs)), 3),
        }
        mpath = out / "manifest.json"
        mpath.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"wrote {mpath}")


if __name__ == "__main__":
    main()
