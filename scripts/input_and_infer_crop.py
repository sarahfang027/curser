#!/usr/bin/env python3
"""Crop a full-frame Bayer mosaic to the training patch, then denoise (checkpoint crop).

Saves the cropped noisy mosaic and the denoised output as float16 .npy files.
Crop geometry defaults to fields stored in the checkpoint (crop_size, crop_y, crop_x).

Run from ~/Projects/bayer_denoiser with venv active and `pip install -e .`.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from bayer_denoiser.sort_bayer import mosaic_to_packed_np, packed_to_mosaic_np
from bayer_denoiser.device_util import default_device_string
from bayer_denoiser.model import build_model


def checkpoint_state_dict(ckpt: dict) -> dict:
    if "state_dict" in ckpt:
        return ckpt["state_dict"]
    if "model_state" in ckpt:
        return ckpt["model_state"]
    raise SystemExit("checkpoint missing state_dict or model_state (weights)")


def crop_mosaic(mosaic: np.ndarray, crop_size: int, crop_y: int, crop_x: int) -> np.ndarray:
    return mosaic[crop_y : crop_y + crop_size, crop_x : crop_x + crop_size]


def resolve_crop(
    h: int,
    w: int,
    crop_size: int,
    crop_y: int | None,
    crop_x: int | None,
) -> tuple[int, int]:
    if crop_size % 2 != 0:
        raise SystemExit("--crop-size must be even (Bayer mosaic)")
    if crop_size > h or crop_size > w:
        raise SystemExit(f"crop {crop_size} exceeds mosaic {h}x{w}")
    if crop_y is None and crop_x is None:
        crop_y = (h - crop_size) // 2
        crop_x = (w - crop_size) // 2
    elif crop_y is None or crop_x is None:
        raise SystemExit("pass both --crop-y and --crop-x, or neither for center crop")
    if crop_y % 2 != 0 or crop_x % 2 != 0:
        raise SystemExit("--crop-y and --crop-x must be even")
    if crop_y < 0 or crop_x < 0 or crop_y + crop_size > h or crop_x + crop_size > w:
        raise SystemExit(f"crop ({crop_y},{crop_x})+{crop_size} outside {h}x{w}")
    return crop_y, crop_x


def crop_from_checkpoint(
    ckpt: dict,
    h: int,
    w: int,
    crop_size: int | None,
    crop_y: int | None,
    crop_x: int | None,
) -> tuple[int, int, int]:
    ckpt_size = ckpt.get("crop_size")
    if crop_size is None:
        if ckpt_size is None:
            raise SystemExit(
                "checkpoint has no crop_size (full-frame train). "
                "Pass --crop-size (and optional --crop-y / --crop-x)."
            )
        crop_size = int(ckpt_size)
    elif ckpt_size is not None and int(ckpt_size) != crop_size:
        raise SystemExit(f"--crop-size {crop_size} does not match checkpoint {ckpt_size}")

    if crop_y is None and crop_x is None and ckpt.get("crop_y") is not None:
        crop_y = int(ckpt["crop_y"])
        crop_x = int(ckpt["crop_x"])

    cy, cx = resolve_crop(h, w, crop_size, crop_y, crop_x)
    return crop_size, cy, cx


def default_output_dir(checkpoint: Path) -> Path:
    """RUN/compare when checkpoint lives in RUN (or legacy RUN/train); else {parent}/compare."""
    parent = checkpoint.resolve().parent
    if parent.name == "train":
        return parent.parent / "compare"
    return parent / "compare"


def denoise_mosaic(mosaic: np.ndarray, ckpt: dict, device: torch.device) -> np.ndarray:
    nf = int(ckpt.get("nf", 64))
    model = build_model("bayer_denoise", nf=nf).to(device)
    model.load_state_dict(checkpoint_state_dict(ckpt))
    model.eval()
    with torch.no_grad():
        packed = torch.from_numpy(mosaic_to_packed_np(mosaic)).unsqueeze(0).float().to(device)
        out_p = model(packed)
        return packed_to_mosaic_np(out_p.squeeze(0).cpu().numpy())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, required=True, help="full-frame or any H×W noisy mosaic .npy")
    p.add_argument("--checkpoint", type=Path, required=True, help="model.pt or model_epochXXXX.pt")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "writes cropped_input.npy + denoised_cropped.npy "
            "(default: {checkpoint.parent}/compare — RUN/compare when checkpoint is at RUN root; legacy RUN/train/ → RUN/compare)"
        ),
    )
    p.add_argument("--cropped-input", type=Path, default=None, help="override cropped noisy output path")
    p.add_argument("--denoised-output", type=Path, default=None, help="override denoised output path")
    p.add_argument("--crop-size", type=int, default=None)
    p.add_argument("--crop-y", type=int, default=None)
    p.add_argument("--crop-x", type=int, default=None)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    if not args.input.is_file():
        raise SystemExit(f"missing input: {args.input}")
    if not args.checkpoint.is_file():
        raise SystemExit(f"missing checkpoint: {args.checkpoint}")

    device = torch.device(args.device or default_device_string())
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    if ckpt.get("model") != "bayer_denoise":
        raise SystemExit(f"expected bayer_denoise checkpoint, got {ckpt.get('model')}")

    mosaic = np.load(args.input).astype(np.float16)
    h, w = mosaic.shape
    crop_size, crop_y, crop_x = crop_from_checkpoint(ckpt, h, w, args.crop_size, args.crop_y, args.crop_x)
    cropped = crop_mosaic(mosaic, crop_size, crop_y, crop_x)

    out_dir = args.output_dir or default_output_dir(args.checkpoint)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.input.stem
    cropped_path = args.cropped_input or (out_dir / f"{stem}_cropped_{crop_size}.npy")
    denoised_path = args.denoised_output or (out_dir / f"{stem}_denoised_{crop_size}.npy")

    np.save(cropped_path, cropped.astype(np.float16))
    print(f"full mosaic: {h}x{w}  crop: {crop_size}x{crop_size} at (y={crop_y}, x={crop_x})")
    print(f"cropped input: {cropped_path}")

    denoised = denoise_mosaic(cropped, ckpt, device)
    np.save(denoised_path, denoised.astype(np.float16))
    print(f"denoised output: {denoised_path}  device: {device}")


if __name__ == "__main__":
    main()
