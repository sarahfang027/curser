#!/usr/bin/env python3
"""Run trained model: Bayer in → RGB PNG or Bayer mosaic (.npy)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torchvision.transforms import functional as TF

from bayer_denoiser.sort_bayer import bayer_mosaic_to_packed, packed_to_bayer_mosaic
from bayer_denoiser.tiled_infer import run_bayer_tiled_infer
from bayer_denoiser.device_util import default_device_string
from bayer_denoiser.model import (
    ARCHIVED_MODEL_ALIASES,
    MODEL_NAMES,
    bilinear_demosaic,
    build_model,
    is_bayer_space_model,
    raise_if_archived_model,
)

_ARCHIVED_RAW_HINT = (
    "Headerless .raw ingest is archived; use "
    "python -m bayer_denoiser.archived.io.decode or convert to .npy first."
)


def _load_packed_bayer(
    path: Path,
    *,
    black_level: float,
    white_level: float | None,
    raw_width: int | None,
    raw_height: int | None,
    raw_dtype: str,
) -> torch.Tensor:
    """Forward path: .npy mosaics; legacy formats via archived decode."""
    suffix = path.suffix.lower()
    if suffix == ".npy":
        mosaic_np = np.load(path)
        if mosaic_np.ndim == 3:
            mosaic_np = mosaic_np[..., 0]
        mosaic = torch.from_numpy(np.asarray(mosaic_np, dtype=np.float32))
        if float(mosaic.max()) > 1.0:
            mosaic = mosaic / 255.0
        return bayer_mosaic_to_packed(mosaic.unsqueeze(0).unsqueeze(0)).squeeze(0)

    if suffix == ".raw":
        print(_ARCHIVED_RAW_HINT, file=sys.stderr)

    from bayer_denoiser.archived.io.decode import load_bayer
    from bayer_denoiser.archived.io.raw_io import BayerLoadOptions

    options = BayerLoadOptions(
        black_level=black_level,
        white_level=white_level,
        raw_width=raw_width,
        raw_height=raw_height,
        raw_dtype=raw_dtype,
    )
    loaded = load_bayer(path, options=options, pack=True)
    if loaded.packed is None:
        raise SystemExit("internal error: pack=True but packed is None")
    return loaded.packed


def _load_mosaic_hw(path: Path) -> np.ndarray:
    """H×W float32 mosaic in [0, 1] from .npy (forward path)."""
    mosaic_np = np.load(path)
    if mosaic_np.ndim == 3:
        mosaic_np = mosaic_np[..., 0]
    mosaic = np.asarray(mosaic_np, dtype=np.float32)
    if float(mosaic.max()) > 1.0:
        mosaic = mosaic / 255.0
    return mosaic


def _resolve_tile_size(
    args_tile: int | None,
    ckpt: dict | None,
    mosaic_h: int,
    mosaic_w: int,
) -> int:
    if args_tile is not None:
        return int(args_tile)
    if ckpt is not None:
        ckpt_crop = ckpt.get("crop_size")
        if ckpt_crop is not None:
            return int(ckpt_crop)
    return min(512, mosaic_h, mosaic_w)


def _should_use_tiling(
    *,
    tiled_flag: bool,
    mosaic_h: int,
    mosaic_w: int,
    tile_size: int,
) -> bool:
    if tiled_flag:
        return True
    return mosaic_h > tile_size or mosaic_w > tile_size


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Bayer .npy (forward path); .raw/.dng/PNG/TIF use archived decode (see stderr for .raw)",
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path: .png for RGB (demosaic models) or .npy for Bayer mosaic (bayer_denoise)",
    )
    p.add_argument("--baseline", action="store_true", help="Bilinear demosaic only (no checkpoint weights)")
    p.add_argument(
        "--model",
        choices=(*MODEL_NAMES, *sorted(ARCHIVED_MODEL_ALIASES)),
        default=None,
        help="Override model type if checkpoint lacks it (gcp is archived/deferred)",
    )
    p.add_argument("--nf", type=int, default=None, help="Override denoiser scale (ckpt nf / base_channels)")
    p.add_argument("--noise-std", type=float, default=None, help="Unused for bayer_denoise; kept for ckpt compat")
    p.add_argument("--black-level", type=float, default=0.0)
    p.add_argument("--white-level", type=float, default=None)
    p.add_argument("--raw-width", type=int, default=None)
    p.add_argument("--raw-height", type=int, default=None)
    p.add_argument("--raw-dtype", type=str, default="uint16")
    p.add_argument("--device", default=default_device_string())
    p.add_argument(
        "--tiled",
        action="store_true",
        help="Sliding-window infer (auto-enabled when mosaic exceeds --tile-size)",
    )
    p.add_argument(
        "--tile-size",
        type=int,
        default=None,
        help="Even mosaic edge per tile (default: checkpoint crop_size or 512)",
    )
    p.add_argument(
        "--overlap",
        type=int,
        default=64,
        help="Blend overlap in pixels between tiles (0 = abutting grid; default 64)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.model:
        try:
            raise_if_archived_model(args.model)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    device = torch.device(args.device)
    suffix_in = args.input.suffix.lower()

    with torch.no_grad():
        if args.baseline:
            packed = _load_packed_bayer(
                args.input,
                black_level=args.black_level,
                white_level=args.white_level,
                raw_width=args.raw_width,
                raw_height=args.raw_height,
                raw_dtype=args.raw_dtype,
            ).unsqueeze(0).to(device)
            if is_bayer_space_model(args.model or ""):
                raise SystemExit("--baseline is for RGB demosaic models only")
            rgb = bilinear_demosaic(packed)
            out_bayer = None
        else:
            ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
            model_name = args.model or ckpt.get("model", "bayer_denoise")
            try:
                raise_if_archived_model(model_name)
            except ValueError as exc:
                raise SystemExit(str(exc)) from exc
            nf = args.nf if args.nf is not None else int(ckpt.get("nf", 64))
            base_channels = int(ckpt.get("base_channels", max(16, nf // 2)))
            model = build_model(model_name, nf=nf, base_channels=base_channels).to(device)
            model.load_state_dict(ckpt["model_state"])
            model.eval()
            if is_bayer_space_model(model_name):
                if suffix_in == ".npy":
                    mosaic_hw = _load_mosaic_hw(args.input)
                    h, w = mosaic_hw.shape
                    tile_size = _resolve_tile_size(args.tile_size, ckpt, h, w)
                    if tile_size % 2:
                        raise SystemExit("--tile-size must be even (Bayer mosaic)")
                    use_tiled = _should_use_tiling(
                        tiled_flag=args.tiled,
                        mosaic_h=h,
                        mosaic_w=w,
                        tile_size=tile_size,
                    )
                    if use_tiled:
                        if args.overlap < 0 or args.overlap >= tile_size:
                            raise SystemExit("--overlap must be in [0, tile_size)")
                        out_bayer = run_bayer_tiled_infer(
                            model,
                            mosaic_hw,
                            device=device,
                            tile_size=tile_size,
                            overlap=int(args.overlap),
                        )
                        print(
                            f"Tiled infer: {h}x{w} mosaic, tile={tile_size}, overlap={args.overlap}"
                        )
                    else:
                        packed = bayer_mosaic_to_packed(
                            torch.from_numpy(mosaic_hw).unsqueeze(0).unsqueeze(0).to(device)
                        )
                        denoised = model(packed)
                        out_bayer = (
                            packed_to_bayer_mosaic(denoised).squeeze(0).squeeze(0).cpu().numpy()
                        )
                else:
                    packed = _load_packed_bayer(
                        args.input,
                        black_level=args.black_level,
                        white_level=args.white_level,
                        raw_width=args.raw_width,
                        raw_height=args.raw_height,
                        raw_dtype=args.raw_dtype,
                    ).unsqueeze(0).to(device)
                    denoised = model(packed)
                    out_bayer = (
                        packed_to_bayer_mosaic(denoised).squeeze(0).squeeze(0).cpu().numpy()
                    )
                rgb = None
            else:
                packed = _load_packed_bayer(
                    args.input,
                    black_level=args.black_level,
                    white_level=args.white_level,
                    raw_width=args.raw_width,
                    raw_height=args.raw_height,
                    raw_dtype=args.raw_dtype,
                ).unsqueeze(0).to(device)
                rgb = model(packed)
                out_bayer = None

    args.output.parent.mkdir(parents=True, exist_ok=True)
    suffix = args.output.suffix.lower()
    if out_bayer is not None:
        if suffix not in (".npy", ".raw"):
            raise SystemExit(
                f"bayer_denoise checkpoint requires --output ending in .npy (got {args.output})"
            )
        np.save(args.output, out_bayer.astype(np.float16))
        print(f"Wrote Bayer mosaic {out_bayer.shape} to {args.output}")
        return

    if rgb is None:
        raise SystemExit("internal error: no RGB output")
    if suffix not in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
        raise SystemExit(f"RGB infer expects image extension, got {args.output}")
    out = rgb.squeeze(0).cpu()
    TF.to_pil_image(out).save(args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
