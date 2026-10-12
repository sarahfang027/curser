#!/usr/bin/env python3
"""Visual check: auto-crops from full Bayer + RGB pairs (same logic as training)."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import functional as TF

from bayer_denoiser.archived.core.dataset import (
    _frame_size_for_pair,
    _load_bayer_crop,
    _load_rgb_crop,
    choose_crop,
)
from bayer_denoiser.archived.io.decode import pack_mosaic
from bayer_denoiser.model import bilinear_demosaic
from bayer_denoiser.archived.qa.pairs import BayerRgbPair, discover_pairs, load_pairs_csv
from bayer_denoiser.archived.io.raw_io import BayerLoadOptions


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = p.add_mutually_exclusive_group()
    src.add_argument("--bayer", type=Path, help="Single Bayer file (with --rgb)")
    src.add_argument("--pairs-csv", type=Path, help="CSV manifest bayer,rgb")
    p.add_argument("--rgb", type=Path, help="RGB reference (with --bayer)")
    p.add_argument("--bayer-dir", type=Path)
    p.add_argument("--rgb-dir", type=Path)
    p.add_argument("--pair-index", type=int, default=0, help="Which pair when using dirs or CSV")

    p.add_argument("--crop-size", type=int, default=256)
    p.add_argument("--num-crops", type=int, default=4, help="Number of random crops to render")
    p.add_argument("--seed", type=int, default=0, help="RNG seed for reproducible crop origins")
    p.add_argument("--center", action="store_true", help="Center crop instead of random")
    p.add_argument("--top", type=int, default=None, help="Fixed crop top (even); overrides random")
    p.add_argument("--left", type=int, default=None, help="Fixed crop left (even)")
    p.add_argument("--output-dir", type=Path, default=Path("runs/crop_visual_check"))
    p.add_argument("--overview", action="store_true", help="Save downscaled RGB with crop box (loads full RGB)")
    p.add_argument("--overview-max", type=int, default=1200, help="Long edge for overview image")
    p.add_argument("--black-level", type=float, default=0.0)
    p.add_argument("--white-level", type=float, default=None)
    p.add_argument("--raw-width", type=int, default=None)
    p.add_argument("--raw-height", type=int, default=None)
    p.add_argument("--raw-dtype", type=str, default="uint16")
    return p.parse_args()


def resolve_pair(args: argparse.Namespace) -> BayerRgbPair:
    if args.bayer:
        if not args.rgb:
            raise SystemExit("--rgb is required with --bayer")
        return BayerRgbPair(bayer=args.bayer, rgb=args.rgb)
    if args.pairs_csv:
        pairs = load_pairs_csv(args.pairs_csv)
        return pairs[args.pair_index]
    if args.bayer_dir and args.rgb_dir:
        pairs = discover_pairs(args.bayer_dir, args.rgb_dir)
        return pairs[args.pair_index]
    raise SystemExit("Provide --bayer + --rgb, --pairs-csv, or --bayer-dir + --rgb-dir")


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = torch.mean((pred - target) ** 2).item()
    if mse <= 0:
        return 99.0
    return 10.0 * torch.log10(torch.tensor(1.0 / mse)).item()


def mosaic_to_pil(mosaic_hw: torch.Tensor) -> Image.Image:
    return TF.to_pil_image(mosaic_hw.unsqueeze(0).clamp(0, 1))


def rgb_to_pil(rgb: torch.Tensor) -> Image.Image:
    return TF.to_pil_image(rgb.clamp(0, 1))


def diff_heatmap(rgb: torch.Tensor, pred: torch.Tensor) -> Image.Image:
    d = (rgb - pred).abs().mean(dim=0, keepdim=True)
    d = (d / d.max().clamp(min=1e-6)).clamp(0, 1)
    return TF.to_pil_image(d)


def make_strip(
    labels: list[str],
    images: list[Image.Image],
    gap: int = 8,
) -> Image.Image:
    w = max(im.width for im in images)
    h = max(im.height for im in images)
    label_h = 22
    total_w = len(images) * w + (len(images) + 1) * gap
    total_h = label_h + h + gap
    canvas = Image.new("RGB", (total_w, total_h), (32, 32, 32))
    draw = ImageDraw.Draw(canvas)
    x = gap
    for label, im in zip(labels, images):
        canvas.paste(im.resize((w, h)), (x, label_h))
        draw.text((x + 4, 2), label, fill=(220, 220, 220))
        x += w + gap
    return canvas


def save_overview(
    rgb_path: Path,
    *,
    top: int,
    left: int,
    crop_size: int,
    out_path: Path,
    max_edge: int,
) -> None:
    from bayer_denoiser.archived.core.dataset import _load_rgb_full

    rgb = _load_rgb_full(rgb_path)
    _, h, w = rgb.shape
    scale = min(1.0, max_edge / max(h, w))
    nh, nw = int(h * scale), int(w * scale)
    thumb = TF.resize(rgb, [nh, nw], antialias=True)
    pil = rgb_to_pil(thumb)
    draw = ImageDraw.Draw(pil)
    x0 = int(left * scale)
    y0 = int(top * scale)
    x1 = int((left + crop_size) * scale)
    y1 = int((top + crop_size) * scale)
    draw.rectangle([x0, y0, x1, y1], outline=(255, 64, 64), width=max(2, int(3 * scale)))
    draw.text((x0 + 4, y0 + 4), f"({left},{top}) {crop_size}px", fill=(255, 64, 64))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pil.save(out_path)


def crop_origin(
    h: int,
    w: int,
    crop_size: int,
    args: argparse.Namespace,
    sample_idx: int,
) -> tuple[int, int]:
    if args.top is not None and args.left is not None:
        return args.top, args.left
    if args.center:
        return choose_crop(h, w, crop_size, random_crop=False)
    torch.manual_seed(args.seed + sample_idx * 7919)
    return choose_crop(h, w, crop_size, random_crop=True)


def main() -> None:
    args = parse_args()
    if args.crop_size % 2:
        raise SystemExit("--crop-size must be even (RGGB)")
    if args.top is not None and args.left is None:
        raise SystemExit("Provide both --top and --left for a fixed crop")
    if args.left is not None and args.top is None:
        raise SystemExit("Provide both --top and --left for a fixed crop")

    pair = resolve_pair(args)
    opts = BayerLoadOptions(
        black_level=args.black_level,
        white_level=args.white_level,
        raw_width=args.raw_width,
        raw_height=args.raw_height,
        raw_dtype=args.raw_dtype,
    )
    h, w = _frame_size_for_pair(pair, opts)
    if args.crop_size > h or args.crop_size > w:
        raise SystemExit(f"crop-size {args.crop_size} exceeds frame {h}x{w}")

    out_root = args.output_dir / pair.bayer.stem
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"pair: {pair.bayer.name} ↔ {pair.rgb.name}  frame {h}x{w}  crop {args.crop_size}")

    summary_lines: list[str] = []

    for i in range(args.num_crops):
        top, left = crop_origin(h, w, args.crop_size, args, i)
        bayer = _load_bayer_crop(pair.bayer, opts, top=top, left=left, size=args.crop_size)
        rgb = _load_rgb_crop(pair.rgb, top=top, left=left, size=args.crop_size)

        packed = pack_mosaic(bayer).unsqueeze(0)
        demosaic = bilinear_demosaic(packed).squeeze(0)
        mae = torch.mean((demosaic - rgb).abs()).item()
        p = psnr(demosaic, rgb)

        mosaic_pil = mosaic_to_pil(bayer)
        rgb_pil = rgb_to_pil(rgb)
        dem_pil = rgb_to_pil(demosaic)
        diff_pil = diff_heatmap(rgb, demosaic)

        strip = make_strip(
            ["mosaic", "RGB ref", "bilinear", "|RGB−bil|"],
            [mosaic_pil, rgb_pil, dem_pil, diff_pil.convert("RGB")],
        )

        tag = f"crop_{i:02d}_t{top}_l{left}"
        strip.save(out_root / f"{tag}_strip.png")
        rgb_pil.save(out_root / f"{tag}_rgb.png")
        mosaic_pil.save(out_root / f"{tag}_bayer_mosaic.png")

        line = f"{tag}: top={top} left={left}  MAE(bilinear vs RGB)={mae:.5f}  PSNR={p:.2f} dB"
        summary_lines.append(line)
        print(line)

        if args.overview:
            save_overview(
                pair.rgb,
                top=top,
                left=left,
                crop_size=args.crop_size,
                out_path=out_root / f"{tag}_overview.png",
                max_edge=args.overview_max,
            )

    (out_root / "summary.txt").write_text("\n".join(summary_lines) + "\n")
    print(f"Wrote visuals under {out_root}")


if __name__ == "__main__":
    main()
