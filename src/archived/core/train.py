#!/usr/bin/env python3
"""Train Bayer-space denoiser (or small RGB demosaic baseline)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Paired/synthetic loaders live in archived/core (stock train + smoke only).
from bayer_denoiser.archived.core.dataset import (
    PairedRawBayerDataset,
    RgbFolderDataset,
    SyntheticBayerDataset,
    SyntheticBayerDenoiseDataset,
)
from bayer_denoiser.device_util import default_device_string
from bayer_denoiser.model import (
    ARCHIVED_MODEL_ALIASES,
    MODEL_NAMES,
    build_model,
    is_bayer_space_model,
    raise_if_archived_model,
)
from bayer_denoiser.archived.io.raw_io import BayerLoadOptions


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    mode = p.add_argument_group("data source (pick one)")
    mode.add_argument("--data-dir", type=Path, default=None, help="RGB folder; Bayer is simulated")
    mode.add_argument("--bayer-dir", type=Path, default=None, help="Raw Bayer mosaics (paired with --rgb-dir)")
    mode.add_argument("--rgb-dir", type=Path, default=None, help="Reference RGB for each Bayer frame")
    mode.add_argument("--pairs-csv", type=Path, default=None, help="CSV with columns: bayer,rgb")

    raw = p.add_argument_group("raw Bayer decoding")
    raw.add_argument("--black-level", type=float, default=0.0)
    raw.add_argument(
        "--white-level",
        type=float,
        default=None,
        help="Upper bound for normalization; auto from bit depth if unset",
    )
    raw.add_argument("--raw-width", type=int, default=None, help="Flat .raw width if no sidecar JSON")
    raw.add_argument("--raw-height", type=int, default=None, help="Flat .raw height if no sidecar JSON")
    raw.add_argument("--raw-dtype", type=str, default="uint16", help="Flat .raw dtype: float16, uint16, …")
    raw.add_argument("--resize", type=int, default=0, help="Optional even resize before crop (raw mode)")

    p.add_argument(
        "--model",
        choices=(*MODEL_NAMES, *sorted(ARCHIVED_MODEL_ALIASES)),
        default="bayer_denoise",
        help="Network architecture (gcp is archived/deferred)",
    )
    p.add_argument("--nf", type=int, default=64, help="BayerDenoiseNet scale (nf//2 → base channels)")
    p.add_argument("--output-dir", type=Path, default=Path("runs/default"))
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--image-size", type=int, default=128, help="Auto-crop size from full plates (raw mode); resize for RGB-folder mode")
    p.add_argument(
        "--crops-per-frame",
        type=int,
        default=1,
        help="Random crops drawn per full plate each epoch (raw mode only)",
    )
    p.add_argument(
        "--full-frame",
        action="store_true",
        help="Disable auto-crop; train on full resolution (raw mode)",
    )
    p.add_argument(
        "--no-random-crop",
        action="store_true",
        help="Center crop instead of random (raw mode)",
    )
    p.add_argument("--synthetic-length", type=int, default=512, help="Samples per epoch if no data-dir")
    p.add_argument("--noise-std", type=float, default=0.02)
    p.add_argument("--lr", type=float, default=1e-4, help="Adam LR")
    p.add_argument("--device", default=default_device_string())
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = torch.mean((pred - target) ** 2).item()
    if mse <= 0:
        return 99.0
    return 10.0 * torch.log10(torch.tensor(1.0 / mse)).item()


def build_dataset(args: argparse.Namespace) -> torch.utils.data.Dataset:
    bayer_space = is_bayer_space_model(args.model)
    raw_mode = bool(args.bayer_dir or args.pairs_csv)
    if raw_mode:
        if args.bayer_dir and not args.rgb_dir and not args.pairs_csv:
            raise SystemExit("--bayer-dir requires --rgb-dir (or use --pairs-csv)")
        if args.rgb_dir and not args.bayer_dir and not args.pairs_csv:
            raise SystemExit("--rgb-dir requires --bayer-dir (or use --pairs-csv)")
        if args.data_dir:
            raise SystemExit("Use either --data-dir or raw pairing flags, not both")

        crop_size = 0 if args.full_frame else args.image_size
        options = BayerLoadOptions(
            black_level=args.black_level,
            white_level=args.white_level,
            raw_width=args.raw_width,
            raw_height=args.raw_height,
            raw_dtype=args.raw_dtype,
        )
        return PairedRawBayerDataset(
            bayer_dir=args.bayer_dir,
            rgb_dir=args.rgb_dir,
            pairs_csv=args.pairs_csv,
            bayer_options=options,
            crop_size=crop_size,
            random_crop=not args.no_random_crop,
            crops_per_frame=1 if args.full_frame else args.crops_per_frame,
            noise_std=args.noise_std,
            resize=args.resize,
            bayer_target=bayer_space,
        )

    if args.data_dir:
        if args.bayer_dir or args.rgb_dir or args.pairs_csv:
            raise SystemExit("Simulated RGB folder cannot be combined with raw pairing flags")
        if bayer_space:
            raise SystemExit("--model bayer_denoise requires synthetic or paired raw Bayer, not --data-dir")
        return RgbFolderDataset(
            args.data_dir,
            image_size=args.image_size,
            noise_std=args.noise_std,
        )

    if args.bayer_dir or args.rgb_dir or args.pairs_csv:
        raise SystemExit("Incomplete raw pairing; need --bayer-dir and --rgb-dir or --pairs-csv")

    if bayer_space:
        return SyntheticBayerDenoiseDataset(
            length=args.synthetic_length,
            image_size=args.image_size,
            noise_std=args.noise_std,
            seed=args.seed,
        )

    return SyntheticBayerDataset(
        length=args.synthetic_length,
        image_size=args.image_size,
        noise_std=args.noise_std,
        seed=args.seed,
    )


def main() -> None:
    args = parse_args()
    try:
        raise_if_archived_model(args.model)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_dataset(args)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=args.device.startswith("cuda"),
    )

    device = torch.device(args.device)
    bayer_space = is_bayer_space_model(args.model)
    model = build_model(args.model, nf=args.nf, base_channels=max(16, args.nf // 2)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model={args.model}  params={n_params:,}  device={device}")

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.L1Loss()

    history: list[dict] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_psnr = 0.0
        n_batches = 0
        for packed, target in tqdm(loader, desc=f"epoch {epoch}/{args.epochs}"):
            packed = packed.to(device)
            target = target.to(device)
            pred = model(packed)
            loss = loss_fn(pred, target)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            running_loss += loss.item()
            running_psnr += psnr(pred.detach(), target)
            n_batches += 1

        avg_loss = running_loss / max(n_batches, 1)
        avg_psnr = running_psnr / max(n_batches, 1)
        history.append({"epoch": epoch, "loss": avg_loss, "psnr": avg_psnr})
        print(f"epoch {epoch}: loss={avg_loss:.5f} psnr={avg_psnr:.2f} dB")

    ckpt = {
        "model_state": model.state_dict(),
        "model": args.model,
        "nf": args.nf,
        "base_channels": max(16, args.nf // 2),
        "image_size": args.image_size,
        "noise_std": args.noise_std,
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }
    torch.save(ckpt, args.output_dir / "model.pt")
    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"Saved checkpoint to {args.output_dir / 'model.pt'}")


if __name__ == "__main__":
    main()
