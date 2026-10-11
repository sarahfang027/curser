#!/usr/bin/env python3
"""Train BayerDenoiseNet on a static noisy sequence (many noisy frames, one clean mosaic).

Expects the folder written by make_static_noisy_sequence.py:
  clean_mosaic.npy, frame_0000.npy, frame_0001.npy, ... (optional manifest.json)

Every epoch visits all frames (shuffled); each step is L1(pred, clean) in packed RGGB.
`clean` is usually Gaussian GT (pseudo ground truth), so this is a pipeline/plumbing test.
Run from repo root with venv active and `pip install -e .`:

  python -m bayer_denoiser.train_static_noisy_sequence --help
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from bayer_denoiser.sort_bayer import mosaic_to_packed_np, packed_to_mosaic_np
from bayer_denoiser.device_util import default_device_string
from bayer_denoiser.model import build_model


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
    return 99.0 if mse <= 0 else 10.0 * np.log10(1.0 / mse)


def crop_mosaic(
    mosaic: np.ndarray,
    crop_size: int,
    crop_y: int,
    crop_x: int,
) -> np.ndarray:
    return mosaic[crop_y : crop_y + crop_size, crop_x : crop_x + crop_size]


def resolve_crop(
    h: int,
    w: int,
    crop_size: int | None,
    crop_y: int | None,
    crop_x: int | None,
    rng: random.Random | None,
) -> tuple[int | None, int, int]:
    if crop_size is None:
        return None, 0, 0
    if crop_size % 2 != 0:
        raise SystemExit("--crop-size must be even (Bayer mosaic)")
    if crop_size > h or crop_size > w:
        raise SystemExit(f"--crop-size {crop_size} exceeds mosaic {h}x{w}")
    if crop_y is None and crop_x is None:
        if rng is None:
            crop_y = (h - crop_size) // 2
            crop_x = (w - crop_size) // 2
        else:
            crop_y = rng.randrange(0, h - crop_size + 1, 2)
            crop_x = rng.randrange(0, w - crop_size + 1, 2)
    elif crop_y is None or crop_x is None:
        raise SystemExit("pass both --crop-y and --crop-x, or neither for center/random crop")
    if crop_y % 2 != 0 or crop_x % 2 != 0:
        raise SystemExit("--crop-y and --crop-x must be even")
    return crop_size, crop_y, crop_x


def merge_resume_training_args(args: argparse.Namespace, ckpt: dict) -> None:
    if ckpt.get("model") != "bayer_denoise":
        raise SystemExit(f"checkpoint model must be bayer_denoise, got {ckpt.get('model')}")

    ckpt_nf = ckpt.get("nf")
    if ckpt_nf is not None and args.nf != ckpt_nf:
        raise SystemExit(f"--nf {args.nf} does not match checkpoint nf={ckpt_nf}")
    if ckpt_nf is not None:
        args.nf = int(ckpt_nf)

    ckpt_crop = ckpt.get("crop_size")
    if ckpt_crop is not None:
        if args.crop_size is not None and args.crop_size != ckpt_crop:
            raise SystemExit(f"--crop-size {args.crop_size} does not match checkpoint {ckpt_crop}")
        args.crop_size = int(ckpt_crop)
        for key in ("crop_y", "crop_x"):
            cv = ckpt.get(key)
            if cv is not None:
                av = getattr(args, key)
                if av is not None and av != cv:
                    raise SystemExit(f"--{key.replace('_', '-')} {av} does not match checkpoint {cv}")
                setattr(args, key, int(cv))
    elif args.crop_size is not None:
        print("warning: resuming checkpoint without crop metadata; using CLI crop settings")

    ckpt_random = bool(ckpt.get("random_crop"))
    if ckpt_random:
        args.random_crop = True
    elif args.random_crop and ckpt_crop is not None:
        print("warning: --random-crop set but checkpoint was trained with fixed crop")


def load_history(output_dir: Path, ckpt: dict) -> tuple[list, int]:
    history_path = output_dir / "history.json"
    history: list = []
    if history_path.is_file():
        history = json.loads(history_path.read_text())
    if history:
        start_epoch = int(history[-1]["epoch"]) + 1
    else:
        start_epoch = int(ckpt.get("epoch", 0)) + 1
    return history, start_epoch


def snapshot_before_resume(output_dir: Path, resume_path: Path, last_epoch: int) -> Path:
    """Copy checkpoint to output_dir before resume overwrites model.pt."""
    base = output_dir / f"model_snapshot_epoch{int(last_epoch):04d}.pt"
    snap = base
    n = 2
    while snap.is_file():
        snap = output_dir / f"model_snapshot_epoch{int(last_epoch):04d}_{n}.pt"
        n += 1
    shutil.copy2(resume_path, snap)
    return snap


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sequence-dir", type=Path, required=True, help="folder with clean_mosaic.npy + frame_*.npy")
    p.add_argument("--output-dir", type=Path, default=Path("runs/static-seq-train"))
    p.add_argument("--epochs", type=int, default=8000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--nf", type=int, default=64)
    p.add_argument("--device", default=None)
    p.add_argument("--frames", type=int, default=None, help="use only the first N frames (default: all)")
    p.add_argument("--seed", type=int, default=0, help="shuffle seed")
    p.add_argument("--no-shuffle", action="store_true", help="visit frames in sorted order every epoch")
    p.add_argument("--crop-size", type=int, default=None, help="even H=W crop (recommended on Mac for large mosaics)")
    p.add_argument("--crop-y", type=int, default=None)
    p.add_argument("--crop-x", type=int, default=None)
    p.add_argument(
        "--random-crop",
        action="store_true",
        help="new random crop each training step (requires --crop-size)",
    )
    p.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="model.pt to continue; runs --epochs more steps, appends history.json",
    )
    p.add_argument(
        "--no-resume-snapshot",
        action="store_true",
        help="do not copy checkpoint to model_snapshot_epochXXXX.pt before resume",
    )
    p.add_argument(
        "--write-denoised-mosaic",
        action="store_true",
        help="write denoised_mosaic.npy (frame_0000 forward, float16) for debug; default off",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.random_crop and args.crop_size is None:
        raise SystemExit("--random-crop requires --crop-size")
    if args.resume is not None and not args.resume.is_file():
        raise SystemExit(f"--resume not found: {args.resume}")

    resume_ckpt: dict | None = None
    if args.resume is not None:
        resume_ckpt = torch.load(args.resume, map_location="cpu")
        merge_resume_training_args(args, resume_ckpt)

    seq_dir: Path = args.sequence_dir
    clean_path = seq_dir / "clean_mosaic.npy"
    if not clean_path.is_file():
        raise SystemExit(f"missing {clean_path}")
    frame_paths = sorted(seq_dir.glob("frame_*.npy"))
    if args.frames is not None:
        if args.frames < 1:
            raise SystemExit("--frames must be >= 1")
        frame_paths = frame_paths[: args.frames]
    if not frame_paths:
        raise SystemExit(f"no frame_*.npy in {seq_dir}")

    device = torch.device(args.device or default_device_string())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    clean_np = np.load(clean_path).astype(np.float16)
    frames_np = []
    for fp in frame_paths:
        f = np.load(fp).astype(np.float16)
        if f.shape != clean_np.shape:
            raise SystemExit(f"shape mismatch: {fp.name} {f.shape} vs clean {clean_np.shape}")
        frames_np.append(f)

    h, w = clean_np.shape
    if args.crop_size is None and max(h, w) > 1024:
        print(
            "warning: full-frame Bayer U-Net can use many GB RAM; try --crop-size 512 "
            "(or --random-crop with --crop-size for training)."
        )

    full_h, full_w = h, w
    eval_crop_y, eval_crop_x = 0, 0
    if args.crop_size is not None and not args.random_crop:
        cs, eval_crop_y, eval_crop_x = resolve_crop(h, w, args.crop_size, args.crop_y, args.crop_x, None)
        clean_np = crop_mosaic(clean_np, cs, eval_crop_y, eval_crop_x)
        frames_np = [crop_mosaic(f, cs, eval_crop_y, eval_crop_x) for f in frames_np]
        print(f"fixed crop: {cs}x{cs} at (y={eval_crop_y}, x={eval_crop_x})")
    elif args.random_crop:
        _, eval_crop_y, eval_crop_x = resolve_crop(full_h, full_w, args.crop_size, args.crop_y, args.crop_x, None)
        print(f"random crop: {args.crop_size}x{args.crop_size} each step; eval at (y={eval_crop_y}, x={eval_crop_x})")

    clean_p = None
    frames_p = None
    if not args.random_crop:
        clean_p = torch.from_numpy(mosaic_to_packed_np(clean_np)).unsqueeze(0).float().to(device)
        frames_p = [torch.from_numpy(mosaic_to_packed_np(f)).unsqueeze(0).float() for f in frames_np]

    train_shape = (
        (args.crop_size, args.crop_size) if args.random_crop else (clean_np.shape[0], clean_np.shape[1])
    )
    print(f"sequence: {seq_dir}  frames: {len(frames_np)}  train patch: {train_shape}  device: {device}")

    model = build_model("bayer_denoise", nf=args.nf).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    history: list = []
    start_epoch = 1
    if resume_ckpt is not None:
        history, start_epoch = load_history(args.output_dir, resume_ckpt)
        last = int(history[-1]["epoch"] if history else resume_ckpt.get("epoch", 0))
        if not args.no_resume_snapshot:
            snap = snapshot_before_resume(args.output_dir, args.resume, last)
            print(f"snapshot before resume: {snap}")
        model.load_state_dict(resume_ckpt["state_dict"])
        if "optimizer_state_dict" in resume_ckpt:
            opt.load_state_dict(resume_ckpt["optimizer_state_dict"])
        else:
            print("warning: checkpoint has no optimizer_state_dict; optimizer starts fresh")
        print(f"resume: {args.resume}  last epoch {last}  training epochs {start_epoch}..{start_epoch + args.epochs - 1}")

    order = list(range(len(frames_np)))
    for epoch in range(start_epoch, start_epoch + args.epochs):
        if not args.no_shuffle:
            rng.shuffle(order)
        model.train()
        loss_sum = 0.0
        psnr_sum = 0.0
        for i in order:
            if args.random_crop:
                cs, cy, cx = resolve_crop(full_h, full_w, args.crop_size, args.crop_y, args.crop_x, rng)
                noisy_p = (
                    torch.from_numpy(mosaic_to_packed_np(crop_mosaic(frames_np[i], cs, cy, cx)))
                    .unsqueeze(0)
                    .float()
                    .to(device)
                )
                clean_step_p = (
                    torch.from_numpy(mosaic_to_packed_np(crop_mosaic(clean_np, cs, cy, cx)))
                    .unsqueeze(0)
                    .float()
                    .to(device)
                )
            else:
                noisy_p = frames_p[i].float().to(device)
                clean_step_p = clean_p
            opt.zero_grad(set_to_none=True)
            pred = model(noisy_p)
            loss = F.l1_loss(pred, clean_step_p)
            loss.backward()
            opt.step()
            with torch.no_grad():
                mse = F.mse_loss(pred, clean_step_p).item()
            loss_sum += float(loss.item())
            psnr_sum += 99.0 if mse <= 0 else 10.0 * np.log10(1.0 / mse)
        n = len(order)
        avg_loss, avg_psnr = loss_sum / n, psnr_sum / n
        history.append({"epoch": epoch, "loss": avg_loss, "psnr": float(avg_psnr)})
        end_epoch = start_epoch + args.epochs - 1
        if epoch == start_epoch or epoch == end_epoch or epoch % 10 == 0:
            print(f"epoch {epoch}: avg loss={avg_loss:.5f} avg psnr={avg_psnr:.2f} dB")

    last_epoch = int(history[-1]["epoch"]) if history else 0
    weights = model.state_dict()
    ckpt_payload = {
        "model": "bayer_denoise",
        "nf": args.nf,
        "state_dict": weights,
        "model_state": weights,
        "optimizer_state_dict": opt.state_dict(),
        "epoch": last_epoch,
        "noise_std": 0.0,
        "image_size": train_shape[0],
        "crop_size": args.crop_size,
        "crop_y": eval_crop_y if args.crop_size else None,
        "crop_x": eval_crop_x if args.crop_size else None,
        "random_crop": bool(args.random_crop),
    }
    ckpt_epoch = args.output_dir / f"model_epoch{last_epoch:04d}.pt"
    torch.save(ckpt_payload, ckpt_epoch)
    ckpt = args.output_dir / "model.pt"
    shutil.copy2(ckpt_epoch, ckpt)
    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2))

    model.eval()
    noisy_psnrs, den_psnrs = [], []
    first_den = None
    cs = args.crop_size
    with torch.no_grad():
        for idx, fp in enumerate(frame_paths):
            f_np = frames_np[idx]
            c_np = clean_np
            if args.random_crop:
                f_np = crop_mosaic(f_np, cs, eval_crop_y, eval_crop_x)
                c_np = crop_mosaic(clean_np, cs, eval_crop_y, eval_crop_x)
            f_p = torch.from_numpy(mosaic_to_packed_np(f_np)).unsqueeze(0).float().to(device)
            out_m = packed_to_mosaic_np(model(f_p).squeeze(0).cpu().numpy())
            if idx == 0:
                first_den = out_m
            noisy_psnrs.append(psnr(f_np, c_np))
            den_psnrs.append(psnr(out_m, c_np))
            print(f"{fp.name}: noisy {noisy_psnrs[-1]:.2f} dB -> denoised {den_psnrs[-1]:.2f} dB")

    if args.write_denoised_mosaic:
        den_path = args.output_dir / "denoised_mosaic.npy"
        np.save(den_path, first_den.astype(np.float16))

    print("--- mosaic PSNR vs clean (pipeline test) ---")
    print(f"mean noisy:     {np.mean(noisy_psnrs):.2f} dB")
    print(f"mean denoised:  {np.mean(den_psnrs):.2f} dB  ({len(frame_paths)} frames)")
    print(f"checkpoint: {ckpt_epoch}  (latest copy: {ckpt})")
    print(f"history:    {args.output_dir / 'history.json'}")
    if args.write_denoised_mosaic:
        print(f"denoised ({frame_paths[0].name}): {den_path}")


if __name__ == "__main__":
    main()
