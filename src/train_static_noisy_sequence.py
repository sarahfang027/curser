#!/usr/bin/env python3
"""Train BayerDenoiseNet on a static noisy sequence (many noisy frames, one clean mosaic).

Layout (pick one):

1. ``--sequence-dir`` — folder from make_static_noisy_sequence.py:
   clean_mosaic.npy, frame_0000.npy, frame_0001.npy, ... (optional manifest.json)

2. ``--data-root`` — paired dataset on disk (Setup A), no manual assemble step:
   ``<data-root>/GT/*.npy`` (one clean mosaic; first file, or clean_mosaic.npy if present)
   ``<data-root>/input/*.npy`` (all noisy frames, sorted)

3. ``--gt-dir`` + ``--input-dir`` — same as (2) with explicit folders.

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

from bayer_denoiser.crop_pairs import load_manifest, manifest_origins, resolve_origins
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


def ensure_crop_in_bounds(h: int, w: int, crop_size: int, crop_y: int, crop_x: int) -> None:
    if crop_y < 0 or crop_x < 0 or crop_y + crop_size > h or crop_x + crop_size > w:
        hint = ""
        if crop_y >= h or crop_x >= w:
            hint = (
                " Crop origin is outside the loaded mosaic — use full-plate "
                "clean_mosaic.npy / frame_*.npy (e.g. ~17712×18400), not a 512×512 "
                "smoke tile, unless --crop-y/--crop-x fall inside that tile."
            )
        raise SystemExit(f"crop ({crop_y},{crop_x})+{crop_size} outside mosaic {h}x{w}.{hint}")


def validate_manifest_origins(h: int, w: int, manifest_crop_list: list[tuple[int, int, int]]) -> None:
    for y, x, t in manifest_crop_list:
        if t % 2 != 0:
            raise SystemExit(f"manifest tile_size {t} must be even (Bayer mosaic)")
        if t > h or t > w:
            raise SystemExit(f"manifest tile {t} exceeds loaded sequence mosaic {h}x{w}")
        ensure_crop_in_bounds(h, w, t, y, x)


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
    ensure_crop_in_bounds(h, w, crop_size, crop_y, crop_x)
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


def pick_clean_mosaic_path(gt_dir: Path) -> Path:
    named = gt_dir / "clean_mosaic.npy"
    if named.is_file():
        return named
    candidates = sorted(gt_dir.glob("*.npy"))
    if not candidates:
        raise SystemExit(f"no .npy in GT dir {gt_dir}")
    return candidates[0]


def list_noisy_frame_paths(input_dir: Path) -> list[Path]:
    frame_paths = sorted(input_dir.glob("*.npy"))
    if not frame_paths:
        raise SystemExit(f"no .npy in input dir {input_dir}")
    return frame_paths


def resolve_sequence_sources(
    sequence_dir: Path | None,
    data_root: Path | None,
    gt_dir: Path | None,
    input_dir: Path | None,
    frames_limit: int | None,
) -> tuple[Path, list[Path], Path]:
    """Return (clean_path, frame_paths, layout_dir for crops/manifest defaults)."""
    if sequence_dir is not None:
        if data_root is not None or gt_dir is not None or input_dir is not None:
            raise SystemExit("use either --sequence-dir or --data-root / --gt-dir+--input-dir, not both")
        seq_dir = sequence_dir
        clean_path = seq_dir / "clean_mosaic.npy"
        if not clean_path.is_file():
            raise SystemExit(f"missing {clean_path}")
        frame_paths = sorted(seq_dir.glob("frame_*.npy"))
        layout_dir = seq_dir
    else:
        if gt_dir is not None or input_dir is not None:
            if gt_dir is None or input_dir is None:
                raise SystemExit("pass both --gt-dir and --input-dir, or use --data-root")
            gt_path = gt_dir
            input_path = input_dir
            layout_dir = gt_path.parent
        elif data_root is not None:
            gt_path = data_root / "GT"
            input_path = data_root / "input"
            layout_dir = data_root
        else:
            raise SystemExit(
                "pass --sequence-dir, or --data-root (GT/ + input/), or --gt-dir and --input-dir"
            )
        if not gt_path.is_dir():
            raise SystemExit(f"GT dir not found: {gt_path}")
        if not input_path.is_dir():
            raise SystemExit(f"input dir not found: {input_path}")
        clean_path = pick_clean_mosaic_path(gt_path)
        frame_paths = list_noisy_frame_paths(input_path)

    if frames_limit is not None:
        if frames_limit < 1:
            raise SystemExit("--frames must be >= 1")
        frame_paths = frame_paths[:frames_limit]
    if not frame_paths:
        raise SystemExit("no noisy frames to train on")
    return clean_path, frame_paths, layout_dir


def parse_crop_grid(s: str) -> tuple[int, int]:
    parts = s.lower().replace("x", " ").split()
    if len(parts) != 2:
        raise SystemExit("--crop-grid must look like 4x4 or '4 4'")
    return int(parts[0]), int(parts[1])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--sequence-dir",
        type=Path,
        default=None,
        help="folder with clean_mosaic.npy + frame_*.npy (from make_static_noisy_sequence.py)",
    )
    p.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="paired dataset root with GT/ and input/ subfolders (no static_sequence assemble)",
    )
    p.add_argument("--gt-dir", type=Path, default=None, help="clean GT folder (with --input-dir)")
    p.add_argument("--input-dir", type=Path, default=None, help="noisy frames folder (with --gt-dir)")
    p.add_argument("--output-dir", type=Path, default=Path("runs/static-seq-train"))
    p.add_argument("--epochs", type=int, default=100)
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
        "--num-crops",
        type=int,
        default=None,
        help="multi-crop windows on full mosaics (with --crop-size); must match --crop-grid when grid set",
    )
    p.add_argument(
        "--crop-grid",
        type=str,
        default=None,
        help="even grid on full plate, e.g. 3x3 or 4x4 (same idea as pipeline CROP_GRID)",
    )
    p.add_argument(
        "--crop-placement-seed",
        type=int,
        default=42,
        help="RNG for random multi-crop placement when --crop-grid is omitted (default 42)",
    )
    p.add_argument(
        "--crops-manifest",
        type=Path,
        default=None,
        help="train/crops/manifest.json; default: <sequence-dir>/../crops/manifest.json if present",
    )
    p.add_argument(
        "--multi-crop-mode",
        choices=("random", "cycle"),
        default="random",
        help="with a multi-entry crops manifest: pick crop index per step (default random)",
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
    if args.random_crop and args.crops_manifest is not None:
        raise SystemExit("use either --random-crop or --crops-manifest, not both")
    if args.random_crop and (args.num_crops is not None or args.crop_grid is not None):
        raise SystemExit("use either --random-crop or --num-crops/--crop-grid, not both")
    if args.crop_grid is not None and args.num_crops is not None and args.num_crops < 1:
        raise SystemExit("--num-crops must be >= 1")
    if args.resume is not None and not args.resume.is_file():
        raise SystemExit(f"--resume not found: {args.resume}")

    resume_ckpt: dict | None = None
    if args.resume is not None:
        resume_ckpt = torch.load(args.resume, map_location="cpu")
        merge_resume_training_args(args, resume_ckpt)

    clean_path, frame_paths, layout_dir = resolve_sequence_sources(
        args.sequence_dir,
        args.data_root,
        args.gt_dir,
        args.input_dir,
        args.frames,
    )
    seq_dir = layout_dir

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
    print(f"loaded mosaics: {h}x{w}  ({len(frames_np)} frames)")

    cli_grid: tuple[int, int] | None = None
    if args.crop_grid is not None:
        cli_grid = parse_crop_grid(args.crop_grid)
    cli_multi = args.crop_grid is not None or (args.num_crops is not None and args.num_crops > 1)

    manifest_path = args.crops_manifest
    if cli_multi:
        if manifest_path is not None:
            raise SystemExit("pass --num-crops/--crop-grid or --crops-manifest, not both")
        if args.crop_size is None:
            raise SystemExit("--crop-size required with --num-crops/--crop-grid")
        num_crops = args.num_crops if args.num_crops is not None else 1
        if cli_grid is not None and args.num_crops is None:
            num_crops = cli_grid[0] * cli_grid[1]
        try:
            origin_pairs = resolve_origins(
                h, w, args.crop_size, num_crops, cli_grid, args.crop_placement_seed
            )
        except ValueError as e:
            raise SystemExit(str(e)) from e
    else:
        origin_pairs = []

    manifest_data: dict | None = None
    manifest_crop_list: list[tuple[int, int, int]] = []
    use_manifest_crops = False
    cycle_crop_idx = 0

    if cli_multi:
        manifest_crop_list = [(y, x, args.crop_size) for y, x in origin_pairs]
        if len(manifest_crop_list) > 1:
            use_manifest_crops = True
            print(
                f"multi-crop (train-time): {len(manifest_crop_list)} windows  "
                f"tile={args.crop_size}  mode={args.multi_crop_mode}"
            )
        elif len(manifest_crop_list) == 1:
            y0, x0 = origin_pairs[0]
            if args.crop_y is None and args.crop_x is None:
                args.crop_y, args.crop_x = y0, x0
    elif manifest_path is None:
        manifest_candidates: list[Path] = []
        if args.sequence_dir is not None:
            manifest_candidates.append(args.sequence_dir.parent / "crops" / "manifest.json")
        manifest_candidates.append(layout_dir / "crops" / "manifest.json")
        for candidate in manifest_candidates:
            if candidate.is_file():
                manifest_path = candidate
                break

    if not cli_multi and manifest_path is not None:
        if not manifest_path.is_file():
            raise SystemExit(f"--crops-manifest not found: {manifest_path}")
        manifest_data = load_manifest(manifest_path)
        manifest_crop_list = manifest_origins(manifest_data)
        m_tile = int(manifest_data["tile_size"])
        src_shape = tuple(manifest_data.get("source_shape", [h, w]))
        if (h, w) == (m_tile, m_tile) and len(manifest_crop_list) > 1:
            print(
                f"note: sequence is already {m_tile}x{m_tile}; "
                f"ignoring multi-crop manifest ({len(manifest_crop_list)} entries)"
            )
            manifest_crop_list = []
        elif (h, w) != tuple(src_shape) and len(manifest_crop_list) > 0:
            print(
                f"warning: sequence shape {(h, w)} != manifest source {src_shape}; "
                "using manifest origins on loaded mosaics"
            )
        if len(manifest_crop_list) > 1:
            use_manifest_crops = True
            if args.crop_size is None:
                args.crop_size = m_tile
            elif args.crop_size != m_tile:
                raise SystemExit(
                    f"--crop-size {args.crop_size} does not match manifest tile_size {m_tile}"
                )
            print(
                f"multi-crop manifest: {manifest_path}  "
                f"{len(manifest_crop_list)} windows  mode={args.multi_crop_mode}"
            )
        elif len(manifest_crop_list) == 1:
            y0, x0, t0 = manifest_crop_list[0]
            if args.crop_size is None:
                args.crop_size = t0
            if args.crop_y is None and args.crop_x is None:
                args.crop_y, args.crop_x = y0, x0

    if manifest_crop_list:
        validate_manifest_origins(h, w, manifest_crop_list)

    if args.crop_size is None and max(h, w) > 1024:
        print(
            "warning: full-frame Bayer U-Net can use many GB RAM; try --crop-size 512 "
            "(or --random-crop with --crop-size for training)."
        )

    full_h, full_w = h, w
    eval_crop_y, eval_crop_x = 0, 0
    if args.crop_size is not None and not args.random_crop and not use_manifest_crops:
        cs, eval_crop_y, eval_crop_x = resolve_crop(h, w, args.crop_size, args.crop_y, args.crop_x, None)
        clean_np = crop_mosaic(clean_np, cs, eval_crop_y, eval_crop_x)
        frames_np = [crop_mosaic(f, cs, eval_crop_y, eval_crop_x) for f in frames_np]
        if clean_np.shape != (cs, cs):
            raise SystemExit(
                f"cropped mosaic shape {clean_np.shape} != expected ({cs}, {cs}); "
                "check --crop-y/--crop-x against loaded sequence size"
            )
        print(f"fixed crop: {cs}x{cs} at (y={eval_crop_y}, x={eval_crop_x})")
    elif use_manifest_crops:
        eval_crop_y, eval_crop_x, _ = manifest_crop_list[0]
        print(f"manifest crops: {args.crop_size}x{args.crop_size}; eval at (y={eval_crop_y}, x={eval_crop_x})")
    elif args.random_crop:
        _, eval_crop_y, eval_crop_x = resolve_crop(full_h, full_w, args.crop_size, args.crop_y, args.crop_x, None)
        print(f"random crop: {args.crop_size}x{args.crop_size} each step; eval at (y={eval_crop_y}, x={eval_crop_x})")

    clean_p = None
    frames_p = None
    if not args.random_crop and not use_manifest_crops:
        clean_p = torch.from_numpy(mosaic_to_packed_np(clean_np)).unsqueeze(0).float().to(device)
        frames_p = [torch.from_numpy(mosaic_to_packed_np(f)).unsqueeze(0).float() for f in frames_np]

    train_shape = (
        (args.crop_size, args.crop_size)
        if (args.random_crop or use_manifest_crops)
        else (clean_np.shape[0], clean_np.shape[1])
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
            if use_manifest_crops:
                if args.multi_crop_mode == "random":
                    cy, cx, cs = rng.choice(manifest_crop_list)
                else:
                    cy, cx, cs = manifest_crop_list[cycle_crop_idx % len(manifest_crop_list)]
                    cycle_crop_idx += 1
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
            elif args.random_crop:
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
        "crops_manifest": str(manifest_path) if manifest_path else None,
        "multi_crop_mode": args.multi_crop_mode if use_manifest_crops else None,
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
            if args.random_crop or use_manifest_crops:
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
