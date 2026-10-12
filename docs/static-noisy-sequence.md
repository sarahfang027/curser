# Static noisy sequence (20 frames)

## What it is

A **static sequence** is a short burst where **nothing moves**: every frame is the **same clean image** with **independent random noise** on each frame — like 20 tripod shots in the dark with identical scene, different grain.

Use it to:

- Exercise denoiser **train → checkpoint → infer → PSNR** on many noisy views of one known clean target.
- Later try temporal ideas (frame averaging should cancel noise).
- Stay simple: no motion, no alignment.

The clean target is **Gaussian GT** (blurred synthetic “clean”, not real ground truth). For a **1-frame** smoke first, see [single-frame pipeline test](static-noisy-singleframe-pipeline-test.md).

**Canonical run:** `runs/static-noisy-sequence-test/` — one-shot via `bash scripts/static_noisy_sequence_pipeline_test.sh`.

## Prerequisites

From the Mac repo (`~/Projects/bayer_denoiser`):

```bash
cd ~/Projects/bayer_denoiser
source .venv/bin/activate
bash scripts/sync_context_to_repo.sh   # if scripts/docs lag Context
```

You need a Gaussian GT crop (usually already built by the single-frame pipeline):

- `runs/static-noisy-singleframe-test/train/gaussian_GT_crop.npy`

### Training epoch defaults

| Entry point | Default epochs | Role |
|-------------|----------------|------|
| `scripts/static_noisy_sequence_pipeline_test.sh` | **8000** (`EPOCHS` env) | Full static-sequence pipeline |
| `scripts/static_noisy_singleframe_pipeline_test.sh` | **100** (`EPOCHS` env) | Quick single-frame smoke — [single-frame guide](static-noisy-singleframe-pipeline-test.md) |
| `python -m bayer_denoiser.train_static_noisy_sequence` | **100** (`--epochs`) | Direct CLI default; pipeline scripts pass their own `--epochs` |

Quick plumbing on the **sequence** pipeline: `EPOCHS=100 bash scripts/static_noisy_sequence_pipeline_test.sh`.

**RGGB packing** (`src/sort_bayer.py`): R, G1 `[1::2,0::2]`, G2 `[0::2,1::2]`, B — shared by train/infer/preview via `mosaic_to_packed_np` / `packed_to_mosaic_np`. Older checkpoints from before this fix swapped G1/G2; **retrain** for correct infer.

## Step-by-step

All paths below use **`runs/static-noisy-sequence-test/`** unless noted.

### 1. Build the sequence (or let the pipeline do it)

**Recommended:** skip to step 2. Manual build:

```bash
PIPELINE_RUN=runs/static-noisy-sequence-test bash scripts/make_static_noisy_sequence.sh
```

Writes `train/static_sequence/` with `clean_mosaic.npy`, `frame_0000.npy` … `frame_0019.npy`, optional `manifest.json` — float16 RGGB [0, 1]; ~**34 dB** noisy PSNR at `--noise-std 0.02`. No clean crop? Use `--input` / `--sigma` on `make_static_noisy_sequence.py` ([options](#make_static_noisy_sequencepy-options)).

### 2. One-shot pipeline (smoke or full train)

```bash
bash scripts/static_noisy_sequence_pipeline_test.sh
```

Defaults: **8000** epochs, center **512×512** train crop, checkpoints at run root, compare artifacts under `compare/`.

Fast plumbing check:

```bash
CROP_SIZE=512 EPOCHS=100 bash scripts/static_noisy_sequence_pipeline_test.sh
```

**Start with a fixed center crop** (default) — many steps on one patch. Do **not** use `--random-crop` for first plumbing; that gives one step per random window.

Expected under `compare/`: `denoised_infer.npy`, `noisy.png`, `clean_gaussian_gt.png`, `denoised.png`, and `training_curves.png` when `PLOT_HISTORY=1` (default). Checkpoint: `model.pt` at run root. With `CROP_SIZE` set, the script also writes `compare/infer_input_crop.npy` for patch infer.

### 3. Optional: multi-crop training

Cover more of the plate by setting **`NUM_CROPS`** or **`CROP_GRID`** on the pipeline script. It writes aligned float16 pairs under `train/crops/` plus `train/crops/manifest.json`. Legacy `noisy_input_crop.npy` / `gaussian_GT_crop.npy` remain crop **0**. When the static sequence is **full-plate** and shapes match the manifest, training auto-loads the manifest and picks a crop index each step (`--multi-crop-mode random`, default).

```bash
NUM_CROPS=4 CROP_GRID=2x2 TILE=512 bash scripts/static_noisy_sequence_pipeline_test.sh
NUM_CROPS=9 CROP_GRID=3x3 TILE=512 bash scripts/static_noisy_sequence_pipeline_test.sh
NUM_CROPS=4 CROP_SEED=7 TILE=512 bash scripts/static_noisy_sequence_pipeline_test.sh   # random windows
```

Or `python scripts/generate_training_crop_pairs.py` with `--train-dir runs/static-noisy-sequence-test/train`. Default `NUM_CROPS=1` = center crop only.

### 4. Train manually (same semantics as the pipeline)

```bash
python -m bayer_denoiser.train_static_noisy_sequence \
  --sequence-dir runs/static-noisy-sequence-test/train/static_sequence \
  --output-dir runs/static-noisy-sequence-test \
  --crop-size 512 \
  --device cpu
```

`train_static_noisy_sequence` loads `clean_mosaic.npy` once, then each epoch visits every `frame_*.npy` (shuffled) with L1 vs the same packed RGGB clean. Use **`--frames 1`** for only `frame_0000`.

On Mac, full-frame U-Nets can OOM — use an even `--crop-size` (512 or 256). **`--random-crop`** with `--crop-size` = new random crop each step (multi-patch training, not first plumbing).

**Resume** (same `--sequence-dir` and `--output-dir`; `--epochs` = *additional* epochs):

```bash
python -m bayer_denoiser.train_static_noisy_sequence \
  --sequence-dir runs/static-noisy-sequence-test/train/static_sequence \
  --output-dir runs/static-noisy-sequence-test \
  --resume runs/static-noisy-sequence-test/model_epoch0100.pt \
  --epochs 100 \
  --device cpu
```

Crop / `nf` follow the checkpoint; `history.json` appends. Resume snapshots `model_snapshot_epochXXXX.pt` unless `--no-resume-snapshot`. Checkpoints and `history.json` sit at **run root** (see [layout](#run-folder-layout)).

### 5. Patch infer and verify

Default pipeline output is a **512×512** patch — `compare/denoised_infer.npy` should match the training forward on the same patch.

```bash
python -m bayer_denoiser.infer \
  --checkpoint runs/static-noisy-sequence-test/model.pt \
  --input runs/static-noisy-sequence-test/compare/infer_input_crop.npy \
  --output runs/static-noisy-sequence-test/compare/denoised_infer.npy
```

(`--input` can be `train/static_sequence/frame_0000.npy` instead.) Shared clean target → PSNR checks **plumbing**, not generalization.

### 6. Full-plate tiled infer (optional)

For a **full-resolution** mosaic and PNG on the sample plate:

```bash
FULL_INFER=1 bash scripts/static_noisy_sequence_pipeline_test.sh
# or: INFER_TILED=1 INFER_TILE_SIZE=512 INFER_OVERLAP=64 ...
```

Runs `bayer_denoiser.infer --tiled` on the full `SAMPLE` `.npy`, writes full H×W `compare/denoised_infer.npy`, and `bayer_denoiser.preview` → `compare/denoised.png`. Also writes patch-sized `compare/denoised_infer_patch.npy` + `denoised_patch.png` for quick QA.

Manual after training:

```bash
python -m bayer_denoiser.infer \
  --checkpoint runs/static-noisy-sequence-test/model.pt \
  --input data/samples/A001C006_0101_4E4951.npy \
  --output runs/static-noisy-sequence-test/compare/denoised_infer.npy \
  --tiled --tile-size 512 --overlap 64

python -m bayer_denoiser.preview \
  runs/static-noisy-sequence-test/compare/denoised_infer.npy \
  --write-mosaic runs/static-noisy-sequence-test/compare/denoised.png
```

Tiling **auto-enables** when input mosaic H×W exceeds `--tile-size` (even without `--tiled`). Tile size defaults to checkpoint `crop_size` if present, else **512**. Overlap defaults to **64** (linear blend); use `--overlap 0` for abutting tiles.

### 7. Training curves

`history.json` is a JSON array: each row has **`epoch`**, **`loss`**, **`psnr`** (older files may use `avg_loss` / `avg_psnr` — plotter accepts both).

```bash
python -m bayer_denoiser.plot_training_history \
  --history runs/static-noisy-sequence-test/history.json \
  --output runs/static-noisy-sequence-test/compare/training_curves.png
```

Pipeline runs this when **`PLOT_HISTORY=1`** (default; `0` to skip). Flags: `--history`, `--output` (default `<history-dir>/training_curves.png`), optional `--title`.

### 8. Success criteria

- Training **PSNR** clearly above noisy PSNR on the train patch.
- **`compare/denoised_infer.npy`** matches `bayer_denoiser.infer` on the same patch as training.
- After step 2, open **`runs/static-noisy-sequence-test/compare/`** for PNGs and curves.

Optional debug: `--write-denoised-mosaic` or pipeline **`DEBUG_DENOISED_MOSAIC=1`** → `compare/denoised_mosaic.npy` (not canonical).

High PSNR vs Gaussian GT validates **pipeline wiring**, not real-world denoising quality.

## Run folder layout

```text
runs/static-noisy-sequence-test/
  train/
    noisy_input_crop.npy          # crop 0 (legacy infer smoke)
    gaussian_GT_crop.npy
    gaussian_GT_full.npy          # multi-crop only
    crops/
      manifest.json
      noisy_000.npy … noisy_NNN.npy
      gaussian_GT_000.npy … gaussian_GT_NNN.npy
    static_sequence/
      clean_mosaic.npy
      frame_0000.npy … frame_0019.npy
      manifest.json
  compare/
    infer_input_crop.npy          # when CROP_SIZE set
    denoised_infer.npy            # canonical (bayer_denoiser.infer)
    denoised_infer_patch.npy      # FULL_INFER QA
    noisy.png
    clean_gaussian_gt.png
    denoised.png
    training_curves.png           # PLOT_HISTORY=1
  model.pt
  model_epoch0100.pt
  history.json
```

Same run-root pattern as [single-frame pipeline test](static-noisy-singleframe-pipeline-test.md).

## Reference

### `make_static_noisy_sequence.py` options

| Option | Default | Meaning |
|--------|---------|---------|
| `--clean` | — | Clean H×W mosaic `.npy` (use this **or** `--input`) |
| `--input`, `--sigma` | — / 1.0 | Noisy mosaic + blur strength to build Gaussian GT first |
| `--tile-size` | 0 | With `--input`: center-crop size (0 = full frame) |
| `--output-dir` | required | Where frames are written |
| `--frames` | 20 | Number of frames |
| `--noise-std` | 0.02 | Noise on [0, 1] pixels (bigger = grainier) |
| `--seed` | 0 | Reproducible frames |
| `--write-manifest` | off | Write `manifest.json` with files and settings |

`bash scripts/make_static_noisy_sequence.sh` without `PIPELINE_RUN` still defaults to legacy **`runs/static_sequence/`** — prefer `PIPELINE_RUN=runs/static-noisy-sequence-test` or the pipeline script.

### `train_static_noisy_sequence` CLI options

| Option | Default | Meaning |
|--------|---------|---------|
| `--sequence-dir` | required | `clean_mosaic.npy` + `frame_*.npy` |
| `--output-dir` | `runs/static-seq-train` | Run root for checkpoints and history |
| `--epochs` | 100 | Passes over all frames (pipeline uses **8000** unless `EPOCHS` set) |
| `--lr` | 1e-3 | Adam learning rate |
| `--nf` | 64 | Network width |
| `--device` | auto | e.g. `mps`, `cpu` |
| `--frames` | all | First N frames only (sorted) |
| `--seed`, `--no-shuffle` | 0 / off | Frame order |
| `--crop-size`, `--crop-y`, `--crop-x` | off | Even square crop (center if origin omitted) |
| `--random-crop` | off | New random crop each step (needs `--crop-size`) |
| `--crops-manifest` | auto | `train/crops/manifest.json` when present |
| `--multi-crop-mode` | `random` | `random` or `cycle` when manifest has 2+ entries |
| `--resume` | off | Continue from checkpoint; `--epochs` = more epochs |
| `--no-resume-snapshot` | off | Skip pre-resume snapshot copy |
| `--write-denoised-mosaic` | off | Debug mosaic from frame_0000 forward pass |

## Legacy and advanced

<details>
<summary><strong>Legacy paths</strong></summary>

Older workflows used top-level **`runs/static_sequence/`** (default for `make_static_noisy_sequence.sh` without `PIPELINE_RUN`) and ad-hoc `--output-dir` values.

```text
runs/static_sequence/          # sequence only
  clean_mosaic.npy
  frame_0000.npy … frame_0019.npy
  manifest.json                # with --write-manifest
```

Example train + legacy helper infer:

```bash
python -m bayer_denoiser.train_static_noisy_sequence \
  --sequence-dir runs/static_sequence \
  --output-dir runs/static-seq-plumbing \
  --crop-size 512 --device cpu

python scripts/input_and_infer_crop.py \
  --input runs/static_sequence/frame_0000.npy \
  --checkpoint runs/static-seq-plumbing/model.pt
```

Prefer **`denoised_infer.npy`** via `bayer_denoiser.infer` over `input_and_infer_crop.py`.

</details>

<details>
<summary><strong>Advanced training patterns</strong></summary>

- **`--tile-size`** on `make_static_noisy_sequence.py` — smaller mosaics on disk when using `--input`.
- **`--frames 1 --random-crop`** — one sequence frame, new crop each step (not default 20-frame training).
- **`--device mps`** — same CLI; use `--crop-size 512` on Mac to avoid RAM/MPS limits.
- Mosaic shape: `python -c "import numpy as np; print(np.load('runs/static-noisy-sequence-test/train/static_sequence/clean_mosaic.npy').shape)"`

</details>
