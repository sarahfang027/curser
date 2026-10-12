# Static noisy sequence (20 frames)

## What it is

A **static sequence** is a short burst where **nothing moves**: every frame is the **same clean image** with **independent random noise** on each frame — like 20 tripod shots in the dark with identical scene, different grain.

Use it to:

- Exercise denoiser **train → checkpoint → infer → PSNR** on many noisy views of one known clean target.
- Later try temporal ideas (frame averaging should cancel noise).
- Stay simple: no motion, no alignment.

**Default run directory:** `runs/static-noisy-sequence-test/` — one-shot via `bash scripts/static_noisy_sequence_pipeline_test.sh`.

## Training epoch defaults

| Entry point | Default epochs | Role |
|-------------|----------------|------|
| `scripts/static_noisy_sequence_pipeline_test.sh` | **8000** (`EPOCHS` env) | Full static-sequence pipeline |
| `scripts/static_noisy_singleframe_pipeline_test.sh` | **100** (`EPOCHS` env) | Quick single-frame smoke — [single-frame guide](static-noisy-singleframe-pipeline-test.md) |
| `python -m bayer_denoiser.train_static_noisy_sequence` | **100** (`--epochs`) | Direct CLI default; pipeline scripts pass their own `--epochs` |

Quick plumbing on the **sequence** pipeline: `EPOCHS=100 bash scripts/static_noisy_sequence_pipeline_test.sh`.

## Step-by-step

All paths below use **`runs/static-noisy-sequence-test/`** unless noted.

### 1. Build the sequence (or let the pipeline do it)

**Recommended:** skip to step 2. Manual build:

```bash
PIPELINE_RUN=runs/static-noisy-sequence-test bash scripts/make_static_noisy_sequence.sh
```

Writes `train/static_sequence/` with `clean_mosaic.npy`, `frame_0000.npy` … `frame_0019.npy`, optional `manifest.json` — float16 RGGB [0, 1]; ~**34 dB** noisy PSNR at `--noise-std 0.02`. No clean crop? Use `--input` / `--sigma` on `make_static_noisy_sequence.py` ([options](#make-static-sequence-options)).

### 2. Pipeline (smoke → multi-crop goal)

One script builds data (if needed), trains, full-plate infer, and writes `compare/`. **Goal:** multi-crop on the **full-sample** static sequence — several **512×512** train windows on the plate (`train/crops/` + `manifest.json`), then **8000** epochs and full-plate denoised output. Details: [Training progression](#training-progression-smoke--multi-crop).

**Input plate:** default `data/samples/A001C006_0101_4E4951.npy` under the repo root. The script loads that full Bayer mosaic `.npy`, builds Gaussian GT + `train/static_sequence/`, trains on the frames, and runs full-plate infer on the same file. Use another plate as the **first argument**:

```bash
bash scripts/static_noisy_sequence_pipeline_test.sh /path/to/your_mosaic.npy
```

Env vars (`NUM_CROPS`, `EPOCHS`, …) work the same; put them before `bash` if you combine them.

**Smoke** (fast `EPOCHS=100`, noisy/GT crop PNGs in `train/` via auto `SMOKE_CROP_PNG=1`):

```bash
# Center 512×512 window (default smoke)
CROP_SIZE=512 EPOCHS=100 bash scripts/static_noisy_sequence_pipeline_test.sh

# Fixed window on full plate — even top-left (y, x); NUM_CROPS=1; do not set CROP_GRID
CROP_Y=4000 CROP_X=6000 TILE=512 EPOCHS=100 NUM_CROPS=1 SMOKE_CROP_PNG=1 \
  bash scripts/static_noisy_sequence_pipeline_test.sh

# Paired synthetic on disk (one clean in GT/, noisy frames in input/) — no static_sequence/ copy step
DATA_ROOT=data/sythatic_sequence RUN=runs/synthetic-sequence \
  CROP_SIZE=512 CROP_Y=4000 CROP_X=6000 EPOCHS=100 NUM_CROPS=1 SMOKE_CROP_PNG=1 \
  bash scripts/static_noisy_sequence_pipeline_test.sh
```

| Stage | Command |
|-------|---------|
| **Multi-crop proof** | `NUM_CROPS=9 CROP_GRID=3x3 TILE=512 EPOCHS=100 bash scripts/static_noisy_sequence_pipeline_test.sh` |
| **Long run (goal)** | `NUM_CROPS=16 CROP_GRID=4x4 TILE=512 bash scripts/static_noisy_sequence_pipeline_test.sh` |

With no env vars, defaults are **8000** epochs, **`NUM_CROPS=1`** (center crop only), **`FULL_INFER=1`** (tiled infer on the sample plate). Each training step uses **one** manifest window when multi-crop is enabled.

**Compare outputs:**

| File | Role |
|------|------|
| `denoised_infer.npy` | Full-plate denoised mosaic |
| `denoised.png` | Preview of the full result |
| `denoised_patch.png` (smoke) or `denoised_infer_patch.npy` + `denoised_patch.png` | **512×512** patch QA at the training window (center, or `CROP_Y`/`CROP_X`) |
| `train/noisy_input_crop.png`, `train/gaussian_GT_crop.png` | Smoke (`EPOCHS=100`, `NUM_CROPS=1`): auto `SMOKE_CROP_PNG=1` (replaces crop `.npy`); center, **`CROP_Y`/`CROP_X`**, or `CROP_GRID=1x1` |
| `infer_input_crop.png`, `denoised_infer_patch.png`, … | Smoke patch QA under **`compare/`** |
| `noisy.png`, `clean_gaussian_gt.png` | Previews |
| `training_curves.png` | When `PLOT_HISTORY=1` (default) |

Checkpoint: `model.pt` at run root. Faster smoke infer: `FULL_INFER=0` ([step 3](#3-patch-only-infer)).

<details>
<summary><strong>Manual train / resume</strong></summary>

Same training as the pipeline; use when debugging or resuming without re-running infer/plots.

```bash
python -m bayer_denoiser.train_static_noisy_sequence \
  --sequence-dir runs/static-noisy-sequence-test/train/static_sequence \
  --output-dir runs/static-noisy-sequence-test \
  --crop-size 512 \
  --device cpu
```

Loads `clean_mosaic.npy` once, then each epoch visits every `frame_*.npy` (shuffled) with L1 vs the same packed RGGB clean. Use **`--frames 1`** for only `frame_0000`. On Mac, use an even **`--crop-size`** (512 or 256) to avoid OOM.

**Resume** (`--epochs` = *additional* epochs):

```bash
python -m bayer_denoiser.train_static_noisy_sequence \
  --sequence-dir runs/static-noisy-sequence-test/train/static_sequence \
  --output-dir runs/static-noisy-sequence-test \
  --resume runs/static-noisy-sequence-test/model_epoch0100.pt \
  --epochs 100 \
  --device cpu
```

Crop / `nf` follow the checkpoint; `history.json` appends. Resume snapshots `model_snapshot_epochXXXX.pt` unless `--no-resume-snapshot`. Checkpoints and `history.json` sit at **run root** (see [layout](#run-folder-layout)). Flags: [CLI options](#train_static_noisy_sequence-cli-options) in Reference.

</details>

### 3. Patch-only infer

Default pipeline already runs **full-plate** tiled infer (`FULL_INFER=1`). For a **fast** run (patch infer only, smaller `denoised_infer.npy`):

```bash
FULL_INFER=0 bash scripts/static_noisy_sequence_pipeline_test.sh
```

**Manual** full-plate infer after training:

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

### 4. Success criteria

- Training **PSNR** clearly above noisy PSNR on the train patch.
- Patch infer matches training forward: **`compare/denoised_infer_patch.png`** on smoke (`SMOKE_CROP_PNG=1`), else **`compare/denoised_infer_patch.npy`** (or patch-sized `denoised_infer.npy` when `FULL_INFER=0`).
- After step 2, open **`runs/static-noisy-sequence-test/compare/`** for PNGs and curves.

Optional debug: `--write-denoised-mosaic` or pipeline **`DEBUG_DENOISED_MOSAIC=1`** → `compare/denoised_mosaic.npy` (not canonical).

High PSNR vs Gaussian GT validates **pipeline wiring**, not real-world denoising quality.

## Real paired plates (Setup A)

Use this when you already have **aligned** noisy and clean Bayer mosaics (float16 H×W `.npy`, same shape, RGGB packing per [float16 pipeline](float16-pipeline-standard.md)). Point the pipeline at a dataset root with **`GT/`** and **`input/`**, or call the trainer with **`--data-root`** (see [testing synthetic data](testing-synthetic-data.md#b--paired-synthetic-clean--noisy-you-generated)).

**Not covered here:** `.sky` ingest (see [sky-format-conversion.md](sky-format-conversion.md)); the synthetic pipeline (`static_noisy_sequence_pipeline_test.sh`), which builds Gaussian GT from a single plate.

Example run root: `runs/my-real-paired/`.

### 1. On-disk layout (full plates)

**Preferred:** dataset root with paired folders (no manual `static_sequence/` copy):

```text
data/my_clip/
  GT/          # one clean .npy (or clean_mosaic.npy); shared by all frames
  input/       # every noisy frame .npy (sorted lexicographically)
```

Smoke:

```bash
DATA_ROOT=data/my_clip RUN=runs/my-real-paired \
  CROP_SIZE=512 CROP_Y=4000 CROP_X=6000 EPOCHS=100 \
  bash scripts/static_noisy_sequence_pipeline_test.sh
```

**Alternate:** legacy folder from `make_static_noisy_sequence.py` or a manual copy:

```text
runs/my-real-paired/train/static_sequence/
  clean_mosaic.npy
  frame_0000.npy
```

Do **not** use `make_static_noisy_sequence.py` on real clean plates (it adds synthetic noise). One noisy frame is enough; use **`--frames 1`** when training if you only have one file under `input/`.

Keep **full H×W** mosaics (same shape for clean and every noisy frame). You do **not** need `train/crops/*.npy` on disk unless you pre-built a manifest.

### 2. Train (multi-crop at train time)

Each step crops a **512×512** window from the loaded full mosaics (noisy frame + clean). Pick windows like the synthetic pipeline: **`--num-crops`** + **`--crop-grid`** (or random placement with **`--num-crops`** + **`--crop-placement-seed`**). See [Training progression](#training-progression-smoke--multi-crop) for 3×3 proof vs 4×4 long run.

```bash
# Multi-crop proof (3×3)
python -m bayer_denoiser.train_static_noisy_sequence \
  --sequence-dir runs/my-real-paired/train/static_sequence \
  --output-dir runs/my-real-paired \
  --crop-size 512 \
  --num-crops 9 \
  --crop-grid 3x3 \
  --epochs 100 \
  --device mps

# Long run (4×4)
python -m bayer_denoiser.train_static_noisy_sequence \
  --sequence-dir runs/my-real-paired/train/static_sequence \
  --output-dir runs/my-real-paired \
  --crop-size 512 \
  --num-crops 16 \
  --crop-grid 4x4 \
  --epochs 8000 \
  --device mps
```

`--num-crops` must match the grid when `--crop-grid` is set (`3x3→9`, `4x4→16`). Checkpoints and `history.json` at **run root** — same layout as [Run folder layout](#run-folder-layout).

Optional: pre-built `train/crops/manifest.json` from `generate_training_crop_pairs.py` (still crops from full mosaics at train time; disk tiles are not required for training).

### 3. Infer and previews

No pipeline wrapper — run infer on your noisy plate (tiled if full resolution):

```bash
mkdir -p runs/my-real-paired/compare
python -m bayer_denoiser.infer \
  --checkpoint runs/my-real-paired/model.pt \
  --input /path/to/noisy_full.npy \
  --output runs/my-real-paired/compare/denoised_infer.npy \
  --tiled --tile-size 512 --overlap 64

python -m bayer_denoiser.preview \
  runs/my-real-paired/compare/denoised_infer.npy \
  --write-mosaic runs/my-real-paired/compare/denoised.png
```

PSNR vs real clean is meaningful here; vs Gaussian GT it was only a plumbing check.

## Training progression (smoke → multi-crop)

Typical order: **smoke/debug done** → **first good multi-crop** on the **full sample** static sequence → longer `EPOCHS`. Every pipeline run ends with **full-plate** tiled infer by default (`FULL_INFER=1`).

| Stage | Goal | Settings |
|-------|------|----------|
| **Smoke** | Train → checkpoint → infer → PSNR wiring | **`EPOCHS=100`**, **`NUM_CROPS=1`**. Center: `CROP_SIZE=512`. Off-center on full plate: **`CROP_Y` / `CROP_X`** (even). Single-frame: [single-frame guide](static-noisy-singleframe-pipeline-test.md). Commands in [step 2](#2-pipeline-smoke--multi-crop-goal). |
| **First good multi-crop** | Prove 9 regions + manifest + train on full-plate sequence | **`NUM_CROPS=9` `CROP_GRID=3x3` `TILE=512` `EPOCHS=100`** — one short run before the long job |
| **Long run (recommended)** | **8000** epochs, even spread, reasonable Mac cost | **`NUM_CROPS=16` `CROP_GRID=4x4` `TILE=512`** (omit `EPOCHS` → pipeline default **8000**) |
| **Long run (lighter)** | Less disk / slightly less spatial spread | **`NUM_CROPS=9` `CROP_GRID=3x3` `TILE=512`** for all **8000** epochs |
| **Long run (denser grid)** | More regions; heavier build (16 pairs × float16 on disk + full-plate sequence) | **`NUM_CROPS=25` `CROP_GRID=5x5` `TILE=512`** |

**Long-run crop rules (read before an 8000-epoch job)**

1. **`NUM_CROPS` must match the grid** when you set **`CROP_GRID`**: `3x3→9`, `4x4→16`, `5x5→25`. A mismatch fails the script.
2. **Prefer `CROP_GRID` for long runs** — reproducible, evenly spread windows. **`CROP_SEED` does nothing for grid placement** (ignore it when `CROP_GRID` is set).
3. **Use `CROP_SEED` only for random placement** (`NUM_CROPS=N` and **no** `CROP_GRID`). Pick **one** seed and keep it for the whole project line (e.g. **`CROP_SEED=42`**). Changing seed after a long train means **new** `train/crops/` — treat as a new experiment, not resume.
4. **`--seed` on the trainer** (default **0**) shuffles **frames** each epoch and (with manifest) picks **which of the N windows** per step. It does **not** move the N windows on disk. You usually **leave train `--seed` at 0** unless you want a different frame order.
5. **Do not change `TILE` / grid / `NUM_CROPS` mid-run** on the same `--output-dir` without rebuilding `train/crops/` and usually retraining from scratch. Checkpoints encode **`crop_size`** from the manifest tile.

**What to use when (full sample ~17712×18400, `TILE=512`)**

| Your priority | `NUM_CROPS` | `CROP_GRID` | `CROP_SEED` | Notes |
|---------------|-------------|-------------|-------------|--------|
| **Default long run** | **16** | **4x4** | *(omit)* | Best balance: 16 regions corner-to-corner, not 25× export weight |
| First multi-crop proof only | 9 | 3x3 | *(omit)* | Use **`EPOCHS=100`** first; then rerun with 4×4 for 8000 |
| Minimize build size | 9 | 3x3 | *(omit)* | OK for 8000 if 4×4 OOMs during **data** build |
| Max grid spread (still practical) | 25 | 5x5 | *(omit)* | Heavier; use if 4×4 trains well and you want finer grid |
| Random windows (advanced) | **16** or **25** | *(omit)* | **42** (fixed) | Overlap possible; less even than 4×4 — only if you deliberately want random layout |

**Seeds and crop placement (short)**

| Knob | Controls |
|------|----------|
| **`CROP_GRID`** | **Where** the N tiles sit (deterministic). **`NUM_CROPS`** = rows×cols. |
| **`CROP_Y`, `CROP_X`** | **One** fixed window (even top-left) on the full plate; **`NUM_CROPS=1`**; not with **`CROP_GRID`**. |
| **`CROP_SEED`** | **Where** the N tiles sit when **no** grid (random origins once at build). Not a crop count. |
| **`train --seed`** | **Which frame** and **which manifest index** each step (random mode); not build placement. |

Multi-crop training **spreads optimization** across the plate; it is **not** full pixel coverage. **Whole-frame** denoised output comes from infer tiling (pipeline default), not from training on hundreds of crops.

On a **~17712×18400** plate, one **512×512** tile is a tiny fraction of pixels (~**0.08%** of the mosaic). Do **not** aim for **~1190** non-overlapping 512 tiles in **training** (that number is “tile the whole frame once”; inference tiling is separate).

**Mac** (`~/Projects/bayer_denoiser`, venv active):

```bash
cd ~/Projects/bayer_denoiser
source .venv/bin/activate

# Smoke (sequence) — center crop, fast epochs
CROP_SIZE=512 EPOCHS=100 bash scripts/static_noisy_sequence_pipeline_test.sh

# Smoke — fixed window on full plate (example coordinates)
CROP_Y=4000 CROP_X=6000 TILE=512 EPOCHS=100 NUM_CROPS=1 SMOKE_CROP_PNG=1 \
  bash scripts/static_noisy_sequence_pipeline_test.sh

# Step A — first multi-crop proof (short)
NUM_CROPS=9 CROP_GRID=3x3 TILE=512 EPOCHS=100 bash scripts/static_noisy_sequence_pipeline_test.sh

# Step B — long run (recommended): 4×4, 8000 epochs (default EPOCHS)
NUM_CROPS=16 CROP_GRID=4x4 TILE=512 bash scripts/static_noisy_sequence_pipeline_test.sh

# Step B alt — lighter long run (3×3, 8000 epochs)
NUM_CROPS=9 CROP_GRID=3x3 TILE=512 bash scripts/static_noisy_sequence_pipeline_test.sh

# Step B alt — denser long run (5×5, 8000 epochs)
NUM_CROPS=25 CROP_GRID=5x5 TILE=512 bash scripts/static_noisy_sequence_pipeline_test.sh

# Step B alt — random 16 windows (no grid); fix layout with CROP_SEED only
NUM_CROPS=16 CROP_SEED=42 TILE=512 bash scripts/static_noisy_sequence_pipeline_test.sh

# Long run (full-plate infer is default; omit FULL_INFER)
NUM_CROPS=16 CROP_GRID=4x4 TILE=512 bash scripts/static_noisy_sequence_pipeline_test.sh
```

## Run folder layout

```text
runs/static-noisy-sequence-test/
  train/
    noisy_input_crop.png          # smoke QA only (SMOKE_CROP_PNG=1); no gaussian_GT_crop.npy
    gaussian_GT_crop.png
    gaussian_GT_full.npy          # not kept (clean lives in static_sequence/clean_mosaic.npy)
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
    denoised_infer_patch.npy      # patch QA (default full infer)
    denoised_patch.png
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

### Make static sequence options

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

### Patch infer and verify

When the pipeline already ran, skip this. Default output is a **512×512** patch — `compare/denoised_infer.npy` should match the training forward on the same patch.

```bash
python -m bayer_denoiser.infer \
  --checkpoint runs/static-noisy-sequence-test/model.pt \
  --input runs/static-noisy-sequence-test/compare/infer_input_crop.npy \
  --output runs/static-noisy-sequence-test/compare/denoised_infer.npy
```

(`--input` can be `train/static_sequence/frame_0000.npy` instead.) Shared clean target → PSNR checks **plumbing**, not generalization.

### Training curves

`history.json` rows: **`epoch`**, **`loss`**, **`psnr`** (older files may use `avg_loss` / `avg_psnr`).

```bash
python -m bayer_denoiser.plot_training_history \
  --history runs/static-noisy-sequence-test/history.json \
  --output runs/static-noisy-sequence-test/compare/training_curves.png
```

The pipeline runs this when **`PLOT_HISTORY=1`** (default; `0` to skip). Optional **`--title`**.

### `train_static_noisy_sequence` CLI options

| Option | Default | Meaning |
|--------|---------|---------|
| `--sequence-dir` | — | `clean_mosaic.npy` + `frame_*.npy` (from `make_static_noisy_sequence.py`) |
| `--data-root` | — | `GT/` + `input/` subfolders (Setup A on disk) |
| `--gt-dir`, `--input-dir` | — | Explicit clean / noisy folders (instead of `--data-root`) |
| `--output-dir` | `runs/static-seq-train` | Run root for checkpoints and history |
| `--epochs` | 100 | Passes over all frames (pipeline uses **8000** unless `EPOCHS` set) |
| `--lr` | 1e-3 | Adam learning rate |
| `--nf` | 64 | Network width |
| `--device` | auto | e.g. `mps`, `cpu` |
| `--frames` | all | First N frames only (sorted) |
| `--seed`, `--no-shuffle` | 0 / off | Frame order |
| `--crop-size`, `--crop-y`, `--crop-x` | off | Even square crop (center if origin omitted). Origins must lie inside **loaded** `clean_mosaic.npy` H×W (full plate for plate coordinates; smoke runs use ~512×512 tiles — use center crop or rebuild sequence from full-frame GT). Trainer errors if origin+crop exceeds mosaic bounds. |
| `--num-crops`, `--crop-grid` | off | Multi-crop windows on **full** mosaics (train-time crop; like pipeline `NUM_CROPS` / `CROP_GRID`) |
| `--crop-placement-seed` | 42 | Random window layout when `--crop-grid` omitted |
| `--crops-manifest` | auto | `train/crops/manifest.json` when present (do not combine with `--num-crops` / `--crop-grid`) |
| `--multi-crop-mode` | `random` | `random` or `cycle` when manifest has 2+ entries |
| `--resume` | off | Continue from checkpoint; `--epochs` = more epochs |
| `--no-resume-snapshot` | off | Skip pre-resume snapshot copy |
| `--write-denoised-mosaic` | off | Debug mosaic from frame_0000 forward pass |
