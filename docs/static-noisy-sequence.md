# Static noisy sequence (20 frames)

## What it is

A **static sequence** is a short burst where **nothing moves**: every frame is the **same clean image** with **independent random noise** on each frame — like 20 tripod shots in the dark with identical scene, different grain.

Use it to:

- Exercise denoiser **train → checkpoint → infer → PSNR** on many noisy views of one known clean target.
- Later try temporal ideas (frame averaging should cancel noise).
- Stay simple: no motion, no alignment.

**Default run directory:** `runs/static-noisy-sequence-test/` — Gaussian synthetic path via `bash scripts/static_noisy_sequence_pipeline_test.sh`.

**Two ways to feed training:**

| Mode | On-disk input | Pipeline |
|------|----------------|----------|
| **Gaussian synthetic** | One sample plate `.npy`; script builds Gaussian GT + `train/static_sequence/` | Default (no `DATA_ROOT`) |
| **Paired plates (Setup A)** | `DATA_ROOT/GT/` (one clean) + `DATA_ROOT/input/` (noisy frames) | Set **`DATA_ROOT=...`** — no `train/static_sequence/` |

Paired synthetic testing: [testing-synthetic-data.md](testing-synthetic-data.md).

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

One script trains, runs **full-plate tiled infer** (`FULL_INFER=1` default), and writes outputs under **`RUN`**.

**Gaussian path** — default sample plate `data/samples/A001C006_0101_4E4951.npy` (or pass another mosaic as the first argument). Builds Gaussian GT + `train/static_sequence/`, then trains. **Goal on this path:** multi-crop on the full sample (`train/crops/` + `manifest.json`), then **8000** epochs. Details: [Training progression](#training-progression-smoke--multi-crop).

```bash
bash scripts/static_noisy_sequence_pipeline_test.sh
bash scripts/static_noisy_sequence_pipeline_test.sh /path/to/your_mosaic.npy
```

**Paired path (`DATA_ROOT`)** — reads **`GT/`** + **`input/`** at the dataset root; **`RUN`** is only checkpoints and QA (no manual `static_sequence/` folder). See [Setup A](#real-paired-plates-setup-a).

Env vars (`NUM_CROPS`, `EPOCHS`, `DATA_ROOT`, `RUN`, …) go **before** `bash`.

**Smoke** (`EPOCHS=100`, auto **`SMOKE_CROP_PNG=1`** when `NUM_CROPS=1`):

```bash
# Gaussian — center 512×512 crop
CROP_SIZE=512 EPOCHS=100 bash scripts/static_noisy_sequence_pipeline_test.sh

# Paired — e.g. data/sythatic_sequence/{GT,input}
DATA_ROOT=data/sythatic_sequence RUN=runs/synthetic-sequence \
  CROP_SIZE=512 CROP_Y=4000 CROP_X=6000 EPOCHS=100 NUM_CROPS=1 \
  bash scripts/static_noisy_sequence_pipeline_test.sh
```

Use even **`CROP_Y` / `CROP_X`** inside your plate H×W, or omit them for a center crop.

| Stage | Command |
|-------|---------|
| **Multi-crop proof** | `NUM_CROPS=9 CROP_GRID=3x3 TILE=512 EPOCHS=100 bash scripts/static_noisy_sequence_pipeline_test.sh` |
| **Long run (goal)** | `NUM_CROPS=16 CROP_GRID=4x4 TILE=512 bash scripts/static_noisy_sequence_pipeline_test.sh` |

With no env vars, defaults are **8000** epochs, **`NUM_CROPS=1`** (center crop only), **`FULL_INFER=1`** (tiled infer on the sample plate). Each training step uses **one** manifest window when multi-crop is enabled.

**Compare outputs:**

| File | Role |
|------|------|
| `denoised_infer.npy` | Full-plate denoised mosaic |
| `noisy.png` | Full-frame **input** preview (same plate as tiled infer) |
| `denoised.png` | Full-frame **denoised** preview |

**Smoke train crops** (`EPOCHS=100`, `NUM_CROPS=1`, auto `SMOKE_CROP_PNG=1`): `train/noisy_input_crop.png`, `train/gaussian_GT_crop.png` (training window; not under `compare/`).

**Run root:** `training_curves.png` when `PLOT_HISTORY=1` (default).

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
- Smoke: **`train/noisy_input_crop.png`** and **`train/gaussian_GT_crop.png`** match the training window; **`compare/noisy.png`** and **`compare/denoised.png`** are full-frame QA.
- After step 2, open **`compare/`** for the two full-frame PNGs (+ `denoised_infer.npy`); **`training_curves.png`** at run root when plotted.

Optional debug: `--write-denoised-mosaic` or pipeline **`DEBUG_DENOISED_MOSAIC=1`** → `compare/denoised_mosaic.npy` (not canonical).

High PSNR vs Gaussian GT validates **pipeline wiring**, not real-world denoising quality.

## Real paired plates (Setup A)

**Aligned** noisy + clean Bayer mosaics (float16 H×W `.npy`, same shape, [float16 pipeline](float16-pipeline-standard.md)). **Do not** run `make_static_noisy_sequence.py` on real clean plates (it adds synthetic noise).

**Dataset layout** (prep once under repo `data/` or anywhere):

```text
data/my_clip/
  GT/          # one clean .npy (prefers clean_mosaic.npy if present)
  input/       # all noisy frames (*.npy, sorted)
```

**Pipeline** (recommended) — same script as Gaussian path, with **`DATA_ROOT`**:

```bash
DATA_ROOT=data/my_clip RUN=runs/my-real-paired \
  CROP_SIZE=512 EPOCHS=100 \
  bash scripts/static_noisy_sequence_pipeline_test.sh
```

**Trainer only** (no infer/plots):

```bash
python -m bayer_denoiser.train_static_noisy_sequence \
  --data-root data/my_clip \
  --output-dir runs/my-real-paired \
  --crop-size 512 \
  --num-crops 9 \
  --crop-grid 3x3 \
  --epochs 100 \
  --device mps
```

Multi-crop rules match [Training progression](#training-progression-smoke--multi-crop). **`--num-crops`** must match **`--crop-grid`** (`3x3→9`, `4x4→16`).

**Legacy:** `train/static_sequence/` with `clean_mosaic.npy` + `frame_*.npy` still works via **`--sequence-dir`** (manual assemble or old runs).

**Infer without pipeline:** tiled infer + the two compare PNGs (input + denoised):

```bash
mkdir -p runs/my-real-paired/compare
python -m bayer_denoiser.infer \
  --checkpoint runs/my-real-paired/model.pt \
  --input /path/to/noisy_full.npy \
  --output runs/my-real-paired/compare/denoised_infer.npy \
  --tiled --tile-size 512 --overlap 64
python -m bayer_denoiser.preview /path/to/noisy_full.npy \
  --write-mosaic runs/my-real-paired/compare/noisy.png
python -m bayer_denoiser.preview runs/my-real-paired/compare/denoised_infer.npy \
  --write-mosaic runs/my-real-paired/compare/denoised.png
```

PSNR vs real clean is meaningful on this path.

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

# Smoke (Gaussian) — center crop
CROP_SIZE=512 EPOCHS=100 bash scripts/static_noisy_sequence_pipeline_test.sh

# Smoke (paired DATA_ROOT) — off-center example
DATA_ROOT=data/sythatic_sequence RUN=runs/synthetic-sequence \
  CROP_SIZE=512 CROP_Y=4000 CROP_X=6000 EPOCHS=100 NUM_CROPS=1 \
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

**Gaussian path** (`runs/static-noisy-sequence-test/`):

```text
runs/<RUN>/
  model.pt, model_epoch*.pt, history.json
  training_curves.png             # PLOT_HISTORY=1 (run root, not compare/)
  train/
    static_sequence/              # clean_mosaic.npy + frame_*.npy
    crops/                        # multi-crop only: manifest.json + noisy_*/gaussian_GT_*
    noisy_input_crop.png          # smoke (SMOKE_CROP_PNG=1)
    gaussian_GT_crop.png          # smoke; name kept for Gaussian path
  compare/
    denoised_infer.npy
    noisy.png                     # full-frame input preview
    denoised.png                  # full-frame denoised preview
```

**Paired path (`DATA_ROOT`)** — same run layout except **no** `train/static_sequence/`; data stays under **`DATA_ROOT/GT`** and **`DATA_ROOT/input`**. Smoke crop PNGs still under **`train/`** (real clean + noisy at the training window).

Same checkpoint / compare PNG convention as [single-frame pipeline test](static-noisy-singleframe-pipeline-test.md).

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

### Patch-only infer (`FULL_INFER=0`)

Default pipeline uses **full-plate** tiled infer. With **`FULL_INFER=0`**, `denoised_infer.npy` and **`denoised.png`** are **patch-sized** (training window); **`noisy.png`** matches that patch.

### Training curves

`history.json` rows: **`epoch`**, **`loss`**, **`psnr`** (older files may use `avg_loss` / `avg_psnr`).

```bash
python -m bayer_denoiser.plot_training_history \
  --history runs/static-noisy-sequence-test/history.json \
  --output runs/static-noisy-sequence-test/training_curves.png
```

The pipeline writes **`$RUN/training_curves.png`** when **`PLOT_HISTORY=1`** (default; `0` to skip). Optional **`--title`**.

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
| `--crop-size`, `--crop-y`, `--crop-x` | off | Even square crop (center if origin omitted). Origins must lie inside loaded mosaic H×W (`--data-root` or `--sequence-dir`). Trainer errors if origin+crop exceeds bounds. |
| `--num-crops`, `--crop-grid` | off | Multi-crop windows on **full** mosaics (train-time crop; like pipeline `NUM_CROPS` / `CROP_GRID`) |
| `--crop-placement-seed` | 42 | Random window layout when `--crop-grid` omitted |
| `--crops-manifest` | auto | `train/crops/manifest.json` when present (do not combine with `--num-crops` / `--crop-grid`) |
| `--multi-crop-mode` | `random` | `random` or `cycle` when manifest has 2+ entries |
| `--resume` | off | Continue from checkpoint; `--epochs` = more epochs |
| `--no-resume-snapshot` | off | Skip pre-resume snapshot copy |
| `--write-denoised-mosaic` | off | Debug mosaic from frame_0000 forward pass |
