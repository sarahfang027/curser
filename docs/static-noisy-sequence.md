# Static noisy sequence (20 frames)

## What it is

A **static sequence** is a short "video" where **nothing moves**: every frame shows the **same clean image**, but each frame gets its **own fresh random noise**. Think of taking 20 photos of a still scene on a tripod in the dark — the scene is identical, the grain is different each time.

Why bother:

- Test a denoiser on several noisy inputs with one known-clean answer.
- Later, try multi-frame (temporal) ideas: averaging frames should cancel noise.
- Keeps the test simple — no motion, no alignment.

The clean image here is the **Gaussian GT** (a blurred stand-in for "clean", not real ground truth). See [Gaussian GT pipeline test](gaussian-gt-pipeline-test.md).

## Workflow

1. **Build the clean image** (Gaussian GT) — already done if `runs/gaussian-gt-test/train/gaussian_GT_crop.npy` exists.
2. **Run the sequence script** — it adds noise 20 times and saves the frames.
3. **Train** with checkpoints at the **run root**; put **infer + PNG previews** under **`compare/`** (same layout as [Gaussian GT pipeline test](gaussian-gt-pipeline-test.md)).

**One-shot (recommended):** `bash scripts/static_noisy_sequence_pipeline_test.sh` → `runs/static-noisy-test/` with `train/static_sequence/`, `model.pt` at run root, and `compare/denoised_infer.npy` + `*.png`.

Legacy top-level sequence folder `runs/static_sequence/` still works; see [Run folder layout](#run-folder-layout) below.

## Phased plan (current)

**Phase 1 — plumbing (now):** fixed **center crop** on the full-plate static sequence, train with **`python -m bayer_denoiser.train_static_noisy_sequence`**, confirm loss/PSNR improve and **`bayer_denoiser.infer`** matches the checkpoint on the **same 512×512 patch**. Do **not** use `--random-crop` yet — you want many steps on one patch, not one step per patch.

**Phase 2 — cover the plate (later):** tiled or multi-patch training, optional smaller mosaics on disk (`--tile-size` when building the sequence), crop-per-epoch or grid of fixed patches, tiled infer on ~17K×18K. Not implemented as one command yet; extend after Phase 1 is green.

### Phase 1 checklist (full plate on disk, 512 train patch)

**Canonical run layout** (`runs/static-noisy-test/`):

```bash
cd ~/Projects/bayer_denoiser && source .venv/bin/activate
bash scripts/sync_context_to_repo.sh   # if scripts/docs lag Context

bash scripts/static_noisy_sequence_pipeline_test.sh
# full-plate sample, 512 train crop, 20 frames (quick test: lower EPOCHS):
CROP_SIZE=512 EPOCHS=50 bash scripts/static_noisy_sequence_pipeline_test.sh
```

Artifacts: **`compare/denoised_infer.npy`**, **`compare/noisy.png`**, **`compare/clean_gaussian_gt.png`**, **`compare/denoised.png`**; checkpoints at **`runs/static-noisy-test/model.pt`**.

**Manual steps** (same semantics):

```bash
PIPELINE_RUN=runs/static-noisy-test bash scripts/make_static_noisy_sequence.sh

python -m bayer_denoiser.train_static_noisy_sequence \
  --sequence-dir runs/static-noisy-test/train/static_sequence \
  --output-dir runs/static-noisy-test \
  --crop-size 512 \
  --device cpu

python -m bayer_denoiser.infer \
  --checkpoint runs/static-noisy-test/model.pt \
  --input runs/static-noisy-test/compare/infer_input_crop.npy \
  --output runs/static-noisy-test/compare/denoised_infer.npy
```

(`static_noisy_sequence_pipeline_test.sh` writes **`compare/infer_input_crop.npy`** when `CROP_SIZE` is set.)

**Legacy** top-level `runs/static_sequence/` + ad-hoc output dir:

```bash
python -m bayer_denoiser.train_static_noisy_sequence \
  --sequence-dir runs/static_sequence \
  --output-dir runs/static-seq-plumbing \
  --crop-size 512 \
  --device cpu

python scripts/input_and_infer_crop.py \
  --input runs/static_sequence/frame_0000.npy \
  --checkpoint runs/static-seq-plumbing/model.pt
```

`input_and_infer_crop.py` defaults **`--output-dir`** to **`{checkpoint.parent}/compare`**. Writes **`frame_0000_cropped_512.npy`** and **`frame_0000_denoised_512.npy`**. For parity with the Gaussian pipeline, prefer **`denoised_infer.npy`** via `bayer_denoiser.infer` or the static pipeline script.

**Plumbing success:** training PSNR clearly above noisy PSNR on the patch; **`compare/denoised_infer.npy`** matches training forward on the same patch. Optional debug: `--write-denoised-mosaic` or pipeline **`DEBUG_DENOISED_MOSAIC=1`** → **`compare/denoised_mosaic.npy`** only (not canonical).

## Run it

From the Mac repo (`~/Projects/bayer_denoiser`):

```bash
cd ~/Projects/bayer_denoiser
source .venv/bin/activate

python scripts/make_static_noisy_sequence.py \
  --clean runs/gaussian-gt-test/train/gaussian_GT_crop.npy \
  --output-dir runs/static_sequence \
  --frames 20 \
  --noise-std 0.02 \
  --seed 42 \
  --write-manifest
```

Or the shortcut with defaults (writes **`runs/static_sequence/`** — legacy top-level path):

```bash
bash scripts/make_static_noisy_sequence.sh
NOISE_STD=0.03 SEED=1 bash scripts/make_static_noisy_sequence.sh   # tweak noise / seed
```

Train-relative sequence (matches **`static_noisy_sequence_pipeline_test.sh`**):

```bash
PIPELINE_RUN=runs/static-noisy-test bash scripts/make_static_noisy_sequence.sh
```

No clean file yet? Build it on the fly from a noisy mosaic:

```bash
python scripts/make_static_noisy_sequence.py \
  --input runs/gaussian-gt-test/train/noisy_input_crop.npy --sigma 1.0 \
  --output-dir runs/static_sequence
```

## Options

| Option | Default | Meaning |
|--------|---------|---------|
| `--clean` | — | Clean H×W mosaic `.npy` (use this **or** `--input`) |
| `--input`, `--sigma` | — / 1.0 | Noisy mosaic + blur strength to build Gaussian GT first |
| `--tile-size` | 0 | With `--input`: center-crop size (0 = full frame) |
| `--output-dir` | required | Where frames are written |
| `--frames` | 20 | Number of frames |
| `--noise-std` | 0.02 | Noise strength on pixel values in [0, 1] (bigger = grainier) |
| `--seed` | 0 | Same seed → identical frames every run |
| `--write-manifest` | off | Also write `manifest.json` listing files and settings |

## Run folder layout

| Layout | Sequence path | Checkpoints | Compare |
|--------|---------------|-------------|---------|
| **Pipeline (canonical)** | `runs/static-noisy-test/train/static_sequence/` | `runs/static-noisy-test/model.pt` | `runs/static-noisy-test/compare/` |
| **Legacy** | `runs/static_sequence/` | any `--output-dir` you choose | `{checkpoint.parent}/compare/` when using `input_and_infer_crop.py` or pipeline script |

```text
runs/static-noisy-test/
  train/
    noisy_input_crop.npy
    gaussian_GT_crop.npy
    static_sequence/
      clean_mosaic.npy
      frame_0000.npy … frame_0019.npy
      manifest.json
  compare/
    denoised_infer.npy    # canonical (bayer_denoiser.infer)
    noisy.png
    clean_gaussian_gt.png
    denoised.png
  model.pt
  model_epoch0100.pt
  history.json
```

## What you get (sequence folder only)

```
runs/static_sequence/          # legacy default for make_static_noisy_sequence.sh
  clean_mosaic.npy     the one clean image (saved once)
  frame_0000.npy       noisy copy 1
  ...
  frame_0019.npy       noisy copy 20
  manifest.json        (only with --write-manifest)
```

- All files are **float16**, H×W Bayer mosaic (RGGB), values in [0, 1] — same as the rest of the pipeline.
- Noise is added in float32, clipped to [0, 1], then saved as float16.
- The script prints each frame's PSNR vs clean, computed in **float32**. At `--noise-std 0.02` expect roughly **34 dB**.
- Frames differ from each other only by noise; change `--seed` for a different set.

## Train on the sequence

`bayer_denoiser.train_static_noisy_sequence` trains `BayerDenoiseNet` (`bayer_denoise`) on the folder: it loads `clean_mosaic.npy` once, then each epoch visits every `frame_*.npy` (shuffled) and minimizes L1 between the network output and the same clean packed RGGB target. Use **`--frames 1`** when you only want `frame_0000`.

**RGGB packing is centralized in `src/sort_bayer.py`.** Training, infer, preview and the helper scripts all use the same channel order — **R, G1 (`[1::2, 0::2]`), G2 (`[0::2, 1::2]`), B**. Train/scripts use the NumPy twins `mosaic_to_packed_np` / `packed_to_mosaic_np` (float32, identical to the torch `bayer_mosaic_to_packed` / `packed_to_bayer_mosaic`); do not re-implement pack/unpack locally.

> **Retrain needed:** checkpoints trained with `train_static_noisy_sequence` *before* this change used a swapped G1/G2 order, so `infer` (which always used the canonical order) fed them mismatched green channels. Retrain such checkpoints for correct infer alignment.

```bash
cd ~/Projects/bayer_denoiser
source .venv/bin/activate

python -m bayer_denoiser.train_static_noisy_sequence \
  --sequence-dir runs/static_sequence \
  --output-dir runs/static-seq-train \
  --crop-size 512
```

On a Mac, full-frame Bayer U-Nets can exhaust RAM (`zsh: killed` on CPU) or hit MPS conv limits. Use an even `--crop-size` (512 or 256). Add `--random-crop` with `--crop-size` for a new random crop each training step.

```bash
python -m bayer_denoiser.train_static_noisy_sequence \
  --sequence-dir runs/static_sequence \
  --output-dir runs/static-seq-train-mps \
  --crop-size 512 \
  --device mps
```

**Resume** (same `--sequence-dir`, same `--output-dir` as the first run; `--epochs` = *additional* epochs):

```bash
python -m bayer_denoiser.train_static_noisy_sequence \
  --sequence-dir runs/static_sequence \
  --output-dir runs/static-seq-plumbing \
  --resume runs/static-seq-plumbing/model_epoch0100.pt \
  --epochs 100 \
  --device cpu
```

Crop / `nf` are taken from the checkpoint (do not pass conflicting `--crop-size` or `--nf`). `history.json` is appended; epoch numbers continue (e.g. 101–200 after a 100-epoch run). New checkpoints include `optimizer_state_dict` for a proper Adam resume; older `model.pt` files still load weights only.

**Before each resume**, the script copies `--resume` into the same `--output-dir` as **`model_snapshot_epochXXXX.pt`** (epoch = last completed epoch). If that name already exists, it uses `_2`, `_3`, … Skip with `--no-resume-snapshot`.

```bash
python -m bayer_denoiser.train_static_noisy_sequence \
  --sequence-dir runs/static_sequence \
  --output-dir runs/static-seq-plumbing \
  --resume runs/static-seq-plumbing/model.pt \
  --epochs 100
```

| Option | Default | Meaning |
|--------|---------|---------|
| `--sequence-dir` | required | Folder with `clean_mosaic.npy` + `frame_*.npy` |
| `--output-dir` | `runs/static-seq-train` | Where results go |
| `--epochs` | 8000 | Passes over all frames |
| `--lr` | 1e-3 | Adam learning rate |
| `--nf` | 64 | Network width |
| `--device` | auto | e.g. `mps`, `cpu` |
| `--frames` | all | Use only the first N frames (sorted by name) |
| `--seed`, `--no-shuffle` | 0 / off | Frame order control |
| `--crop-size`, `--crop-y`, `--crop-x` | off | Even square crop (center if origin omitted) |
| `--random-crop` | off | New random crop each step (needs `--crop-size`) |
| `--resume` | off | Path to `model.pt`; train `--epochs` more, append `history.json` |
| `--no-resume-snapshot` | off | Skip `model_snapshot_epochXXXX.pt` copy before resume |
| `--write-denoised-mosaic` | off | Write `denoised_mosaic.npy` (frame_0000 training forward, float16); debug only |

Outputs in `--output-dir` (prefer **run root**, not `train/`):

```
model_epoch0100.pt            canonical checkpoint (epoch count in filename = last completed epoch)
model.pt                      same weights as model_epochXXXX.pt (convenience for infer scripts)
model_snapshot_epoch0100.pt   copy of --resume checkpoint before continuing (default on resume)
history.json                  per-epoch avg loss and avg PSNR (packed, float32 metric)
denoised_mosaic.npy           optional (--write-denoised-mosaic): debug only; pipeline moves to compare/
```

**Run roots:** Gaussian GT → `--output-dir runs/gaussian-gt-test`; static sequence pipeline → `--output-dir runs/static-noisy-test`. Sequence data lives under **`train/`** (`tile_train_seq/` or `train/static_sequence/`). Compare artifacts → **`compare/`** via `static_noisy_sequence_pipeline_test.sh` or manual `bayer_denoiser.infer` + `preview`.

After training it prints noisy and denoised PSNR vs clean for every frame, plus the means. Infer with the checkpoint as usual:

```bash
python -m bayer_denoiser.infer \
  --checkpoint runs/static-noisy-test/model.pt \
  --input runs/static-noisy-test/train/static_sequence/frame_0000.npy \
  --output runs/static-noisy-test/compare/denoised_infer.npy
```

Caveat: all frames share one clean target, so the net can partly memorize that image; PSNR here checks the training plumbing, not generalization.

**One frame only** (e.g. full-plate static sequence with crop):

```bash
python -m bayer_denoiser.train_static_noisy_sequence \
  --sequence-dir runs/static_sequence \
  --output-dir runs/static-seq-train-frame0 \
  --frames 1 \
  --crop-size 512 \
  --random-crop \
  --epochs 80 \
  --device cpu
```

Check mosaic size: `python -c "import numpy as np; print(np.load('runs/static_sequence/clean_mosaic.npy').shape)"`

## Try it on a frame

After **`bash scripts/static_noisy_sequence_pipeline_test.sh`**, open **`runs/static-noisy-test/compare/`**. To eval a Gaussian-tile checkpoint on a legacy sequence frame:

```bash
python -m bayer_denoiser.infer \
  --checkpoint runs/gaussian-gt-test/model.pt \
  --input runs/static_sequence/frame_0000.npy \
  --output runs/gaussian-gt-test/compare/denoised_infer.npy
```

Reminder: high PSNR vs Gaussian GT checks the **plumbing**, not real-world quality.
