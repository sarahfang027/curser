# Training guide — Bayer-space denoising (`--model bayer_denoise`)

This guide covers the **current** training path: **packed RGGB Bayer in → packed clean Bayer out**. You do not need RGB targets at train or infer time for this model; RGB references are only used on real plates to **align random crops** with the same window as the demosaic/GCP workflow.

**Deferred:** joint demosaic + denoise to RGB with `--model gcp` (GCPStillNet). See [GCP_STILL.md](GCP_STILL.md) and [MANUAL.md](MANUAL.md) when that path matters.

**Related:** [BAYER_DENOISE.md](BAYER_DENOISE.md) (model summary), [RAW_SPEC.md](../formats/RAW_SPEC.md) (`.raw` contract), [PIPELINE.md](PIPELINE.md) (smoke scripts).

---

## Prerequisites

From the repo root (`~/Projects/bayer_denoiser`):

```bash
cd ~/Projects/bayer_denoiser
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .   # recommended: import bayer_denoiser from any directory
```

| File | When you need it |
|------|------------------|
| **`requirements.txt`** | Always — PyTorch, torchvision, numpy, Pillow, tqdm |
| **`requirements-raw.txt`** | When you read **`.exr`** RGB references (paired plates) or other formats that go through **imageio** in the loader |

```bash
pip install -r requirements-raw.txt   # before Phase B if rgb/ uses .exr
```

Activate the venv for every command below:

```bash
source .venv/bin/activate
```

Entry point for training:

```bash
python -m bayer_denoiser.core.train -h
```

(`src/train.py` — do not use `--data-dir` with `bayer_denoise`; the CLI will exit with an error.)

---

## Device defaults (Mac MPS)

`train` and `infer` call `default_device_string()`:

1. **`cuda`** if NVIDIA CUDA is available  
2. Else **`mps`** on Apple Silicon when MPS is available  
3. Else **`cpu`**

Override explicitly when debugging or if MPS misbehaves:

```bash
python -m bayer_denoiser.core.train ... --device cpu
python -m bayer_denoiser.core.infer ... --device cpu
```

Training uses `num_workers=0` (simple, works on Mac). Pin memory is enabled only for CUDA.

---

## Phase A — No dataset (synthetic)

Use this to prove **train → checkpoint → infer** before production plates exist.

### Step-by-step

1. **Activate venv** (see above).

2. **Run a short synthetic train** (defaults below are a reasonable starting point; the smoke script uses smaller sizes for speed):

   ```bash
   python -m bayer_denoiser.core.train \
     --model bayer_denoise \
     --epochs 5 \
     --batch-size 4 \
     --image-size 128 \
     --synthetic-length 64 \
     --noise-std 0.05 \
     --lr 1e-4 \
     --output-dir runs/bayer-denoise-synthetic
   ```

   What happens internally:

   - `SyntheticBayerDenoiseDataset` draws random RGB patches, converts to Bayer, adds Gaussian noise (`--noise-std`), packs to `(4, H/2, W/2)`.
   - Target is **clean packed Bayer** (same shape).
   - Loss: **L1** in packed space. Logged **PSNR** is also in packed space vs the clean target.

3. **Inspect outputs** under `--output-dir`:

   | Artifact | Purpose |
   |----------|---------|
   | **`model.pt`** | Checkpoint: weights, `model`, `nf`, `noise_std`, `image_size`, saved args |
   | **`history.json`** | Per-epoch metrics |

4. **Read `history.json`** — JSON array, one object per epoch:

   ```json
   [
     { "epoch": 1, "loss": 0.469, "psnr": 6.45 },
     { "epoch": 2, "loss": 0.445, "psnr": 6.85 }
   ]
   ```

   - **`loss`**: average L1 for that epoch (lower is better).  
   - **`psnr`**: average PSNR in **packed** space (higher is better).  
   - Expect noisy early values on synthetic data; trend matters more than absolute dB for a 5-epoch smoke run.

5. **Infer on a Bayer mosaic** (`.npy` float32 **H×W** in [0, 1], or `.raw` + sidecar — see infer section).

### Recommended flags (Phase A)

| Flag | Suggested | Notes |
|------|-----------|--------|
| `--model` | `bayer_denoise` | Required |
| `--epochs` | `3–10` | More once loss stabilizes |
| `--batch-size` | `4–8` | Lower if OOM |
| `--image-size` | `64–128` | Even; patch edge in synthetic mode |
| `--synthetic-length` | `64–512` | Samples **per epoch** (default **512**) |
| `--noise-std` | `0.02–0.05` | Must match infer/eval noise level you care about |
| `--nf` | `64` (default) | Base width; U-Net uses `max(16, nf//2)` channels |
| `--lr` | `1e-4` (default) | Adam |
| `--output-dir` | `runs/...` | Created if missing |
| `--seed` | `42` (default) | Reproducible synthetic noise |

### One-command smoke

```bash
./scripts/bayer_denoise_smoke.sh
```

Writes **`runs/bayer-denoise-smoke/`** (override with `OUT=my-run ./scripts/bayer_denoise_smoke.sh`): trains 3 epochs, builds tiny `.npy` mosaics, runs infer, prints train + infer PSNR. Use this after env changes or code edits.

### Viewing smoke test outputs

After the smoke script succeeds, the output folder (default **`runs/bayer-denoise-smoke/`** under the repo) contains:

| File | Purpose |
|------|---------|
| `model.pt` | Trained checkpoint |
| `history.json` | Per-epoch training loss and PSNR |
| `noisy_mosaic.npy` | Synthetic noisy input (64×64) |
| `clean_mosaic.npy` | Clean reference mosaic (for PSNR only) |
| `denoised_mosaic.npy` | Model output from `infer` |

**Finder:** **Go → Go to Folder…** (⇧⌘G) and paste:

```text
~/Projects/bayer_denoiser/runs/bayer-denoise-smoke
```

If you set `OUT=my-run`, open `~/Projects/bayer_denoiser/my-run` instead.

**`.npy` files are not pictures** — macOS Preview cannot show them. Use Terminal commands below to read numbers or write PNGs.

**PSNR check (Terminal):** The smoke script already prints `train_last_psnr`, `infer_psnr_noisy_vs_clean`, and `infer_psnr_denoised_vs_clean` at the end. To recompute infer PSNR without retraining (venv active, repo root):

```bash
cd ~/Projects/bayer_denoiser
source .venv/bin/activate
python - <<'PY'
import json
import numpy as np
from pathlib import Path

run = Path("runs/bayer-denoise-smoke")

def psnr(a, b):
    mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
    return 99.0 if mse <= 0 else 10.0 * np.log10(1.0 / mse)

noisy = np.load(run / "noisy_mosaic.npy")
clean = np.load(run / "clean_mosaic.npy")
den = np.load(run / "denoised_mosaic.npy")
print(f"noisy vs clean: {psnr(noisy, clean):.2f} dB")
print(f"denoised vs clean: {psnr(den, clean):.2f} dB")
hist = json.loads((run / "history.json").read_text())
print("last train epoch:", hist[-1])
PY
```

**PNG previews:** `python -m bayer_denoiser.qa.preview` loads the same `.npy` mosaics as infer. For **`.npy`**, it writes an 8-bit PNG **next to the input** (`clean_mosaic.npy` → `clean_mosaic.png`) even if you only pass crop flags like `--tile-size`. Use **`--write-mosaic PATH.png`** to choose a different output path. For **`.raw`**, stats-only is the default; pass **`--write-mosaic`** to save a PNG.

```bash
cd ~/Projects/bayer_denoiser
source .venv/bin/activate
# Smoke mosaics (64×64): optional --tile-size crops before PNG; default PNG is <stem>.png
python -m bayer_denoiser.qa.preview runs/bayer-denoise-smoke/clean_mosaic.npy --tile-size 32
python -m bayer_denoiser.qa.preview runs/bayer-denoise-smoke/noisy_mosaic.npy \
  --write-mosaic runs/bayer-denoise-smoke/noisy_mosaic.png
python -m bayer_denoiser.qa.preview runs/bayer-denoise-smoke/denoised_mosaic.npy \
  --write-mosaic runs/bayer-denoise-smoke/denoised_mosaic.png
```

Open the `.png` files in Preview. Optionally preview `clean_mosaic.npy` the same way for a reference strip.

**Crop QA tool:** `python -m bayer_denoiser.archived.qa.visualize_crops` is for **paired** `data/bayer` + `data/rgb` plates (alignment strips). It does **not** apply to smoke outputs — there are no RGB references, only the `.npy` mosaics above.

---

## Phase B — Real plates (paired RAW)

RGB is **not** the training target for `bayer_denoise`. The loader sets **`bayer_target=True`**: target is **clean packed Bayer** from the plate; **`--noise-std`** adds noise on the input mosaic only. RGB paths are still required so **crop windows** match between Bayer partial reads and RGB slices (same as GCP training).

### Data layout

```text
data/
  bayer/
    shot_0100.raw
    shot_0100.raw.json
  rgb/
    shot_0100.exr          # or .png / .tif — same stem, same H×W as mosaic
```

Optional manifest **`pairs.csv`**:

```csv
bayer,rgb
/path/to/shot_0100.raw,/path/to/shot_0100.exr
```

Sidecar JSON, geometry, and naming rules: [RAW_SPEC.md](../formats/RAW_SPEC.md) and [MANUAL.md](MANUAL.md) §3.

Create sidecar if missing:

```bash
python -m bayer_denoiser.archived.qa.make_sidecar data/bayer/shot_0100.raw
```

### Validate before long runs

```bash
python -m bayer_denoiser.archived.qa.validate_raw --dir data/bayer

python -m bayer_denoiser.archived.qa.validate_pairs \
  --bayer-dir data/bayer \
  --rgb-dir data/rgb
```

Fix any `[MISMATCH]` on spatial size or missing sidecars before training.

### Visual crop QA

Same-window check as GCP training (strips show RGB vs bilinear for **alignment**, not Bayer-denoise targets):

```bash
python -m bayer_denoiser.archived.qa.visualize_crops \
  --bayer-dir data/bayer \
  --rgb-dir data/rgb \
  --pair-index 0 \
  --crop-size 256 \
  --num-crops 8 \
  --seed 42 \
  --output-dir runs/crop_visual_check
```

If strips look **shifted**, fix pairing or export before denoise training.

### Paired train command

```bash
python -m bayer_denoiser.core.train \
  --model bayer_denoise \
  --bayer-dir data/bayer \
  --rgb-dir data/rgb \
  --raw-dtype float16 \
  --black-level 0 \
  --white-level 1 \
  --image-size 256 \
  --crops-per-frame 16 \
  --batch-size 1 \
  --noise-std 0.02 \
  --epochs 30 \
  --lr 1e-4 \
  --nf 64 \
  --output-dir runs/bayer-denoise-plates
```

Use **`--pairs-csv pairs.csv`** instead of `--bayer-dir` / `--rgb-dir` if paths live outside `data/`.

### Large plates (e.g. 17592×18024)

The loader **does not load the full `.raw` every step** when `--image-size` > 0: it reads only the crop window. RGB crops are sliced to the same `(top, left, size)`.

| Setting | Guidance |
|---------|----------|
| **`--image-size`** | **128–256** typical; must be **even** |
| **`--batch-size`** | **1** on large patches; increase only if memory allows |
| **`--crops-per-frame`** | **8–32** for more diversity per plate per epoch |
| **`--full-frame`** | Avoid unless you have enormous RAM — trains full resolution |
| **`--noise-std`** | Synthetic noise on top of plate; use **0** only if you intentionally want noise-free inputs |

Raw decode flags (`--raw-width`, `--raw-height`, `--raw-dtype`, `--black-level`, `--white-level`, `--resize`) match [MANUAL.md](MANUAL.md) §7.2.

---

## Infer and evaluate `.npy` outputs

### Infer

```bash
python -m bayer_denoiser.core.infer \
  --checkpoint runs/bayer-denoise-plates/model.pt \
  --input data/bayer/shot_0100.raw \
  --output /tmp/shot_0100_denoised.npy
```

- **`bayer_denoise` checkpoints require `--output` ending in `.npy`** (or `.raw`).
- Input: `.npy` mosaic, `.raw` + sidecar, or other paths supported by `load_bayer` (see `infer -h`).
- Output: float32 **H×W** Bayer mosaic in [0, 1] (not packed).

Sidecar overrides match training:

```bash
  --black-level 0 --white-level 1 \
  --raw-dtype float16 --raw-width 17592 --raw-height 18024
```

**Memory:** `infer` loads the **full** input frame (no tile flag). Full 18K forwards may **OOM** on MPS/GPU; test on a cropped `.npy` or use `preview --tile-size 512` for decode-only QA ([MANUAL.md](MANUAL.md) §8).

### Evaluate denoised `.npy`

There is no separate `evaluate` CLI. Compare mosaics in numpy (same shape, [0, 1]):

```python
import json
import numpy as np
from pathlib import Path

def psnr(a, b):
    mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
    return 99.0 if mse <= 0 else 10.0 * np.log10(1.0 / mse)

run = Path("runs/bayer-denoise-smoke")
noisy = np.load(run / "noisy_mosaic.npy")
clean = np.load(run / "clean_mosaic.npy")
den = np.load(run / "denoised_mosaic.npy")
print("noisy vs clean:", psnr(noisy, clean))
print("denoised vs clean:", psnr(den, clean))
hist = json.loads((run / "history.json").read_text())
print("last train epoch:", hist[-1])
```

On real plates you rarely have a noise-free reference mosaic; use synthetic smoke or held-out clean `.npy` for quantitative checks.

---

## Common CLI flags (`python -m bayer_denoiser.core.train`)

| Flag | Default | Role for `bayer_denoise` |
|------|---------|---------------------------|
| `--epochs` | `10` | Full passes over the dataset |
| `--batch-size` | `8` | Lower to `1` for large `--image-size` on plates |
| `--image-size` | `128` | Synthetic patch size; **auto-crop edge** on paired RAW |
| `--noise-std` | `0.02` | Input noise (synthetic or on-plate); target stays clean Bayer |
| `--nf` | `64` | Feature width / U-Net scale |
| `--lr` | `1e-4` | Adam learning rate |
| `--output-dir` | `runs/default` | Writes `model.pt` + `history.json` |
| `--synthetic-length` | `512` | Phase A only — samples per epoch |
| `--crops-per-frame` | `1` | Phase B — random crops per plate per epoch |
| `--device` | cuda → mps → cpu | See device section |
| `--model` | `gcp` | **Must set `bayer_denoise`** for this guide |

### `--lr` (learning rate)

- **Default:** `1e-4` (0.0001)
- **Meaning:** Step size for **Adam** when updating weights after each batch.
- **Effect:**
  - **Higher** (e.g. `3e-4`, `1e-3`) — learns faster early, but loss can bounce or diverge.
  - **Lower** (e.g. `3e-5`, `1e-5`) — more stable, slower; useful if loss spikes or plateaus on long plate trains.
- **Typical:** Keep **`1e-4`** for Phase A synthetic; change only if training looks unstable or stuck.

### `--nf` (network feature width)

- **Default:** `64`
- **Meaning:** Main **width knob** for `BayerDenoiseNet`’s U-Net. Internally `base_channels = max(16, nf // 2)`, so **`--nf 64` → 32 base channels** (~459k parameters at default).
- **Effect:**
  - **Larger `--nf`** (e.g. `96`, `128`) — more capacity for harder noise/detail, **more memory and slower** training.
  - **Smaller `--nf`** (e.g. `32`) — lighter and faster, less MPS/GPU memory; may underfit difficult real sensor noise.
- **Typical:** **`64`** for smoke and first real runs. If you hit OOM, lower **`--batch-size`** or **`--image-size`** first (then `--nf` if needed).

**Together:** `--lr` controls *how large each optimization step is*; `--nf` controls *how big the model is*.

---

## Troubleshooting

| Symptom | Likely cause | What to do |
|---------|----------------|------------|
| **CUDA/MPS OOM** during train | `--image-size`, `--batch-size`, or `--nf` too large | Use `--batch-size 1`, `--image-size 128–256`, fewer `--crops-per-frame`; avoid `--full-frame` |
| **OOM on infer** | Full-resolution forward | Infer a smaller `.npy` crop; no built-in tiling |
| **Slow on Mac** | Large batches on CPU | Confirm MPS: omit `--device` or set `--device mps`; reduce batch/size |
| **`Reading .exr RGB requires: pip install imageio`** | Missing optional deps | `pip install -r requirements-raw.txt` |
| **`--model bayer_denoise requires synthetic or paired raw Bayer, not --data-dir`** | Wrong data mode | Drop `--data-dir`; use Phase A (no dirs) or Phase B (`--bayer-dir` + `--rgb-dir`) |
| **`bayer_denoise checkpoint requires --output ending in .npy`** | Wrong infer extension | Use `.npy` output |
| **Pair / size errors** | Stem or resolution mismatch | Run `validate_pairs`; match [RAW_SPEC.md](../formats/RAW_SPEC.md) |

---

## Quick reference

| Step | Command |
|------|---------|
| Smoke | `./scripts/bayer_denoise_smoke.sh` |
| Synthetic train | `python -m bayer_denoiser.core.train --model bayer_denoise ...` |
| Paired train | `--bayer-dir` + `--rgb-dir` (or `--pairs-csv`) + `--model bayer_denoise` |
| Infer | `python -m bayer_denoiser.core.infer --checkpoint ... --input ... --output ...npy` |
| Pair check | `python -m bayer_denoiser.archived.qa.validate_pairs ...` |
| Crop QA | `python -m bayer_denoiser.archived.qa.visualize_crops ...` |
