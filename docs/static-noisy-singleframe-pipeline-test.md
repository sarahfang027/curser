# Static noisy single-frame pipeline test

**Single-frame** smoke for static-sequence training plumbing: one noisy tile (`frame_0000`) against a **Gaussian-smoothed** synthetic clean (`gaussian_GT_crop.npy`). Exercises **train → checkpoint → infer → PSNR** without a noise-free plate.

Multi-frame goal run: [static-noisy-sequence.md](static-noisy-sequence.md) and `bash scripts/static_noisy_sequence_pipeline_test.sh` → `runs/static-noisy-sequence-test/`.

## Canonical location: your Mac repo

Everything for this test should live under **`~/Projects/bayer_denoiser`**, not only in Cursor Context / cloud.

**One-time install** (copies scripts + this doc from synced Context into the repo):

```bash
STORE="$HOME/Library/Application Support/Cursor/AgentStores/cursor_agent_stores/bc-a64d420e-f2ca-403e-91f6-786d63f9c464/files"
bash "$STORE/scripts/sync_context_to_repo.sh"
```

Then use:

| What | Path |
|------|------|
| This guide | `~/Projects/bayer_denoiser/docs/static-noisy-singleframe-pipeline-test.md` |
| Pipeline script | `~/Projects/bayer_denoiser/scripts/static_noisy_singleframe_pipeline_test.sh` |
| Gaussian GT builder | `~/Projects/bayer_denoiser/scripts/make_bayer_gaussian_gt.py` |
| Static-sequence train | `python -m bayer_denoiser.train_static_noisy_sequence` |

Re-run the sync script after Context updates. Prefer editing files **in the repo** and committing locally; treat Context as a backup mirror.

**Important:** Gaussian GT is **not** true ground truth — it is a **soft, biased target** (detail loss, cross-channel bleed if you blur wrong). Use it to validate **plumbing** (shapes, packed RGGB, I/O, metrics). Do not treat high PSNR vs Gaussian as “real” denoise quality.

Related: [A001C006 basic denoise test](archived/inference/a001c006-basic-denoise-test.md), [training user guide](archived/guides/training-user-guide.md).

Next: [Static noisy sequence](static-noisy-sequence.md) — 20 frames of the same clean image with independent noise each. Full **train / compare** layout: `bash scripts/static_noisy_sequence_pipeline_test.sh` → `runs/static-noisy-sequence-test/`.

---

## Run folder layout (`runs/static-noisy-singleframe-test/`)

| Subfolder | Contents |
|-----------|----------|
| **`train/`** | **Data only:** `noisy_input_crop.npy`, `gaussian_GT_crop.npy`, `gaussian_GT_full.npy`, `tile_train_seq/` |
| **`compare/`** | `denoised_infer.npy`, `noisy.png`, `clean_gaussian_gt.png`, `denoised.png`; optional `denoised_mosaic.npy` when `DEBUG_DENOISED_MOSAIC=1` |

Static **multi-frame** runs use the same pattern under **`runs/static-noisy-sequence-test/`** (`train/static_sequence/`, checkpoints at run root, `compare/`) — see [static-noisy-sequence.md](static-noisy-sequence.md). Legacy top-level **`runs/static_sequence/`** is still supported.
| **RUN root** (`runs/static-noisy-singleframe-test/`) | `model.pt`, `model_epoch*.pt`, `history.json` — sibling of `train/` and `compare/` |

```text
runs/static-noisy-singleframe-test/
  train/      data only
  compare/    infer + previews
  model.pt
  model_epoch0100.pt
  history.json
```

**Training epochs:** the pipeline script defaults to **`EPOCHS=100`** (quick smoke). Override for longer runs, e.g. `EPOCHS=500 bash scripts/static_noisy_singleframe_pipeline_test.sh`. The multi-frame sequence pipeline uses **8000** by default — see [static-noisy-sequence.md](static-noisy-sequence.md#training-epoch-defaults).

Scripts accept **`TRAIN_DIR`**, **`CHECKPOINT_DIR`** (default **`$RUN`**), **`COMPARE_DIR`**, **`RUN`**, and **`DEBUG_DENOISED_MOSAIC`** env overrides (see `static_noisy_singleframe_pipeline_test.sh` header). The pipeline passes **`--output-dir "$CHECKPOINT_DIR"`** to `train_static_noisy_sequence`. Set **`DEBUG_DENOISED_MOSAIC=1`** to also write/move **`compare/denoised_mosaic.npy`** (training-forward debug). `input_and_infer_crop.py` defaults compare output to `{checkpoint.parent}/compare` (i.e. `RUN/compare` for a RUN-root checkpoint).

**Migrating an old run:** `cd runs/static-noisy-singleframe-test && mv train/model.pt train/model_epoch*.pt train/history.json .` (and move any stray `train/denoised_mosaic.npy` to `compare/`).

---

## Roles of each `.npy`

| File | Role |
|------|------|
| **`train/noisy_input_crop.npy`** | Input — e.g. center crop of your sky sample (`A001C006_…`) |
| **`train/gaussian_GT_crop.npy`** | **Synthetic GT** — same geometry, Bayer-safe Gaussian smooth of `noisy` |
| **`compare/denoised_infer.npy`** | Canonical model output via **`python -m bayer_denoiser.infer`** |
| **`compare/denoised_mosaic.npy`** | Optional debug: training forward (`--write-denoised-mosaic` / `DEBUG_DENOISED_MOSAIC=1`) |
| **`train/gaussian_GT_full.npy`** | **Full-frame** Gaussian GT (same σ logic, no crop) |
| *(noisy full)* | Your source `.npy` (e.g. `data/samples/A001C006_…npy`) — not duplicated |

Naming matches **`runs/bayer-denoise-smoke/`** so you can reuse the same PSNR snippets from the training guide.

**Tile vs full:** The default pipeline trains and infers on a **512×512** center crop. It can also build **`train/gaussian_GT_full.npy`** for metrics. For **full-plate** denoised output without OOM, use tiled infer (`FULL_INFER=1` or `INFER_TILED=1` on the pipeline script, or **`bayer_denoiser.infer --tiled`** manually). See [Full-plate tiled infer](#full-plate-tiled-infer) below.

---

## Step 1 — Build Gaussian GT

Per-plane Gaussian on packed RGGB (never blur the raw H×W mosaic as one image).

From repo root, copy scripts from Project Context into `scripts/` if needed:

```bash
cd ~/Projects/bayer_denoiser
source .venv/bin/activate
# scipy optional — make_bayer_gaussian_gt.py falls back to torch if scipy is missing

# Center 512 crop saved as noisy fixture, then Gaussian GT
mkdir -p runs/static-noisy-singleframe-test/train
python - <<'PY'
import numpy as np
from pathlib import Path
src = np.load("data/samples/A001C006_0101_4E4951.npy")
ts = 512
h, w = src.shape
top = (h - ts) // 2; top -= top % 2
left = (w - ts) // 2; left -= left % 2
crop = src[top:top+ts, left:left+ts].astype(np.float32)
np.save("runs/static-noisy-singleframe-test/train/noisy_input_crop.npy", crop)
PY

python scripts/make_bayer_gaussian_gt.py \
  --input runs/static-noisy-singleframe-test/train/noisy_input_crop.npy \
  --output runs/static-noisy-singleframe-test/train/gaussian_GT_crop.npy \
  --sigma 1.0
```

Tune **`--sigma`** (typical **0.5–2.0** on 512 crops). Larger σ → easier target, more blur.

Optional: add extra AWGN on **`noisy_input_crop.npy` only** before saving (keeps GT fixed):

```python
noise_std = 0.02
noisy = np.clip(crop + np.random.randn(*crop.shape).astype(np.float32) * noise_std, 0, 1)
```

---

## Step 2 — Full-frame Gaussian GT only

No crop; writes **`runs/static-noisy-singleframe-test/train/gaussian_GT_full.npy`**. Noisy input is the source file path.

```bash
cd ~/Projects/bayer_denoiser
source .venv/bin/activate
SIGMA=1.0 bash scripts/make_full_frame_gaussian_gt.sh \
  data/samples/A001C006_0101_4E4951.npy
```

Or by hand:

```bash
python scripts/make_bayer_gaussian_gt.py \
  --input data/samples/A001C006_0101_4E4951.npy \
  --output runs/static-noisy-singleframe-test/train/gaussian_GT_full.npy \
  --sigma 1.0
```

(`--tile-size` defaults to **0** = full frame.)

Optional huge PNG: `PREVIEW_FULL=1 bash scripts/make_full_frame_gaussian_gt.sh`

Full-frame PSNR vs Gaussian GT (loads full arrays — ~1.2 GiB each):

```bash
python - <<'PY'
import numpy as np
from pathlib import Path
def psnr(a,b):
    mse = float(np.mean((a.astype(np.float32)-b.astype(np.float32))**2))
    return 99. if mse<=0 else 10*np.log10(1/mse)
noisy = np.load("data/samples/A001C006_0101_4E4951.npy")
clean = np.load("runs/static-noisy-singleframe-test/train/gaussian_GT_full.npy")
print(f"full noisy vs Gaussian GT: {psnr(noisy, clean):.2f} dB")
PY
```

Use a **center crop** of `compare/denoised_infer.npy` vs the same window in `train/gaussian_GT_full.npy` once you have a model output at full size.

---

## Step 3 — Full pipeline (tile train + full GT)

```bash
bash scripts/static_noisy_singleframe_pipeline_test.sh
# or with custom sample / tile / sigma:
TILE=512 SIGMA=1.0 bash scripts/static_noisy_singleframe_pipeline_test.sh \
  ~/Projects/bayer_denoiser/data/samples/A001C006_0101_4E4951.npy
```

Skip full-frame Gaussian (tile-only):

```bash
BUILD_FULL_GT=0 TILE=512 bash scripts/static_noisy_singleframe_pipeline_test.sh
```

Different σ on full frame:

```bash
SIGMA=1.0 SIGMA_FULL=1.5 bash scripts/static_noisy_singleframe_pipeline_test.sh
```

Writes under **`runs/static-noisy-singleframe-test/`**:

| Artifact | Location | Meaning |
|----------|----------|---------|
| `noisy_input_crop.npy` | `train/` | Input crop |
| `gaussian_GT_crop.npy` | `train/` | Gaussian GT (tile) |
| `gaussian_GT_full.npy` | `train/` | Gaussian GT (**full frame**, if `BUILD_FULL_GT=1`) |
| `model.pt` | RUN root | Training checkpoint (tile) |
| `model_epoch*.pt` | RUN root | Per-epoch checkpoint |
| `history.json` | RUN root | Training epochs (`epoch`, `loss`, `psnr` per row) |
| `training_curves.png` | `compare/` | Avg loss + PSNR vs epoch (`PLOT_HISTORY=1`, default) |
| `denoised_infer.npy` | `compare/` | Canonical output via **`python -m bayer_denoiser.infer`** (tile) |
| `denoised_mosaic.npy` | `compare/` | Optional debug: training forward (`DEBUG_DENOISED_MOSAIC=1`) |
| `*.png` | `compare/` | Preview mosaics (tile) |

**Success criteria (plumbing):**

- Training **PSNR** rises (often **>30 dB** on a single 512 patch).
- With **`DEBUG_DENOISED_MOSAIC=1`**, **`compare/denoised_infer.npy`** should match **`compare/denoised_mosaic.npy`** closely (same checkpoint path).
- **denoised vs gaussian_GT_crop** PSNR beats **noisy vs gaussian_GT_crop**.

---

## Step 4 — Eval only (no custom training)

If you only want **infer + metrics** against Gaussian GT (smoke checkpoint):

```bash
python -m bayer_denoiser.infer \
  --checkpoint runs/bayer-denoise-smoke/model.pt \
  --input runs/static-noisy-singleframe-test/train/noisy_input_crop.npy \
  --output runs/static-noisy-singleframe-test/compare/denoised_smoke.npy
```

Expect **weak** fit — smoke weights were trained on random synthetic patches, not your sky clip.

---

## What the stock `train` CLI does *not* do

`python -m bayer_denoiser.archived.core.train --model bayer_denoise` uses **`SyntheticBayerDenoiseDataset`** (random RGB patches) or **paired RAW** plates — it does **not** load arbitrary `(noisy.npy, clean.npy)` pairs.

For Gaussian GT pairs, use:

- **`bayer_denoiser.train_static_noisy_sequence`** (static sequence folder; `--frames 1` for one noisy frame), or  
- a future **`PairedNpyBayerDataset`** if you add it to the repo.

---

## PSNR check

```bash
python - <<'PY'
import numpy as np
from pathlib import Path

def psnr(a, b):
    mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
    return 99.0 if mse <= 0 else 10.0 * np.log10(1.0 / mse)

run = Path("runs/static-noisy-singleframe-test")
train = run / "train"
compare = run / "compare"
noisy = np.load(train / "noisy_input_crop.npy")
clean = np.load(train / "gaussian_GT_crop.npy")
den = np.load(compare / "denoised_infer.npy")
print(f"noisy vs Gaussian GT:    {psnr(noisy, clean):.2f} dB")
print(f"denoised vs Gaussian GT: {psnr(den, clean):.2f} dB")
PY
```

Compare in **mosaic** space here; training logs PSNR in **packed** space — both are valid; numbers differ slightly.

---

## Full-plate tiled infer

After the default tile train/infer smoke, export a **full-resolution** denoised mosaic:

```bash
FULL_INFER=1 bash scripts/static_noisy_singleframe_pipeline_test.sh
```

Or manually:

```bash
python -m bayer_denoiser.infer \
  --checkpoint runs/static-noisy-singleframe-test/model.pt \
  --input data/samples/A001C006_0101_4E4951.npy \
  --output runs/static-noisy-singleframe-test/compare/denoised_infer.npy \
  --tiled --tile-size 512 --overlap 64

python -m bayer_denoiser.preview \
  runs/static-noisy-singleframe-test/compare/denoised_infer.npy \
  --write-mosaic runs/static-noisy-singleframe-test/compare/denoised.png
```

**`compare/denoised_infer_patch.npy`** and **`denoised_patch.png`** remain the 512 center patch for quick checks. On-disk **`denoised_infer.npy`** is **`float16`** per [float16-pipeline-standard.md](float16-pipeline-standard.md).

---

## Limitations

- Gaussian GT **caps** how “good” the model can look vs a true clean plate.
- Full-frame **17K×18K** training still needs **`--crop-size`**; use **tiled infer** for full-plate export.
- For real quality work, switch GT to **paired clean RAW** or controlled synthetic noise with known clean (see training guide Phase B).
