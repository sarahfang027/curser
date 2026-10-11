# Gaussian GT pipeline test (pseudo ground truth)

Use a **Gaussian-smoothed** Bayer mosaic as synthetic **clean** reference to exercise **train → checkpoint → infer → PSNR** without a noise-free plate.

## Canonical location: your Mac repo

Everything for this test should live under **`~/Projects/bayer_rgb`**, not only in Cursor Context / cloud.

**One-time install** (copies scripts + this doc from synced Context into the repo):

```bash
STORE="$HOME/Library/Application Support/Cursor/AgentStores/cursor_agent_stores/bc-a64d420e-f2ca-403e-91f6-786d63f9c464/files"
bash "$STORE/scripts/sync_context_to_repo.sh"
```

Then use:

| What | Path |
|------|------|
| This guide | `~/Projects/bayer_rgb/docs/inference/static-noisy-singleframe-pipeline-test.md` |
| Pipeline script | `~/Projects/bayer_rgb/scripts/static_noisy_singleframe_pipeline_test.sh` |
| Gaussian GT builder | `~/Projects/bayer_rgb/scripts/make_bayer_gaussian_gt.py` |
| Pair overfit | `~/Projects/bayer_rgb/scripts/overfit_bayer_denoise_pair.py` |

Re-run the sync script after Context updates. Prefer editing files **in the repo** and committing locally; treat Context as a backup mirror.

**Important:** Gaussian GT is **not** true ground truth — it is a **soft, biased target** (detail loss, cross-channel bleed if you blur wrong). Use it to validate **plumbing** (shapes, packed RGGB, I/O, metrics). Do not treat high PSNR vs Gaussian as “real” denoise quality.

Related: [A001C006 basic denoise test](a001c006-basic-denoise-test.md), [training user guide](../guides/training-user-guide.md).

---

## Roles of each `.npy`

| File | Role |
|------|------|
| **`noisy_mosaic.npy`** | Input — e.g. center crop of your sky sample (`A001C006_…`) |
| **`clean_mosaic.npy`** | **Synthetic GT** — same geometry, Bayer-safe Gaussian smooth of `noisy` |
| **`denoised_mosaic.npy`** | Model output from infer (or overfit script) |
| **`clean_mosaic_full.npy`** | **Full-frame** Gaussian GT (same σ logic, no crop) |
| *(noisy full)* | Your source `.npy` (e.g. `data/samples/A001C006_…npy`) — not duplicated |

Naming matches **`runs/bayer-denoise-smoke/`** so you can reuse the same PSNR snippets from the training guide.

**Tile vs full:** The default pipeline script builds **both** a **512×512** pair (for overfit + `infer`) and a **full-frame** Gaussian GT file. Full-frame **`bayer_rgb.infer`** is still likely to **OOM** on ~17K×18K; use full GT for metrics, exports, or future tiled training — not for the default overfit step.

---

## Step 1 — Build Gaussian GT

Per-plane Gaussian on packed RGGB (never blur the raw H×W mosaic as one image).

From repo root, copy scripts from Project Context into `scripts/` if needed:

```bash
cd ~/Projects/bayer_rgb
source .venv/bin/activate
# scipy optional — make_bayer_gaussian_gt.py falls back to torch if scipy is missing

# Center 512 crop saved as noisy fixture, then Gaussian GT
mkdir -p runs/static-noisy-singleframe-test
python - <<'PY'
import numpy as np
from pathlib import Path
src = np.load("data/samples/A001C006_0101_4E4951.npy")
ts = 512
h, w = src.shape
top = (h - ts) // 2; top -= top % 2
left = (w - ts) // 2; left -= left % 2
crop = src[top:top+ts, left:left+ts].astype(np.float32)
np.save("runs/static-noisy-singleframe-test/noisy_mosaic.npy", crop)
PY

python scripts/make_bayer_gaussian_gt.py \
  --input runs/static-noisy-singleframe-test/noisy_mosaic.npy \
  --output runs/static-noisy-singleframe-test/clean_mosaic.npy \
  --sigma 1.0
```

Tune **`--sigma`** (typical **0.5–2.0** on 512 crops). Larger σ → easier target, more blur.

Optional: add extra AWGN on **`noisy_mosaic.npy` only** before saving (keeps GT fixed):

```python
noise_std = 0.02
noisy = np.clip(crop + np.random.randn(*crop.shape).astype(np.float32) * noise_std, 0, 1)
```

---

## Step 2 — Full-frame Gaussian GT only

No crop; writes **`runs/static-noisy-singleframe-test/clean_mosaic_full.npy`**. Noisy input is the source file path.

```bash
cd ~/Projects/bayer_rgb
source .venv/bin/activate
SIGMA=1.0 bash scripts/make_full_frame_gaussian_gt.sh \
  data/samples/A001C006_0101_4E4951.npy
```

Or by hand:

```bash
python scripts/make_bayer_gaussian_gt.py \
  --input data/samples/A001C006_0101_4E4951.npy \
  --output runs/static-noisy-singleframe-test/clean_mosaic_full.npy \
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
clean = np.load("runs/static-noisy-singleframe-test/clean_mosaic_full.npy")
print(f"full noisy vs Gaussian GT: {psnr(noisy, clean):.2f} dB")
PY
```

Use a **center crop** of `denoised_infer.npy` vs the same window in `clean_mosaic_full.npy` once you have a model output at full size.

---

## Step 3 — Full pipeline (tile train + full GT)

```bash
bash scripts/static_noisy_singleframe_pipeline_test.sh
# or with custom sample / tile / sigma:
TILE=512 SIGMA=1.0 bash scripts/static_noisy_singleframe_pipeline_test.sh \
  ~/Projects/bayer_rgb/data/samples/A001C006_0101_4E4951.npy
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

| Artifact | Meaning |
|----------|---------|
| `noisy_mosaic.npy` | Input crop |
| `clean_mosaic.npy` | Gaussian GT (tile) |
| `clean_mosaic_full.npy` | Gaussian GT (**full frame**, if `BUILD_FULL_GT=1`) |
| `model.pt` | Overfit checkpoint (tile) |
| `history.json` | Overfit epochs |
| `denoised_mosaic.npy` | Direct overfit forward |
| `denoised_infer.npy` | Same pair via **`python -m bayer_rgb.infer`** (tile) |
| `*.png` | Preview mosaics (tile) |

**Success criteria (plumbing):**

- Overfit **train PSNR** rises (often **>30 dB** on a single 512 patch).
- **`denoised_infer.npy`** matches **`denoised_mosaic.npy`** closely (same checkpoint path).
- **`denoised vs clean_mosaic` PSNR** beats **`noisy vs clean_mosaic`**.

---

## Step 4 — Eval only (no custom overfit)

If you only want **infer + metrics** against Gaussian GT (smoke checkpoint):

```bash
python -m bayer_rgb.infer \
  --checkpoint runs/bayer-denoise-smoke/model.pt \
  --input runs/static-noisy-singleframe-test/noisy_mosaic.npy \
  --output runs/static-noisy-singleframe-test/denoised_smoke.npy
```

Expect **weak** fit — smoke weights were trained on random synthetic patches, not your sky clip.

---

## What the stock `train` CLI does *not* do

`python -m bayer_rgb.train --model bayer_denoise` uses **`SyntheticBayerDenoiseDataset`** (random RGB patches) or **paired RAW** plates — it does **not** load arbitrary `(noisy.npy, clean.npy)` pairs.

For Gaussian GT pairs, use:

- **`scripts/overfit_bayer_denoise_pair.py`** (single-patch overfit test), or  
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
noisy = np.load(run / "noisy_mosaic.npy")
clean = np.load(run / "clean_mosaic.npy")
den = np.load(run / "denoised_infer.npy")
print(f"noisy vs Gaussian GT:    {psnr(noisy, clean):.2f} dB")
print(f"denoised vs Gaussian GT: {psnr(den, clean):.2f} dB")
PY
```

Compare in **mosaic** space here; training logs PSNR in **packed** space — both are valid; numbers differ slightly.

---

## Limitations

- Gaussian GT **caps** how “good” the model can look vs a true clean plate.
- Full-frame **17K×18K** — build GT on **crops** first; tiled infer is still a repo gap.
- For real quality work, switch GT to **paired clean RAW** or controlled synthetic noise with known clean (see training guide Phase B).
