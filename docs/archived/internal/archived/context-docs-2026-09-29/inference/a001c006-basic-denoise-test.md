# Basic denoise test — `A001C006_0101_4E4951.npy`

Run on your Mac at `~/Projects/bayer_rgb` (venv active). Input path:

```text
/Users/Sarah.Fang/Projects/bayer_rgb/data/samples/A001C006_0101_4E4951.npy
```

(or relative `data/samples/A001C006_0101_4E4951.npy` from repo root).

The mosaic is **~17712×18400** float32 in **[0, 1]** (from [sky conversion](../formats/sky-format-conversion.md)). **`infer` loads the full frame** — do not point the CNN at the full `.npy` on first test; use a **center crop** (512×512 is enough for a smoke-style check).

## One-shot script

From repo root:

```bash
cd ~/Projects/bayer_rgb
source .venv/bin/activate
pip install -e .   # if imports fail

bash scripts/denoise_sample_a001c006.sh
```

Or paste the commands in [scripts/denoise_sample_a001c006.sh](../../../scripts/denoise_sample_a001c006.sh) (copy into your repo if the file is not there yet).

## Manual steps

### 1. Sanity-check the mosaic

```bash
python -m bayer_rgb.preview data/samples/A001C006_0101_4E4951.npy
```

Expect printed shape on the order of `(17712, 18400)` and values in `[0, 1]`.

### 2. Center 512×512 crop (RGGB-aligned)

```bash
mkdir -p runs/sample-tests
python -m bayer_rgb.preview data/samples/A001C006_0101_4E4951.npy \
  --tile-size 512 --center \
  --write-mosaic runs/sample-tests/A001C006_noisy512.png
```

Export the same crop as `.npy` for `infer` (preview writes PNG only):

```bash
python - <<'PY'
import numpy as np
from pathlib import Path

# Match preview --center --tile-size 512 (even top/left for RGGB)
src = np.load("data/samples/A001C006_0101_4E4951.npy")
h, w = src.shape
ts = 512
top = (h - ts) // 2
top -= top % 2
left = (w - ts) // 2
left -= left % 2
crop = src[top : top + ts, left : left + ts].astype(np.float32)
out = Path("runs/sample-tests/A001C006_center512.npy")
out.parent.mkdir(parents=True, exist_ok=True)
np.save(out, crop)
print("saved", out, "shape", crop.shape, "min/max", crop.min(), crop.max())
PY
```

### 3. Bayer-space CNN denoise (`bayer_denoise`)

Use an existing checkpoint (smoke is fine for a **pipeline** test, not production quality):

```bash
python -m bayer_rgb.infer \
  --checkpoint runs/bayer-denoise-smoke/model.pt \
  --input runs/sample-tests/A001C006_center512.npy \
  --output runs/sample-tests/A001C006_denoised512.npy
```

If `runs/bayer-denoise-smoke/model.pt` is missing:

```bash
./scripts/bayer_denoise_smoke.sh
```

Then rerun `infer` above.

### 4. Preview denoised crop

```bash
python -m bayer_rgb.preview runs/sample-tests/A001C006_denoised512.npy \
  --write-mosaic runs/sample-tests/A001C006_denoised512.png
```

Open in Preview:

- `runs/sample-tests/A001C006_noisy512.png`
- `runs/sample-tests/A001C006_denoised512.png`

Optional color context (bilinear demosaic, not the denoiser output):

```bash
python -m bayer_rgb.preview runs/sample-tests/A001C006_denoised512.npy \
  --tile-size 512 --center \
  --write-rgb runs/sample-tests/A001C006_denoised512_rgb.png
```

## Denoise options (CNN vs classical)

| In repo today | Notes |
|---------------|--------|
| **`BayerDenoiseNet`** | `python -m bayer_rgb.infer` on packed Bayer; use **crops** for large plates |
| **`preview`** | QA only — not a denoiser |

**Gaussian / median:** not a built-in CLI yet. Do **not** filter the raw H×W mosaic as one image (RGGB neighbors mix). Filter **four half-res planes** (R, Gr, Gb, B), then reassemble — compare median vs Gaussian vs CNN on the **same 512 crop**.

Suggested ladder on `A001C006_center512.npy`: median 3×3/5×5 per plane → Gaussian σ≈0.5–1.0 per plane → `bayer_denoise` infer.

## Notes

| Topic | Detail |
|-------|--------|
| **Output format** | `infer` writes float32 **H×W Bayer mosaic** `.npy`, same shape as input crop |
| **Checkpoint quality** | `bayer-denoise-smoke` is 3-epoch synthetic — expect weak denoising on real sky noise |
| **Full frame** | No tiled `infer` yet; full `.npy` forward may **OOM** on MPS |
| **Next step** | Train `--model bayer_denoise` on real crops, then infer with `runs/bayer-denoise-plates/model.pt` |

See also [training user guide](../guides/training-user-guide.md) (Infer section) and [project context](../guides/project-context.md).

For **Gaussian-smoothed synthetic ground truth** (train/infer/PSNR plumbing), see [Gaussian GT pipeline test](static-noisy-singleframe-pipeline-test.md).
