# Testing synthetic data in `data/`

Use this before real plates land. Everything below assumes **repo root** `~/Projects/bayer_denoiser`, venv active, `pip install -e .`.

## What you have → which path

| Your `data/` layout | What to run |
|---------------------|-------------|
| **One noisy Bayer `.npy`** per scene (no clean file) | **Gaussian plumbing pipeline** — builds synthetic clean, static sequence, train, infer. [static-noisy-sequence.md](static-noisy-sequence.md) |
| **Paired clean + noisy** `.npy` (same H×W, aligned) | **Setup A** — `GT/` + `input/` at dataset root; pipeline or trainer reads directly. [static-noisy-sequence.md#real-paired-plates-setup-a](static-noisy-sequence.md) |
| **`.sky` files** | Convert first: [sky-format-conversion.md](sky-format-conversion.md) → then one of the rows above |

**On-disk contract:** float16, H×W RGGB mosaic in **[0, 1]** — [float16-pipeline-standard.md](float16-pipeline-standard.md).

## Quick check (any `.npy` plate)

```bash
python -c "
import numpy as np, sys
p = sys.argv[1]
a = np.load(p)
print(p, 'shape', a.shape, 'dtype', a.dtype, 'min/max', float(a.min()), float(a.max()))
" data/your_plate.npy
```

Even H and W are required for Bayer packing. Values should be roughly in [0, 1].

## A — Single noisy plate (synthetic Gaussian GT)

Use a **dedicated run dir** so you do not overwrite `runs/static-noisy-sequence-test/`:

```bash
RUN=runs/synthetic-myset-smoke \
CROP_Y=4000 CROP_X=6000 TILE=512 EPOCHS=100 NUM_CROPS=1 SMOKE_CROP_PNG=1 \
  bash scripts/static_noisy_sequence_pipeline_test.sh data/your_noisy.npy
```

Center smoke (lighter sequence — 512×512 tile path):

```bash
RUN=runs/synthetic-myset-center CROP_SIZE=512 EPOCHS=100 \
  bash scripts/static_noisy_sequence_pipeline_test.sh data/your_noisy.npy
```

Outputs: `runs/<RUN>/train/static_sequence/`, `model.pt`, `compare/` PNGs and `denoised_infer.npy`.

**Meaning:** high PSNR vs Gaussian GT checks **wiring only**, not real sensor noise.

## B — Paired synthetic (clean + noisy you generated)

### Layout: `data/sythatic_sequence/GT` + `input/`

Typical meaning: **one clean plate** under `GT/`, **many noisy frames** under `input/` (static sequence). Paths are relative to repo root (`~/Projects/bayer_denoiser`).

**1. Inspect**

```bash
ls -la data/sythatic_sequence/GT data/sythatic_sequence/input
python -c "
from pathlib import Path
import numpy as np
for d in ('data/sythatic_sequence/GT', 'data/sythatic_sequence/input'):
    print('==', d)
    for p in sorted(Path(d).glob('*.npy'))[:5]:
        a = np.load(p)
        print(' ', p.name, a.shape, a.dtype)
    n = len(list(Path(d).glob('*.npy')))
    if n > 5: print('  ...', n, 'files total')
"
```

**2. Smoke pipeline** (no `train/static_sequence/` assemble — reads `GT/` + `input/` directly):

```bash
DATA_ROOT=data/sythatic_sequence \
RUN=runs/synthetic-sequence \
CROP_SIZE=512 CROP_Y=4000 CROP_X=6000 EPOCHS=100 NUM_CROPS=1 SMOKE_CROP_PNG=1 \
  bash scripts/static_noisy_sequence_pipeline_test.sh
```

Outputs: `runs/synthetic-sequence/model.pt`, `compare/denoised_infer.npy`, `compare/*.png`. Adjust **`CROP_Y` / `CROP_X`** to a valid even window on your plate.

**3. Shape check** (optional — trainer errors on mismatch):

```bash
python -c "
import numpy as np
from pathlib import Path
root = Path('data/sythatic_sequence')
gt = sorted((root / 'GT').glob('*.npy'))
inp = sorted((root / 'input').glob('*.npy'))
c = np.load(gt[0])
for p in inp:
    a = np.load(p)
    assert a.shape == c.shape, (p.name, a.shape, c.shape)
print('ok', c.shape, len(inp), 'frames')
"
```

**4. Manual train + infer** (same data root, no shell pipeline):

```bash
RUN=runs/synthetic-sequence
mkdir -p "$RUN/compare"
python -m bayer_denoiser.train_static_noisy_sequence \
  --data-root data/sythatic_sequence \
  --output-dir "$RUN" \
  --crop-size 512 --crop-y 4000 --crop-x 6000 \
  --epochs 100 --device mps

NOISY=$(ls data/sythatic_sequence/input/*.npy | sort | head -1)
python -m bayer_denoiser.infer \
  --checkpoint "$RUN/model.pt" \
  --input "$NOISY" \
  --output "$RUN/compare/denoised_infer.npy" \
  --tiled --tile-size 512 --overlap 64

python -m bayer_denoiser.preview "$RUN/compare/denoised_infer.npy" \
  --write-mosaic "$RUN/compare/denoised.png"
```

If `GT/` has **one noisy/clean pair per file** (same names in both folders), say so — pairing is per-filename, not one-clean-many-frames.

### Simple pair (one clean, one noisy)

```text
data/synthetic_clip01/
  clean.npy
  noisy.npy
```

Assemble sequence (no `make_static_noisy_sequence.py` — that adds new noise):

```bash
RUN=runs/synthetic-clip01
mkdir -p "$RUN/train/static_sequence"
cp data/synthetic_clip01/clean.npy "$RUN/train/static_sequence/clean_mosaic.npy"
cp data/synthetic_clip01/noisy.npy "$RUN/train/static_sequence/frame_0000.npy"
```

Train + infer (smoke):

```bash
python -m bayer_denoiser.train_static_noisy_sequence \
  --sequence-dir "$RUN/train/static_sequence" \
  --output-dir "$RUN" \
  --crop-size 512 --crop-y 4000 --crop-x 6000 \
  --epochs 100 --frames 1 --device mps

python -m bayer_denoiser.infer \
  --checkpoint "$RUN/model.pt" \
  --input data/synthetic_clip01/noisy.npy \
  --output "$RUN/compare/denoised_infer.npy" \
  --tiled --tile-size 512 --overlap 64
```

PSNR vs your **synthetic clean** is meaningful here; use it to validate the model on data that looks like production shape.

## Before real data

- Keep **one run directory per clip or experiment** (`RUN=...`).
- Prefer **full-plate** `static_sequence/` + train-time crops (same as coordinate smoke).
- Real ingest will be **`.sky` → `.npy`** then the same **Setup A** folder layout; swap files under `train/static_sequence/` and keep the same train/infer CLIs.
