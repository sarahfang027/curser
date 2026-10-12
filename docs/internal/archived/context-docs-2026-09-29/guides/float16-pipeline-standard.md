# Float16 pipeline standard (Bayer denoise)

**Effective:** 2026-09-29  
**Scope:** `~/Projects/bayer_rgb` — ingest, training, infer, and on-disk mosaics for **`--model bayer_denoise`** and shared Bayer I/O (`preview`, Gaussian-GT plumbing).

## Policy (one sentence)

**Store and move Bayer mosaics as `float16` in [0, 1]; run the U-Net in `float32` on device when needed; cast to `float32` only for PSNR, loss aggregation, and other numerically sensitive metrics.**

## On-disk mosaic contract

| Artifact | Dtype | Range | Shape |
|----------|-------|-------|--------|
| `.npy` Bayer mosaic | **`np.float16`** | `[0, 1]` | **H×W** (one scalar per Bayer site, RGGB layout) |
| Packed RGGB (in memory / batches) | **`torch.float16`** or **`float32`** on device (see MPS) | `[0, 1]` | `(4, H/2, W/2)` |
| Checkpoints (`model.pt`) | unchanged (weights stay **`float32`**) | — | — |

**Not** a 4-plane `.npy` on disk — packing happens inside `load_bayer`, `train`, and `infer`.

### Why float16 on disk

- Full plates (~17K×18K) are **~600 MiB** as float32 vs **~300 MiB** as float16 per mosaic.
- Values are normalized to **[0, 1]** after black/white scaling; float16 mantissa is enough for 12-bit sensor data in this pipeline.
- Inference and training **read the same files**; one dtype avoids silent double memory use.

## Stage-by-stage

### `.sky` → `.npy` (`python -m bayer_rgb.sky`)

- After decode, shift, and normalize to `[0, 1]`, **`np.save(..., mosaic.astype(np.float16))`**.
- Sidecar JSON is unchanged (geometry, bit depth, CFA).
- See [sky-format-conversion.md](../formats/sky-format-conversion.md).

### `load_bayer` / `preview`

- **`.npy`:** load with `np.load`; if dtype is `float32` (legacy), accept and optionally warn once; normalize path should **`astype(np.float16, copy=False)`** only when writing, not when feeding the network.
- **For display / stats:** promote to **`float32`** inside the module if a computation needs it.
- **`.raw`:** sidecar **`dtype`** may still say `float16` for production plates; decode to mosaic then same rules as `.npy`.

### Training (`python -m bayer_rgb.train`)

- Mosaic tensors from disk: **`float16`** numpy → **`torch.float32`** on device for forward/backward (default on MPS/CUDA for stability).
- Optional later: true **`torch.float16`** training with autocast — **not required** for this standard; disk I/O is the main win.
- **`--raw-dtype float16`** on CLI matches production `.raw` sidecars (already used in runbooks).
- Logged **PSNR**: compute in **`float32`** (see below).

### Infer (`python -m bayer_rgb.infer`)

- **Input:** `.npy` **`float16`** or legacy **`float32`** H×W `[0, 1]`.
- **Forward:** packed batch **`float32`** on device (same as train).
- **Output:** **`np.save(..., mosaic.astype(np.float16))`** for `.npy` outputs.

### Metrics and PSNR

Always reduce in **`float32`**:

```python
mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
```

Do **not** report PSNR from raw `float16` subtraction — rounding can bias dB at high PSNR.

### Context / helper scripts

Scripts synced from Project Context (`make_bayer_gaussian_gt.py`, `overfit_bayer_denoise_pair.py`, crop snippets in runbooks) should **write `float16`** mosaics and **load** with `.astype(np.float16)` or promote to `float32` only for math.

## Apple Silicon (MPS) notes

| Topic | Guidance |
|-------|----------|
| **Default device** | `cuda` → **`mps`** → `cpu` via `default_device_string()` |
| **Model weights** | Keep **`float32`**; MPS mixed float16 training is optional, not default |
| **Activations** | Safe pattern: **`float16` disk → `float32` tensor → forward → `float16` save** |
| **OOM** | Halving mosaic buffer size on load helps; cropping still required for full 18K infer |
| **Debugging** | `--device cpu` if NaNs or suspicious output; compare in **`float32`** for one crop |

## Migrating existing float32 `.npy` files

No re-decode from `.sky` required if values are already correct:

```bash
cd ~/Projects/bayer_rgb
source .venv/bin/activate
python - <<'PY'
import sys
import numpy as np
from pathlib import Path

for path in map(Path, sys.argv[1:]):
    a = np.load(path)
    if a.dtype == np.float16:
        print(f"skip (already float16): {path}")
        continue
    np.save(path, np.clip(a, 0.0, 1.0).astype(np.float16))
    print(f"converted: {path} {a.dtype} -> float16 shape={a.shape}")
PY
data/samples/*.npy runs/**/*.npy
```

- **Backup** large catalogs before bulk conversion.
- **Git:** binary `.npy` usually gitignored; migration is on-disk / volume data.
- After conversion, re-run **`preview`** on one crop to confirm stats and PNG QA.

## Related docs

- [Sky → `.npy`](../formats/sky-format-conversion.md)
- [Training user guide](training-user-guide.md)
- [Real data runbook](real-data-runbook.md)
- [Project context](project-context.md)
- Mac implementation checklist: [float16-pipeline-implementation](../../../float16-pipeline-implementation.md)

## Coordinator checklist

When reporting pipeline status, include:

- Mosaic dtype on disk: **`float16`** / legacy **`float32`**
- Whether Mac repo patches from the implementation checklist are applied
- Whether PSNR scripts use **`float32`** reduction (required)
