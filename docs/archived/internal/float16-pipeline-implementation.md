---
cursor:
  subagentId: "bc-a2cb8c9e-e272-587a-82b1-a20c4a7b490b"
---

# Float16 pipeline — Mac repo implementation checklist

**Repo:** `~/Projects/bayer_rgb`  
**Policy:** [float16-pipeline-standard.md](../docs/float16-pipeline-standard.md)  
**VM note:** No `bayer_rgb` checkout on cloud worker at implementation time — apply on Mac (or sync worker) using this list.

## Summary rule

| Layer | Dtype |
|-------|--------|
| `.npy` mosaic read/write | **`np.float16`** |
| `np.save` after sky / infer / smoke | **`astype(np.float16)`** |
| Pack/unpack numpy buffers | **`float32`** workspace OK; **save** **`float16`** |
| `torch` batch on device | **`float32`** default (MPS-stable) |
| PSNR / MSE | **`float32`** only |

---

## File-by-file

### `src/sky/convert.py` (and `src/sky/io.py` if save is centralized)

**Find:** final mosaic array before `np.save`, often `astype(np.float32)` or default float64 from division.

**Change:**

```python
mosaic = np.clip(mosaic, 0.0, 1.0).astype(np.float16)
np.save(output_path, mosaic)
```

**Verify:** `python -m bayer_rgb.sky --input clip.sky --output /tmp/t.npy` → `np.load("/tmp/t.npy").dtype == np.float16`.

---

### `src/bayer_rgb/load_bayer.py` (or `bayer_rgb/load_bayer.py` if flat layout)

**Find:** `.npy` branch: `np.load(...).astype(np.float32)` or unconditional float32.

**Change:**

```python
mosaic = np.load(path)
if mosaic.dtype not in (np.float16, np.float32):
    mosaic = mosaic.astype(np.float32)  # exotic dtypes only
# Return mosaic as loaded; callers promote to float32 for torch
return mosaic
```

**Optional:** `warnings.warn` once if `dtype == np.float32` (legacy file).

**RAW path:** unchanged decode; if mosaic is built as float32 from buffer, **`astype(np.float16)`** only when writing `.npy` exports — in-memory train/infer can stay float32 after decode.

---

### `src/bayer_rgb/infer.py`

**Find:** load path (uses `load_bayer`), pack to tensor, save output mosaic.

**Load → tensor:**

```python
mosaic = load_bayer(...)  # float16 or float32
packed = mosaic_to_packed(mosaic.astype(np.float32))  # or existing pack helper
x = torch.from_numpy(packed).unsqueeze(0).to(device=device, dtype=torch.float32)
```

**Save `.npy`:**

```python
out_mosaic = packed_to_bayer_mosaic(out_packed.cpu().numpy())
np.save(output_path, np.clip(out_mosaic, 0.0, 1.0).astype(np.float16))
```

**Find/replace candidates:** `astype(np.float32)` on **output save only** → `float16`; keep float32 for forward.

---

### `src/bayer_rgb/train.py`

**Find:** synthetic dataset `numpy`/`torch` creation; paired dataset mosaic load; smoke fixture saves if any.

**Changes:**

- After loading mosaic for batching: `torch.from_numpy(...).float()` (float32) — **no change** if already `.float()`.
- If train writes `.npy` fixtures (smoke): **`astype(np.float16)`**.
- PSNR helper in train loop: ensure **`float32`** reduction (likely already).

**CLI:** `--raw-dtype float16` already documented — confirm default in argparse matches production plates.

---

### `src/bayer_rgb/dataset.py` (synthetic + paired)

**Find:** `astype(np.float32)` on every mosaic read.

**Change:** load as-is; **`mosaic.astype(np.float32, copy=False)`** only when converting to torch (or use `torch.from_numpy(mosaic).float()`).

---

### `src/bayer_rgb/preview.py`

**Find:** `np.load` + stats / PNG scaling.

**Change:**

```python
mosaic = np.load(path)
work = mosaic.astype(np.float32) if mosaic.dtype == np.float16 else mosaic.astype(np.float32)
# use `work` for min/max/mean and 8-bit PNG scale
```

(Preview never needs to rewrite `.npy` unless adding a `--rewrite-float16` flag — **out of scope**.)

---

### `src/bayer_rgb/bayer.py` (pack/unpack helpers)

Usually fine as float32 math. Document that callers pass **`float32`** workspace; disk remains **`float16`**.

---

### `scripts/bayer_denoise_smoke.sh`

**Find:** inline Python that `np.save`s `noisy_mosaic.npy`, `clean_mosaic.npy`.

**Change:** `.astype(np.float16)` on each save.

---

### Context scripts (sync to `~/Projects/bayer_rgb/scripts/`)

Already updated in Project store; re-sync after pull:

| Script | Disk I/O change |
|--------|------------------|
| `make_bayer_gaussian_gt.py` | `gaussian_clean_mosaic` return **`float16`**; docstring says float16 |
| `overfit_bayer_denoise_pair.py` | load **`float16`**; `np.save(den, ... float16)` |
| `gaussian_gt_pipeline_test.sh`, `run_real_bayer_denoise.sh`, `denoise_sample_a001c006.sh` | crop save **`astype(np.float16)`** |

---

### Docs in repo (mirror Context)

- `docs/MANUAL.md` § infer output dtype → **float16**
- `docs/BAYER_DENOISE.md` if it mentions float32 mosaic files
- `docs/RAW_SPEC.md` — sidecar `dtype: float16` already aligned with train flags

---

## Verification commands (Mac)

```bash
cd ~/Projects/bayer_rgb && source .venv/bin/activate
pip install -e .

# Sky
python -m bayer_rgb.sky --input data/samples/clip.sky --output /tmp/f16.npy
python -c "import numpy as np; a=np.load('/tmp/f16.npy'); assert a.dtype==np.float16; print(a.shape, a.min(), a.max())"

# Smoke
./scripts/bayer_denoise_smoke.sh
python -c "import numpy as np; from pathlib import Path; p=Path('runs/bayer-denoise-smoke/denoised_mosaic.npy'); print(p, np.load(p).dtype)"

# Infer round-trip
python -m bayer_rgb.infer --checkpoint runs/bayer-denoise-smoke/model.pt \
  --input runs/bayer-denoise-smoke/noisy_mosaic.npy \
  --output /tmp/out_f16.npy
python -c "import numpy as np; print(np.load('/tmp/out_f16.npy').dtype)"
```

## Patch template (if file layout matches audit-era flat `bayer_rgb/`)

If `infer.py` still contains:

```python
np.save(args.output, out_mosaic.astype(np.float32))
```

replace with:

```python
np.save(args.output, np.clip(out_mosaic, 0.0, 1.0).astype(np.float16))
```

If `convert.py` contains:

```python
np.save(output, mosaic.astype(np.float32))
```

replace with:

```python
np.save(output, np.clip(mosaic, 0.0, 1.0).astype(np.float16))
```

## Status

| Item | Cloud VM | Mac repo |
|------|----------|----------|
| Policy + user docs | Done (Context store) | Sync via `sync_context_to_repo.sh` + manual policy doc copy |
| `sky` / `infer` code | **No checkout** | **Pending** — apply checklist above |
| Context scripts | Updated in store | Re-sync scripts |
