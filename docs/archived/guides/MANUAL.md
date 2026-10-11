# Bayer RGB / GCP-Net — User Manual

End-to-end guide for training a **still-image** demosaic model on **headerless Bayer `.raw`** plates with paired **RGB references**, using a **still-adapted GCP-Net** (Green Channel Prior).

**Project root:** `/Users/Sarah.Fang/Projects/bayer_denoiser`

**Related docs:**

- [RAW_SPEC.md](../formats/RAW_SPEC.md) — ingest contract (geometry, sidecar, byte size)
- [GCP_STILL.md](GCP_STILL.md) — model architecture notes
- [PIPELINE.md](PIPELINE.md) — synthetic train → infer (no production plates)

---

## 1. What this system does

```text
  shot_0100.raw + shot_0100.raw.json     shot_0100.exr (or .png)
           │                                      │
           └────────── paired by stem ────────────┘
                              │
                    validate → decode → auto-crop
                              │
                    GCPStillNet (train / infer)
                              │
                         RGB output
```

| Stage | Purpose |
|--------|---------|
| **Ingest** | Fixed-format `.raw` + JSON sidecar |
| **Decode** | One path: mosaic **H×W [0,1]** → pack **(4, H/2, W/2)** |
| **Train** | Random crops from **full frames** (no manual cropping) |
| **Check** | Visual crop alignment before long training |
| **Infer** | Bayer → RGB PNG |

Default model: **`gcp`** (GCPStillNet). Optional: **`demosaic`** (small baseline net).

---

## 2. Setup

```bash
cd /Users/Sarah.Fang/Projects/bayer_denoiser
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# EXR RGB + optional extras
pip install -r requirements-raw.txt
```

Activate the venv for every command below:

```bash
source .venv/bin/activate
```

---

## 3. Data layout

### 3.1 Bayer (your plates)

| Item | Value |
|------|--------|
| Geometry | **17592 × 18024** (W × H), even |
| File | Headerless **`shot_0100.raw`** |
| Storage | Little-endian **float16**, row-major |
| CFA | **RGGB** |
| Sidecar | **`shot_0100.raw.json`** next to the `.raw` |
| Expected size | **634,156,416 bytes** (~604.8 MiB) |

Naming stem: `shot_0100`, `plate_a001` — pattern `[A-Za-z][A-Za-z0-9_]*` (for future EXR pairing).

Example sidecar (linear 0–1 half-float):

```json
{
  "width": 17592,
  "height": 18024,
  "dtype": "float16",
  "black_level": 0.0,
  "white_level": 1.0,
  "endian": "little",
  "cfa_pattern": "RGGB",
  "row_major": true,
  "spec_version": "1"
}
```

Create a sidecar: `python -m bayer_denoiser.archived.qa.make_sidecar data/bayer/shot_0100.raw` — field definitions in [RAW_SPEC.md](../formats/RAW_SPEC.md).

### 3.2 RGB reference (training target)

Same **stem**, same **pixel size** as the Bayer mosaic:

```text
data/
  bayer/
    shot_0100.raw
    shot_0100.raw.json
  rgb/
    shot_0100.exr          # preferred at full resolution
```

Supported RGB: `.exr`, `.png`, `.jpg`, `.tif`, … (see `validate_pairs`).

### 3.3 Optional CSV manifest

`pairs.csv`:

```csv
bayer,rgb
/Volumes/plates/bayer/shot_0100.raw,/Volumes/plates/rgb/shot_0100.exr
```

---

## 4. Ingest workflow (before training)

### Step A — Write sidecar

```bash
python -m bayer_denoiser.archived.qa.make_sidecar /path/to/data/bayer/shot_0100.raw
```

Defaults match project geometry (17592×18024 float16). Override with `--width`, `--height`, etc.

### Step B — Validate `.raw` files

Fail fast on missing JSON, wrong geometry, or **byte size mismatch**:

```bash
python -m bayer_denoiser.archived.qa.validate_raw /path/to/data/bayer/shot_0100.raw

python -m bayer_denoiser.archived.qa.validate_raw --dir /path/to/data/bayer
python -m bayer_denoiser.archived.qa.validate_raw --dir /path/to/data/bayer --recursive
```

Use `--no-strict` if sidecar geometry differs from project defaults but is self-consistent.

### Step C — Validate Bayer ↔ RGB pairs

```bash
python -m bayer_denoiser.archived.qa.validate_pairs \
  --bayer-dir /path/to/data/bayer \
  --rgb-dir /path/to/data/rgb
```

Reports `[OK]` or `[MISMATCH]` on spatial size.

---

## 5. Decode pipeline (shared by all tools)

Implemented in `src/decode.py`:

```text
.raw + .raw.json  →  validate  →  float32 mosaic (H, W) [0,1]
                               →  pack  →  (4, H/2, W/2)
```

- **Training** on `.raw`: reads **only the crop window** from disk (not the full 605 MiB plate each step).
- **Crop origins** are **even** (top/left) for RGGB phase.

Quick decode stats (no ML):

```bash
python -m bayer_denoiser.qa.preview /path/to/shot_0100.raw
python -m bayer_denoiser.qa.preview shot_0100.raw --tile-size 512 --tile-top 0 --tile-left 0
```

**Tile preview size (`preview` only):**

| Flag | Default | Notes |
|------|---------|--------|
| `--tile-size` | **`0`** | Off = decode **full** mosaic. Set an **even** edge length (e.g. **512**) to crop before pack. |
| `--tile-top` / `--tile-left` | `0` | Crop origin; prefer **even** for RGGB. |

Recommended spot-check size on large plates: **`--tile-size 512`**. Any even size works (256, 512, 1024, …).

**`infer` has no `--tile-size`.** It always loads the **full frame**. Tiled / sliding-window inference is **not implemented** yet — for memory-safe checks, use `preview --tile-size …` or export a small crop and run `infer` on that.

---

## 6. Visual check — auto crops

Confirms Bayer and RGB crops use the **same window** as training.

```bash
python -m bayer_denoiser.archived.qa.visualize_crops \
  --bayer-dir /path/to/data/bayer \
  --rgb-dir /path/to/data/rgb \
  --pair-index 0 \
  --crop-size 256 \
  --num-crops 8 \
  --seed 42 \
  --output-dir runs/crop_visual_check
```

**Outputs** under `runs/crop_visual_check/shot_0100/`:

| File | Meaning |
|------|---------|
| `crop_XX_…_strip.png` | Mosaic \| RGB ref \| bilinear demosaic \| diff |
| `crop_XX_…_rgb.png` | Reference crop |
| `crop_XX_…_bayer_mosaic.png` | Bayer crop |
| `summary.txt` | `top`, `left`, MAE / PSNR vs bilinear |

**How to read the strip:**

- **Aligned pair:** same scene in RGB and bilinear; diff shows detail mismatch, not a **shifted duplicate**.
- **Misaligned:** double edges or unrelated content in bilinear vs RGB.

Fixed window:

```bash
python -m bayer_denoiser.archived.qa.visualize_crops \
  --bayer /path/to/shot_0100.raw \
  --rgb /path/to/shot_0100.exr \
  --crop-size 512 --top 4096 --left 8192 --num-crops 1
```

Full-frame context (loads full RGB — heavy on 18K EXR):

```bash
python -m bayer_denoiser.archived.qa.visualize_crops ... --overview --overview-max 1200
```

---

## 7. Training

### 7.1 Synthetic smoke test (no real plates)

On Mac without CUDA, **`train` / `infer` default to MPS** when available; use `--device cpu` to force CPU. See also [PIPELINE.md](PIPELINE.md) and `./scripts/pipeline_smoke.sh`.

```bash
python -m bayer_denoiser.archived.core.train \
  --model gcp \
  --epochs 5 \
  --batch-size 2 \
  --image-size 128 \
  --synthetic-length 256 \
  --noise-std 0.02 \
  --lr 1e-4 \
  --output-dir runs/gcp-synthetic
```

### 7.2 Production-style — paired full-frame RAW

**You do not pre-crop files.** Set `--image-size` to patch size; the loader picks random crops.

```bash
python -m bayer_denoiser.archived.core.train \
  --model gcp \
  --nf 64 \
  --bayer-dir /path/to/data/bayer \
  --rgb-dir /path/to/data/rgb \
  --raw-dtype float16 \
  --black-level 0 \
  --white-level 1 \
  --image-size 256 \
  --crops-per-frame 16 \
  --batch-size 1 \
  --noise-std 0 \
  --epochs 30 \
  --lr 1e-4 \
  --output-dir runs/gcp-plates
```

| Flag | Role |
|------|------|
| `--image-size` | Auto-crop edge (even), e.g. 256 |
| `--crops-per-frame` | Random patches per plate per epoch |
| `--no-random-crop` | Center crop instead of random |
| `--full-frame` | No cropping (needs huge GPU RAM) |
| `--noise-std` | Mosaic noise + GCP noise map (use **0** for clean RAW) |
| `--model demosaic` | Small baseline instead of GCP |
| `--pairs-csv` | Manifest instead of dirs |

Checkpoint: `runs/.../model.pt` + `history.json`.

### 7.3 Memory (17592×18024)

- Train with **`--image-size 128–256`**, **`--batch-size 1`**, **`--crops-per-frame` 8–32**.
- Do **not** use `--full-frame` unless you have sufficient GPU/CPU RAM.

---

## 8. Inference

```bash
python -m bayer_denoiser.core.infer \
  --checkpoint runs/gcp-plates/model.pt \
  --input /path/to/data/bayer/shot_0100.raw \
  --output /tmp/shot_0100_rgb.png
```

Sidecar is read automatically. Override decode if needed:

```bash
  --black-level 0 --white-level 1 \
  --raw-dtype float16 --raw-width 17592 --raw-height 18024
```

Bilinear baseline (no learned weights):

```bash
python -m bayer_denoiser.core.infer \
  --baseline \
  --checkpoint runs/gcp-plates/model.pt \
  --input shot_0100.raw \
  --output /tmp/bilinear.png
```

**Tile / memory notes for infer:**

- **`infer` does not take a tile size** — input is always the full Bayer frame (or whatever file you pass).
- Full-frame GCP at **17592×18024** may **OOM**.
- Tiled full-plate inference is **not implemented**. For spot checks: `preview --tile-size 512` (decode only), or crop offline and `infer` that smaller file.
- Related defaults elsewhere: train `--image-size` **128**; `visualize_crops --crop-size` **256**.

---

## 9. Recommended end-to-end checklist

1. **Install** venv + requirements (+ `requirements-raw.txt` for EXR).
2. **Place** `shot_XXXX.raw` + `shot_XXXX.raw.json` and matching `shot_XXXX.exr`.
3. **`make_sidecar`** if JSON missing.
4. **`validate_raw`** on `bayer/`.
5. **`validate_pairs`** on bayer + rgb dirs.
6. **`visualize_crops`** — inspect strips; fix alignment/naming if diff looks shifted.
7. **`train`** with `--image-size`, `--crops-per-frame`, `--model gcp`.
8. **`infer`** on a test `.raw`; compare to reference EXR in Nuke / viewer.

---

## 10. Command reference

| Module | Command |
|--------|---------|
| Sidecar | `python -m bayer_denoiser.archived.qa.make_sidecar FILE.raw` |
| RAW validate | `python -m bayer_denoiser.archived.qa.validate_raw FILE.raw` |
| Pair validate | `python -m bayer_denoiser.archived.qa.validate_pairs --bayer-dir … --rgb-dir …` |
| Decode preview | `python -m bayer_denoiser.qa.preview FILE.raw` |
| Crop visual QA | `python -m bayer_denoiser.archived.qa.visualize_crops …` |
| Train | `python -m bayer_denoiser.archived.core.train …` |
| Infer | `python -m bayer_denoiser.core.infer …` |
| Test export | `python -m bayer_denoiser.archived.qa.export_bayer --rgb … --output …` |

Help on any CLI:

```bash
python -m bayer_denoiser.archived.core.train -h
python -m bayer_denoiser.archived.qa.visualize_crops -h
```

---

## 11. Troubleshooting

| Symptom | Likely cause | Action |
|---------|----------------|--------|
| `file size … != expected` | Wrong export or corrupt `.raw` | Re-export; check sidecar W×H×2 bytes |
| `geometry … != contract` | Sidecar dims ≠ 17592×18024 | Fix sidecar or use `validate_raw --no-strict` for load only |
| `missing canonical sidecar` | No `file.raw.json` | Run `make_sidecar` |
| `Size mismatch` bayer vs rgb | Different resolution or wrong pair | Same stem; same H×W |
| Visual strip shows shift | Wrong RGB or offset metadata | Re-check pairing and pipeline export |
| OOM training | Patch too large / full frame | Lower `--image-size`, `batch-size 1` |
| OOM infer | Full 18K forward; **`infer` has no tile flag** | `preview --tile-size 512` for decode QA, or infer a smaller crop file |
| EXR load error | Missing imageio | `pip install -r requirements-raw.txt` |
| Old checkpoint | Saved as `demosaic` | `infer --model demosaic` or retrain with `--model gcp` |

---

## 12. Project layout (code)

```text
src/               # import name: bayer_denoiser (see pyproject.toml package-dir)
  archived/io/raw_spec.py  # Ingest contract (archived) (single source of truth)
  raw_io.py        # Read .raw / normalize
  decode.py        # load_bayer + pack_mosaic
  sky/             # .sky (csky) conversion
  dataset.py       # Synthetic / paired RAW + auto-crop
  gcp_net.py       # GCPStillNet
  model.py         # build_model, demosaic baseline
  train.py / infer.py / preview.py / visualize_crops.py
  validate_raw.py / validate_pairs.py / make_sidecar.py
docs/
  MANUAL.md        # this file
  RAW_SPEC.md
  GCP_STILL.md
```

---

## 13. What this system is not

- Not the full **burst** GCP-Net (no multi-frame InterF / deformable alignment).
- Not a camera ISP (no per-channel black body, WB, or tone mapping in-graph).
- Not tiled **full-frame** inference at 18K (yet).

For burst video or official pretrained weights, see the [original GCP-Net repository](https://github.com/GuoShi28/GCP-Net).
