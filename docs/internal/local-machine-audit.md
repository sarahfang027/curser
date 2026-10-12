---
cursor:
  subagentId: "bc-658a3721-f35f-564a-ad4d-5eb2f16171ee"
---

# Local machine audit — `bayer_rgb`

**Audited:** 2026-09-28 (America/Los_Angeles)  
**Checkout:** `/Users/Sarah.Fang/Projects/bayer_rgb`  
**Host:** macOS 26.6, **arm64** (Apple Silicon). No external volumes mounted (`/Volumes` → Macintosh HD only).

## Environment

| Item | Finding |
|------|---------|
| **Documentation** | No root `README.md`. Primary guide: `docs/MANUAL.md` (+ `docs/RAW_SPEC.md`, `docs/GCP_STILL.md`). |
| **Python env** | **`.venv`** present (not conda). Activate: `source .venv/bin/activate`. |
| **Python version** | **3.9.6** (system/Xcode-linked interpreter used by the venv). |
| **Dependencies** | `requirements.txt`: torch≥2, torchvision, numpy, Pillow, tqdm. Optional EXR: `requirements-raw.txt` → **imageio** (not installed in venv at audit time). |
| **Installed (venv)** | torch **2.8.0**, torchvision **0.23.0**, numpy **2.0.2**, Pillow **11.3.0**, tqdm **4.70.1**. |
| **Package layout** | Flat package `bayer_rgb/` at repo root; **no** `setup.py` / `pyproject.toml`. CLIs run as `python -m bayer_rgb.<module>` **from repo root** (imports rely on cwd). |
| **GPU / accelerators** | **CUDA:** unavailable. **MPS:** available (`torch.backends.mps.is_available()` → true). |
| **Device defaults in code** | `train.py` and `infer.py` default to `"cuda" if torch.cuda.is_available() else "cpu"` — on this Mac that means **CPU unless `--device mps` is passed**. |
| **Git** | Repo initialized; **no commits on `main` yet**. `.gitignore` excludes `.venv/`, `runs/`, `*.pt`. |
| **Local artifacts** | `runs/` is **gitignored**. Typical dirs: `bayer-denoise-smoke/`, `gaussian-gt-test/`; optional `crop_test/`, `crop_visual_check/`. Inventory: [runs-folder-inventory.md](runs-folder-inventory.md) (under `docs/internal/` on Mac). |

### Dataset paths

| Path | Status |
|------|--------|
| `data/bayer/`, `data/rgb/` (documented in MANUAL) | **Missing** — no `data/` directory in checkout. |
| Production plate `shot_0100.raw` (~605 MiB, 17592×18024) | **Not found** under home search; no `/Volumes/plates`-style mount. |
| Test RAW in repo | Optional `runs/crop_test/bayer/shot_0100.raw` (8 KiB, 64×64) if kept — sidecar may still describe full-plate geometry, so **`validate_raw` / strict load can fail** unless sidecar matches file size or `--no-strict` is used where supported. |
| Sidecar helper | `python -m bayer_rgb.make_sidecar`; see `docs/archived/formats/RAW_SPEC.md` |
| Pair manifest | Optional `pairs.csv` with absolute paths (MANUAL example uses `/Volumes/plates/...`). |

## How to run

**Entrypoints** (all `cd /Users/Sarah.Fang/Projects/bayer_rgb && source .venv/bin/activate`):

| Purpose | Module |
|---------|--------|
| Train | `python -m bayer_rgb.train` |
| Infer | `python -m bayer_rgb.infer` |
| RAW validate / sidecar | `validate_raw`, `make_sidecar` |
| Pair QA | `validate_pairs`, `visualize_crops` |
| Decode preview | `preview` |
| Test Bayer export | `export_bayer` |

### Minimal smoke test (no real plates) — **verified on this machine**

1. Ensure venv + base deps (already satisfied in existing `.venv`):

   ```bash
   cd /Users/Sarah.Fang/Projects/bayer_rgb
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Synthetic train** (CPU default; add `--device mps` for Apple GPU):

   ```bash
   python -m bayer_rgb.train \
     --model gcp \
     --epochs 1 \
     --batch-size 2 \
     --image-size 64 \
     --synthetic-length 8 \
     --output-dir runs/local-smoke
   ```

   Audit run completed successfully (~1.6 s/epoch on CPU for this config); writes `runs/local-smoke/model.pt`.

3. **Infer** on a small `.npy` mosaic (bypasses strict `.raw` contract):

   ```bash
   python - <<'PY'
   import numpy as np, torch
   from bayer_rgb.bayer import rgb_to_bayer
   np.save("/tmp/audit_mosaic.npy", rgb_to_bayer(torch.rand(1,3,64,64)).squeeze().numpy())
   PY
   python -m bayer_rgb.infer \
     --checkpoint runs/local-smoke/model.pt \
     --input /tmp/audit_mosaic.npy \
     --output /tmp/local_smoke_rgb.png \
     --device mps
   ```

### Production-style path (when plates exist)

1. `pip install -r requirements-raw.txt` for `.exr` RGB targets.  
2. Lay out `data/bayer/{stem}.raw` + `{stem}.raw.json` and `data/rgb/{stem}.exr` (same stem, same H×W).  
3. `validate_raw` → `validate_pairs` → `visualize_crops` → `train` with `--bayer-dir` / `--rgb-dir`, `--image-size 256`, `--batch-size 1`.  
4. `infer --input ...shot.raw` (full frame; no tiling flag — large plates may OOM).

## Gaps

1. **No training data in repo** — cannot run real-plate training or crop QA without user-supplied paths or a `data/` tree / CSV manifest.  
2. **`imageio` not installed** — EXR RGB loading for paired training and `visualize_crops` will fail until `pip install -r requirements-raw.txt`.  
3. **MPS not auto-selected** — defaults to CPU; user must pass `--device mps` for faster train/infer on Apple Silicon.  
4. **Strict RAW contract** — default geometry 17592×18024; tiny fixtures under `runs/` do not pass strict validation/preview/infer without matching sidecars or non-`.raw` inputs (e.g. `.npy`).  
5. **No packaged install** — easy to break imports if cwd is not project root; no editable `pip install -e .`.  
6. **Full-frame infer at production resolution** — documented OOM risk; tiled infer not implemented.  
7. **Python 3.9** — works today with torch 2.8; team may want 3.10+ for long-term support (optional, not blocking smoke test).

## Recommended next steps

1. **Confirm plate location** — mount or copy `shot_*.raw` + sidecars and matching EXR/PNG into `data/bayer` and `data/rgb`, or author `pairs.csv` with absolute paths.  
2. **One-time env completion:** `pip install -r requirements-raw.txt`.  
3. **Run synthetic smoke** (commands above) to confirm env; use **`--device mps`** for training once stable.  
4. **Ingest QA** on real files: `validate_raw`, `validate_pairs`, then `visualize_crops` before a long `train` job.  
5. **Optional ergonomics** (code/docs, not required for smoke): auto-pick `mps` when CUDA absent; root `README` pointing to `docs/MANUAL.md`; `pyproject.toml` for editable install.

**Audit verification performed:** synthetic `train` (1 epoch) and `infer` with `--device mps` on `/tmp/audit_mosaic.npy` both succeeded during this audit.

**Copied into Project store:** 2026-09-28 by [Map agents to bayer_rgb](https://cursor.com/agents/bc-fe6e7553-5dfb-59cc-a9ae-0341e87ddbbe) from agent store `bc-658a3721-f35f-564a-ad4d-5eb2f16171ee`.
