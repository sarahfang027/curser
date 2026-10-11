---
cursor:
  subagentId: "bc-079dfe4b-9849-58f5-abbf-34152114a440"
---

# Scripts audit — `bayer_denoiser` Context (2026-10-05)

**Scope:** `/cursor/stores/…/scripts/` vs `/cursor/stores/…/files/scripts/`  
**README:** `scripts/README.md` (identical in both trees)

**Grep** (`deprecated`, `bayer_rgb`, `overfit`, `sync_project`, `TODO`, `obsolete`) under `scripts/`:

| Term | In `scripts/` |
|------|----------------|
| `bayer_rgb` | README rename note; `BAYER_RGB_ROOT` fallback + `~/Projects/bayer_rgb` hint in `sync_context_to_repo.sh`; same env fallback in Mac-oriented `*.sh` |
| `overfit` | **No matches** (script removed; docs still cite `overfit_bayer_denoise_pair.py`) |
| `sync_project` | **No matches** |
| `deprecated` / `obsolete` / `TODO` | **No matches** in script bodies |

**Recursive inventory** (`.sh` / `.py`; exclude `__pycache__`):

| Path | Purpose (from header/comments) |
|------|--------------------------------|
| `sync_context_to_repo.sh` | Context → Mac repo: `docs/`, `scripts/`, `internal/` → `docs/internal/` |
| `sync_repo_to_context.sh` | Mac repo → Context: `docs/` + `scripts/` |
| `reorganize_docs_folders.sh` | One-time idempotent `git mv` of docs into `guides/`, `formats/`, `operations/` |
| `bayer_denoise_smoke.sh` | Synthetic **bayer_denoise** train → infer smoke on `.npy` |
| `pipeline_smoke.sh` | Synthetic **GCP** train → infer smoke → `out_rgb.png` |
| `gaussian_gt_pipeline_test.sh` | Sky sample → Gaussian GT tile (+ optional full) → **`python -m bayer_denoiser.core.train_static_noisy_sequence`** → official infer + previews |
| `make_full_frame_gaussian_gt.sh` | Full-plate Gaussian GT `.npy` only |
| `make_bayer_gaussian_gt.py` | Per-plane Gaussian on packed RGGB |
| `make_static_noisy_sequence.py` / `.sh` | One clean mosaic + N independent noisy `frame_*.npy` |
| `python -m bayer_denoiser.core.train_static_noisy_sequence` / `.sh` | Train `BayerDenoiseNet` on static sequence (replaces tile **overfit** path in `gaussian_gt_pipeline_test.sh`) |
| `input_and_infer_crop.py` | Crop full-frame mosaic to checkpoint geometry, infer, save `.npy` |
| `denoise_sample_a001c006.sh` | Infer-only center 512 on A001C006 sample |
| `run_real_bayer_denoise.sh` | QA / optional legacy train / infer template for real plates |
| `convert_npy_folder_to_png.py` | Batch `bayer_denoiser.preview` PNGs for a folder of `.npy` |
| `sky/convert.py` | CLI: `.sky` → float16 Bayer `.npy` (local `io.py` via importlib) |
| `sky/io.py` | csky decode / deinterleave |
| `sky/__init__.py` | Re-exports from **`bayer_denoiser.sky.io`** (package path) |
| `sky/__main__.py` | Delegates to **`bayer_denoiser.sky.convert`** (package path) |

**Absent (docs still reference):** `overfit_bayer_denoise_pair.py`, old `sync_project*` names.

---

## 1. Remove or archive

| Item | Reason |
|------|--------|
| **`reorganize_docs_folders.sh`** | Documented one-time migration; `notes.md` marks doc categorization done. Safe to **archive** (idempotent skips if re-run). |
| **`scripts/__pycache__/*.pyc`** | Accidental artifacts; not source; should not live in Context or sync to repo. |
| **`pipeline_smoke.sh`** (optional) | **GCP** smoke; `notes.md` says drop `pipeline-smoke` **runs** and focus is **bayer_denoise**. Keep only if Sarah still tests `--model gcp`; otherwise archive beside deferred GCP work. |
| **`sky/__init__.py`**, **`sky/__main__.py`** | Stale package shims (`bayer_denoiser.sky.*`) while canonical implementation is **`scripts/sky/convert.py`** + **`io.py`** (see `internal/sky-scripts-migration.md`). Misleading if copied to repo; **`python scripts/sky/convert.py`** is the supported entry. |

---

## 2. Keep but update

| Item | What’s wrong |
|------|----------------|
| **`files/scripts/sync_repo_to_context.sh`** | **Empty file (0 bytes).** Root `scripts/sync_repo_to_context.sh` is complete (~1.6 KiB). Mac **repo → Context** sync is broken until restored. |
| **`files/scripts/` vs root `scripts/`** | Six paths **differ**; root is **newer** for: `sync_context_to_repo.sh` (git-root default, zsh `#` arg fix), `train_static_noisy_sequence.{py,sh}` (checkpoint writes both `state_dict` and `model_state`), `input_and_infer_crop.py`, `convert_npy_folder_to_png.py`. **Re-mirror root → `files/scripts/`** before Sarah runs sync on Mac. |
| **`files/scripts/`** | Missing entire **`sky/`** tree (only present under store root `scripts/`). Context→repo sync from `files/` will **not** ship sky converters until copied. |
| **`scripts/README.md`** | **`pipeline_smoke.sh`** listed under “repo-only” but present in Context; **`bayer_denoise_smoke.sh`** labeled “repo git — if present” but also in Context. Clarify both are mirrored after sync. |
| **Project docs** (not script files) | Still instruct `overfit_bayer_denoise_pair.py` and Gaussian “overfit” wording; pipeline script already uses **`python -m bayer_denoiser.core.train_static_noisy_sequence`**. Update runbooks when cleaning scripts. |

---

## 3. Current / keep as-is

| Script | Notes |
|--------|--------|
| **`sync_context_to_repo.sh`** / **`sync_repo_to_context.sh`** (root copies) | Current two-way sync pair per README; no `sync_project*`. |
| **`bayer_denoise_smoke.sh`** | Primary package smoke for `--model bayer_denoise`. |
| **`gaussian_gt_pipeline_test.sh`**, **`make_bayer_gaussian_gt.py`**, **`make_full_frame_gaussian_gt.sh`** | Gaussian GT plumbing; train step aligned with static-sequence trainer. |
| **`make_static_noisy_sequence.*`**, **`train_static_noisy_sequence.*`** | Current train path for pseudo-GT / static-sequence work (`BAYER_RGB_ROOT` fallback is intentional). |
| **`input_and_infer_crop.py`**, **`denoise_sample_a001c006.sh`**, **`run_real_bayer_denoise.sh`** | Real-data / infer QA templates. |
| **`convert_npy_folder_to_png.py`** | Utility preview export. |
| **`sky/convert.py`**, **`sky/io.py`** | Active sky ingest (float16 `.npy`); matches `notes.md` sky training path. |

---

## 4. `scripts/` vs `files/scripts/` mismatch summary

| | Root `scripts/` | `files/scripts/` |
|--|-----------------|------------------|
| **Extra files** | `sky/*` (4 files) | — |
| **Byte-identical** | 11 paths (smokes, Gaussian helpers, README, most shells) | same |
| **Differ** | Newer sync + train + infer + convert PNG | Older or **empty** `sync_repo_to_context.sh` |
| **Mac sync source** | Used if store layout is flat (`STORE=$STORE_ROOT`) | **Default** when `files/` exists (`sync_context_to_repo.sh` sets `STORE=$STORE_ROOT/files`) |

**Operational recommendation:** Treat **store root `scripts/`** as editor canonical until `files/scripts/` is refreshed; run **`sync_repo_to_context.sh`** from repo after fix, or copy root → `files/scripts/` including `sky/` and a non-empty `sync_repo_to_context.sh`.
