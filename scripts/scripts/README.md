# Scripts (`bayer_denoiser`)

Project renamed from `bayer_rgb` → **`bayer_denoiser`** (Bayer-space denoise, packed RGGB). See [rename guide](../docs/archived/guides/rename-to-bayer-denoiser.md).

**Canonical:** `~/Projects/bayer_denoiser/scripts/` (git). **Cursor Project → Context** should match after:

```bash
cd ~/Projects/bayer_denoiser
bash scripts/sync_repo_to_context.sh
```

**Context → Mac repo:**

```bash
cd ~/Projects/bayer_denoiser
bash scripts/sync_context_to_repo.sh
```

## Sync (two scripts only)

| Script | Direction |
|--------|-----------|
| `sync_context_to_repo.sh` | Agent Store (`files/` or store root) → repo: `docs/`, `scripts/`, `internal/` → `docs/internal/` |
| `sync_repo_to_context.sh` | Repo → Agent Store: `docs/` and `scripts/` |

## Layout

| Path | Purpose |
|------|---------|
| **`sky/`** | Data prep: `.sky` → `.npy` (`convert.py`, `io.py`) — not part of `pip install -e .` |
| **`archived/`** | Retired one-offs and old smokes (see `archived/README.md`) |
| **Root `*.sh` / `*.py`** | Pipeline tests, sync, training |

## Project scripts (Context / synced)

| Script | Purpose |
|--------|---------|
| `bayer_denoise_smoke.sh` | **Deprecated** — exits 1; use `python -m bayer_denoiser.train_static_noisy_sequence` |
| `gaussian_gt_pipeline_test.sh` | Gaussian GT tile pipeline + optional full-frame GT |
| `make_full_frame_gaussian_gt.sh` | Full-plate Gaussian GT `.npy` |
| `make_bayer_gaussian_gt.py` | Per-plane Gaussian on Bayer mosaic (torch/scipy) |
| `input_and_infer_crop.py` | Crop full-frame `.npy` to match checkpoint, save crop, run denoise |
| `run_real_bayer_denoise.sh` | QA + optional train/infer for real data dirs |
| `denoise_sample_a001c006.sh` | Infer-only on A001C006 crop |
| `convert_npy_folder_to_png.py` | Batch `.npy` → PNG via preview helper |

## Repo-only scripts (may exist only on Mac)

Check with `ls scripts/` for anything not listed above. After adding on Mac, run **`sync_repo_to_context.sh`** so Context matches.
