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
| `sync_context_to_repo.sh` | Context → repo: `docs/`, `scripts/`, **`src/`**, `internal/` → `docs/internal/` |
| `sync_repo_to_context.sh` | Repo → Context: `docs/`, `scripts/`, **`src/`** |

## Layout

| Path | Purpose |
|------|---------|
| **`sky/`** | Data prep: `.sky` → `.npy` (`convert.py`, `io.py`) — not part of `pip install -e .` |
| **`archived/`** | Retired one-offs and old smokes (see `archived/README.md`) |
| **Root `*.sh` / `*.py`** | Pipeline tests, sync, training |

## Project scripts (Context / synced)

| Script | Purpose |
|--------|---------|
| `static_noisy_singleframe_pipeline_test.sh` | Gaussian GT tile pipeline + optional full-frame GT |
| `make_full_frame_gaussian_gt.sh` | Full-plate Gaussian GT `.npy` |
| `make_bayer_gaussian_gt.py` | Per-plane Gaussian on Bayer mosaic (torch/scipy) |
| `input_and_infer_crop.py` | Full-frame `.npy` → crop per checkpoint geometry → denoised patch `.npy` |
| `convert_npy_folder_to_png.py` | Batch `.npy` → PNG via preview helper |

## Repo-only scripts (may exist only on Mac)

Check with `ls scripts/` for anything not listed above. After adding on Mac, run **`sync_repo_to_context.sh`** so Context matches.
