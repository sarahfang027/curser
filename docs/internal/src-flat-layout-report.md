---
cursor:
  subagentId: "bc-4a32e97e-a267-5027-8c76-e7f6d9942d0a"
---

# Flat `src/` layout (active modules)

## What changed

Active package code moved from `src/core/`, `src/io/`, `src/qa/`, `src/utilities/` to **`src/`** root. Empty subpackages removed (`__init__.py` deleted). **`src/archived/`** unchanged.

| From | To |
|------|-----|
| `core/bayer_denoise_net.py` | `bayer_denoise_net.py` |
| `core/model.py` | `model.py` |
| `core/infer.py` | `infer.py` |
| `core/train_static_noisy_sequence.py` | `train_static_noisy_sequence.py` |
| `io/bayer.py` | `bayer.py` |
| `qa/preview.py` | `preview.py` |
| `utilities/device_util.py` | `device_util.py` |

Imports and docs now use flat `bayer_denoiser.<module>` (e.g. `bayer_denoiser.model`, `bayer_denoiser.bayer`). Archived code imports active modules at flat paths (e.g. `bayer_denoiser.bayer` from `archived/io/decode.py`).

`src/__init__.py` re-exports `BayerDenoiseNet` and `build_model` from flat modules.

## Mac sync

1. Pull Context → repo (or wait for cloud sync), then from `~/Projects/bayer_denoiser`:
   ```bash
   bash scripts/sync_context_to_repo.sh
   ```
2. Remove stale subpackage dirs if rsync left empties:
   ```bash
   rm -rf src/core src/io src/qa src/utilities
   ```
3. **`pyproject.toml` (Mac):** point console/script entry at flat modules, e.g. `bayer_denoiser.infer:main` (not `bayer_denoiser.core.infer`). Drop any `core` / `io` / `qa` / `utilities` package entries if listed explicitly.
4. Reverse sync when green:
   ```bash
   bash scripts/sync_repo_to_context.sh
   ```
   Store-root `src/` is updated from `files/src` with `--delete`; **`prune_bad_src_layout`** still removes mistaken `src/core/core/` only (unchanged).

## Install and smoke

```bash
cd ~/Projects/bayer_denoiser
source .venv/bin/activate
pip install -e .
python -m bayer_denoiser.train_static_noisy_sequence --help
python -m bayer_denoiser.infer --help
python -m bayer_denoiser.preview --help
```

Example training (see `docs/static-noisy-sequence.md`):

```bash
python -m bayer_denoiser.train_static_noisy_sequence \
  --sequence-dir runs/gaussian-gt-test/static_sequence \
  --crop-size 512 --epochs 2 --device mps
```

Context worker verified import graph through `bayer_denoiser.__init__` → `model` (stops without `torch` in cloud VM).

## Files touched (canonical: `files/`)

- `files/src/*` — moves, `README.md`, `__init__.py`, `archived/README.md` + archived import fixes
- `files/scripts/*` — `-m` paths and Python imports
- `files/docs/*.md` (live only, not `docs/archived/`)
- `files/preferences.md`
- Mirrored: store-root `src/`, live `docs/`, key `scripts/`, `preferences.md`
