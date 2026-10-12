---
cursor:
  subagentId: "bc-b78f2a0e-3f57-5459-85b5-e7ab896a9886"
---

# Archive `core/train.py` — report

## Code moves

| Action | Path |
|--------|------|
| Moved | `src/core/train.py` → `src/archived/core/train.py` |
| Moved | `scripts/bayer_denoise_smoke.sh` → `scripts/archived/bayer_denoise_smoke.sh` |
| Added | `scripts/bayer_denoise_smoke.sh` — deprecation stub (exit 1 → static-sequence doc) |

`src/core/dataset.py` left in place (duplicate of `archived/core/dataset.py`; README notes removal on Mac if desired).

## Scripts updated

- `scripts/archived/bayer_denoise_smoke.sh`, `scripts/archived/pipeline_smoke.sh` — `ROOT` fixed for `archived/`; train via `python -m bayer_denoiser.archived.core.train`
- `scripts/run_real_bayer_denoise.sh` — `RUN_TRAIN` uses archived train module
- `scripts/denoise_sample_a001c006.sh` — no auto-smoke; requires `CKPT` or static-sequence train
- `scripts/README.md`, `scripts/archived/README.md`
- Mirrored stale copies under `scripts/scripts/` (denoise sample, README, smoke stub)

## Docs updated

- `src/README.md`, `src/archived/README.md`
- `docs/guides/`: `training-user-guide.md`, `project-context.md`, `PIPELINE.md`, `BAYER_DENOISE.md`, `real-data-runbook.md`, `rename-to-bayer-denoiser.md`; bulk `core.train` → `archived.core.train` in guides/operations/inference
- `docs/operations/requirements-guide.md`, `licensing-notes.md`
- `docs/inference/a001c006-basic-denoise-test.md`

**Not changed:** `preferences.md` (already points at `train_static_noisy_sequence.py`). **`pyproject.toml` not in Context** — remove `[project.scripts]` / `bayer-denoiser-train` on Mac if present.

## Sarah — after `sync_context_to_repo.sh`

```bash
cd ~/Projects/bayer_denoiser
bash scripts/sync_context_to_repo.sh
source .venv/bin/activate
pip install -e .

# Primary plumbing smoke (needs static sequence on disk; see guide)
python scripts/train_static_noisy_sequence.py \
  --sequence-dir runs/gaussian-gt-test/static_sequence \
  --output-dir runs/static-seq-plumbing \
  --crop-size 512 --epochs 5 --device mps

# Legacy archived CLI still works:
python -m bayer_denoiser.archived.core.train -h

# Retired root smoke should fail with pointer:
./scripts/bayer_denoise_smoke.sh || true
```

**Mac-only:** delete `src/core/train.py` if sync leaves a stale copy; drop `bayer_denoiser.core.train` console entry from `pyproject.toml`.

## QA paths still on archived dataset

`python -m bayer_denoiser.qa.visualize_crops` imports `bayer_denoiser.archived.core.dataset` — documented in `src/archived/README.md`.
