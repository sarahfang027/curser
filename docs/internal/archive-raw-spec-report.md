---
cursor:
  subagentId: "bc-1e02524d-133c-5b59-8b12-30f3c690a16e"
---

# Archive headerless `.raw` chain — report

## Code moves (`files/src` and store-root `src/` mirrored)

| From | To |
|------|-----|
| `io/raw_spec.py` | `archived/io/raw_spec.py` |
| `io/raw_io.py` | `archived/io/raw_io.py` |
| `io/decode.py` | `archived/io/decode.py` |
| `qa/validate_raw.py` | `archived/qa/validate_raw.py` |
| `qa/make_sidecar.py` | `archived/qa/make_sidecar.py` |

**Added:** `archived/io/__init__.py`, `archived/qa/__init__.py`.

**Active `io/`:** `bayer.py` only (plus package `__init__.py`).

## Import updates

| Module | Change |
|--------|--------|
| `archived/io/decode.py` | `bayer_denoiser.archived.io.raw_io` |
| `archived/core/dataset.py`, `archived/core/train.py` | `archived.io.decode` / `raw_io` |
| `qa/preview.py`, `qa/validate_pairs.py`, `qa/visualize_crops.py`, `qa/pairs.py` | `archived.io.*` |
| `scripts/sky/io.py`, `scripts/scripts/sky/io.py` | `archived.io.raw_io.normalize_bayer_array` |
| `core/infer.py` | `.npy` via `io.bayer`; `.raw` prints archived hint on stderr; other legacy formats lazy-import `archived.io.decode` |

**Unchanged (no `raw_io` / `decode`):** `core/train_static_noisy_sequence.py`, `io/bayer.py`.

## Scripts

- `scripts/run_real_bayer_denoise.sh` — `python -m bayer_denoiser.archived.qa.validate_raw`

## Docs touched

- `docs/formats/RAW_SPEC.md` — enforcement path, archived CLIs, loader paragraph
- `docs/guides/project-context.md` — ingest/decode rows
- `docs/guides/MANUAL.md`, `training-user-guide.md`, `real-data-runbook.md`, `PIPELINE.md`, `docs/archived/legacy-guides/TRAINING_GUIDE.md` — CLI module paths → `archived.qa.*`
- `src/README.md`, `src/archived/README.md`

## Verification notes

- `train_static_noisy_sequence` loads `.npy` only; not affected.
- `preview` / `validate_pairs` / `visualize_crops` still work via **archived** decode/raw_io imports.
- After sync to Mac repo: `pip install -e .` then smoke `python -m bayer_denoiser.core.train_static_noisy_sequence --help` and optional `python -m bayer_denoiser.archived.qa.validate_raw --help`.
