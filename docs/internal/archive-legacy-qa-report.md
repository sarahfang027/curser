---
cursor:
  subagentId: "bc-2a04aff7-5377-524f-a7be-ff8af506fa22"
---

# Archive legacy QA modules — report

## Summary

Four legacy QA modules moved from `src/qa/` to `src/archived/qa/`. **Active `qa/`** now contains only `preview.py` and `__init__.py`.

## File moves

| From `src/qa/` | To `src/archived/qa/` |
|----------------|------------------------|
| `pairs.py` | merged with existing archived qa tree |
| `validate_pairs.py` | |
| `visualize_crops.py` | |
| `export_bayer.py` | |

**Unchanged:** `src/qa/preview.py`, `src/qa/__init__.py` (docstring updated).

## Import / script fixes

| Location | Change |
|----------|--------|
| `archived/qa/validate_pairs.py`, `visualize_crops.py` | `bayer_denoiser.archived.qa.pairs` |
| `archived/core/dataset.py` | `bayer_denoiser.archived.qa.pairs` |
| `scripts/run_real_bayer_denoise.sh` | `archived.qa.validate_pairs`, `archived.qa.visualize_crops` |
| `files/scripts/scripts/run_real_bayer_denoise.sh` | same (store-root `scripts/scripts/` not present) |

## Docs updated (`bayer_denoiser.qa.*` → `bayer_denoiser.archived.qa.*` for pair/crop/export only; **preview** left on `qa.preview`)

- `src/README.md`, `src/archived/README.md`
- `docs/guides/training-user-guide.md`, `real-data-runbook.md`, `MANUAL.md`, `PIPELINE.md`
- `docs/formats/RAW_SPEC.md`
- `docs/archived/legacy-guides/TRAINING_GUIDE.md`

`qa.preview` references intentionally unchanged across guides and scripts.

## Mirror

Applied under **Context `files/`** (primary) and **store-root `src/`**, `scripts/run_real_bayer_denoise.sh`, and matching `docs/` copies.

## Verification

- `rg` over `files/`: no remaining `bayer_denoiser.qa.validate_pairs|visualize_crops|export_bayer|pairs`.
- Full `python -m` smoke not run here (no `pyproject.toml` / `bayer_denoiser` package root in Context snapshot).

## Not broken (by design)

- `core.train_static_noisy_sequence`, `core.infer`, `qa.preview` — paths and modules untouched.
