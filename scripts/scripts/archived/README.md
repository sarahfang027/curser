# Archived scripts

Moved here 2026-10-05 — not part of the active pipeline. Restore by copying back to `scripts/` (or `scripts/sky/` for the Python shims) if needed.

| File | Was | Why archived |
|------|-----|----------------|
| `reorganize_docs_folders.sh` | One-time doc folder `git mv` | Migration completed |
| `pipeline_smoke.sh` | GCP-oriented smoke test | Uses archived `bayer_denoiser.archived.core.train` |
| `bayer_denoise_smoke.sh` | Synthetic train → infer smoke | Superseded by `python -m bayer_denoiser.train_static_noisy_sequence` |
| `__init__.py`, `__main__.py` | `python -m scripts.sky` shims | Use `python scripts/sky/convert.py` instead |
| `gaussian_gt_pipeline_test.sh` | Gaussian GT pipeline test branding | **Superseded by** `../../static_noisy_singleframe_pipeline_test.sh` |
