# Archived scripts

Moved here 2026-10-05 — not part of the active pipeline. Restore by copying back to `scripts/` (or `scripts/sky/` for the Python shims) if needed.

| File | Was | Why archived |
|------|-----|----------------|
| `reorganize_docs_folders.sh` | One-time doc folder `git mv` | Migration completed |
| `pipeline_smoke.sh` | GCP-oriented smoke test | Uses archived `bayer_denoiser.archived.core.train` |
| `bayer_denoise_smoke.sh` | Synthetic train → infer smoke | Uses `archived.core.train`; active path: `python -m bayer_denoiser.train_static_noisy_sequence` ([guide](../../docs/static-noisy-sequence.md)) |
| `__init__.py`, `__main__.py` | `python -m scripts.sky` shims | Use `python scripts/sky/convert.py` instead |
| `denoise_sample_a001c006.sh` | A001C006 center-512 infer smoke | One-off sample test; use `preview` + `infer` modules directly |
| `run_real_bayer_denoise.sh` | Env-driven QA / legacy train / infer | Mostly **archived** QA + `archived.core.train`; active infer → `bayer_denoiser.infer` or `../input_and_infer_crop.py` |
| `gaussian_gt_pipeline_test.sh` | Gaussian GT pipeline test branding | **Superseded by** `../static_noisy_singleframe_pipeline_test.sh` — see [static-noisy-singleframe-pipeline-test.md](../../docs/static-noisy-singleframe-pipeline-test.md) |
