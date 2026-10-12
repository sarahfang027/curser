# `runs/` folder inventory (`~/Projects/bayer_rgb`)

Gitignored local output. **Mac (2026-09-30):** `bayer-denoise-smoke/`, `gaussian-gt-test/`, `crop_test/`, `crop_visual_check/` (+ delete `pipeline-smoke/` when cleaning legacy GCP).

| Subdir | Purpose | Relevance |
|--------|---------|-----------|
| `bayer-denoise-smoke/` | `bayer_denoise` smoke | **Current** — `./scripts/bayer_denoise_smoke.sh` |
| `gaussian-gt-test/` | Gaussian GT + overfit/infer | **Current** — `scripts/make_full_frame_gaussian_gt.sh`, `gaussian_gt_pipeline_test.sh` |
| `crop_test/` | Tiny RAW crop fixtures | Optional QA |
| `crop_visual_check/` | `visualize_crops` PNGs | Optional; regen from runbook |
| `pipeline-smoke/` | GCP train→infer smoke | Legacy; `scripts/pipeline_smoke.sh` recreates if run |

**Rule:** Keep smoke + gaussian-gt-test for denoise; delete legacy GCP and unused fixture dirs anytime.
