# `runs/` folder inventory (`~/Projects/bayer_rgb`)

Gitignored local output. **Mac (2026-09-30):** `bayer-denoise-smoke/`, `static-noisy-singleframe-test/`, `static-noisy-test/`, `crop_test/`, `crop_visual_check/` (+ delete `pipeline-smoke/` when cleaning legacy GCP). Legacy dir name: `gaussian-gt-test/` (rename or re-run pipeline).

| Subdir | Purpose | Relevance |
|--------|---------|-----------|
| `bayer-denoise-smoke/` | `bayer_denoise` smoke | **Current** — `./scripts/bayer_denoise_smoke.sh` |
| `static-noisy-singleframe-test/` | 1-frame Gaussian GT smoke | **Current** — `static_noisy_singleframe_pipeline_test.sh`, `make_full_frame_gaussian_gt.sh` |
| `static-noisy-test/` | Multi-frame static burst | **Current** — `static_noisy_sequence_pipeline_test.sh` |
| `gaussian-gt-test/` | (legacy RUN name) | Same role as `static-noisy-singleframe-test/` — safe to rename locally |
| `crop_test/` | Tiny RAW crop fixtures | Optional QA |
| `crop_visual_check/` | `visualize_crops` PNGs | Optional; regen from runbook |
| `pipeline-smoke/` | GCP train→infer smoke | Legacy; `scripts/pipeline_smoke.sh` recreates if run |

**Rule:** Keep smoke + static-noisy-* test runs for denoise; delete legacy GCP and unused fixture dirs anytime.
