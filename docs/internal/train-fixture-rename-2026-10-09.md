---
cursor:
  subagentId: "bc-558a89b0-d560-569a-b2bd-8b8848236ba5"
---

# Train fixture rename (gaussian-gt pipeline)

## Renames (`train/` root only)

| Old | New |
|-----|-----|
| `noisy_mosaic.npy` | `noisy_input_crop.npy` |
| `clean_mosaic.npy` | `gaussian_GT_crop.npy` |
| `clean_mosaic_full.npy` | `gaussian_GT_full.npy` |

## Unchanged (sequence convention)

- `train/static_sequence/clean_mosaic.npy` — written by `make_static_noisy_sequence.py`
- `train/tile_train_seq/clean_mosaic.npy` — copy of `gaussian_GT_crop.npy` in `gaussian_gt_pipeline_test.sh`
- `src/train_static_noisy_sequence.py` — still loads `clean_mosaic.npy` inside `--sequence-dir`

## Files touched

**Scripts (canonical `scripts/` + mirrors `scripts/scripts/`, `files/scripts/`, `files/scripts/scripts/`):**

- `gaussian_gt_pipeline_test.sh`
- `make_static_noisy_sequence.sh`
- `make_full_frame_gaussian_gt.sh`
- `make_static_noisy_sequence.py` (docstring example only)

**Docs (`docs/` + `files/docs/`):**

- `gaussian-gt-pipeline-test.md`
- `static-noisy-sequence.md`
- `project-context.md`

**Not changed:** archived docs/scripts, `runs/bayer-denoise-smoke` naming in `scripts/archived/bayer_denoise_smoke.sh`, `input_and_infer_crop.py` (no fixture paths).
