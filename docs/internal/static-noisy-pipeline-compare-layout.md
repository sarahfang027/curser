---
cursor:
  subagentId: "bc-4e9c3aa1-4854-598c-857b-660d03d36b7a"
---

# Static noisy sequence — train/compare layout (2026-10-11)

Added `scripts/static_noisy_sequence_pipeline_test.sh` mirroring Gaussian GT `compare/` flow.

- **RUN:** `runs/static-noisy-test/`
- **Train data:** `train/static_sequence/`, `train/gaussian_GT_crop.npy`, `train/noisy_input_crop.npy`
- **Checkpoints:** run root
- **Compare:** `denoised_infer.npy`, PNG previews; `denoised_mosaic.npy` debug-only

`make_static_noisy_sequence.sh`: `PIPELINE_RUN=runs/static-noisy-test` → `train/static_sequence/`. Legacy `runs/static_sequence/` unchanged.

Mac one-liner: `bash scripts/static_noisy_sequence_pipeline_test.sh`
