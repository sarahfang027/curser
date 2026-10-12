---
cursor:
  subagentId: "bc-e7422679-5c2d-51eb-8c78-73cb42596ad1"
---

# Rename: Gaussian GT pipeline test → static noisy single-frame

## Rename table

| Old | New |
|-----|-----|
| `scripts/gaussian_gt_pipeline_test.sh` | `scripts/static_noisy_singleframe_pipeline_test.sh` |
| `scripts/gaussian_gt_pipeline_test.sh` (nested `scripts/scripts/`) | archived → `scripts/scripts/archived/gaussian_gt_pipeline_test.sh` |
| `scripts/gaussian_gt_pipeline_test.sh` (retired) | `scripts/archived/gaussian_gt_pipeline_test.sh` |
| `docs/gaussian-gt-pipeline-test.md` | `docs/static-noisy-singleframe-pipeline-test.md` |
| `docs/gaussian-gt-pipeline-test.md` (retired) | `docs/archived/gaussian-gt-pipeline-test.md` (superseded banner) |
| Default RUN `runs/gaussian-gt-test` | `runs/static-noisy-singleframe-test` |

**Unchanged (algorithm / fixtures):** `make_bayer_gaussian_gt.py`, `gaussian_GT_crop.npy`, `gaussian_GT_full.npy`.

**Cross-link:** single-frame = `static_noisy_singleframe_pipeline_test.sh`; multi-frame = `static_noisy_sequence_pipeline_test.sh` → `runs/static-noisy-test/`.

## Mac

```bash
cd ~/Projects/bayer_denoiser
git pull
bash scripts/sync_context_to_repo.sh   # after merge, if pulling from Context
# or: merge PR then use repo as-is
```

Existing local **`runs/gaussian-gt-test/`** can be renamed:

```bash
mv runs/gaussian-gt-test runs/static-noisy-singleframe-test
```

or re-run `bash scripts/static_noisy_singleframe_pipeline_test.sh` for a fresh RUN.

## Agent Store

Canonical mirror: `files/docs/`, `files/scripts/`. Store-root `docs/` and `scripts/` updated for live guides and active scripts.
