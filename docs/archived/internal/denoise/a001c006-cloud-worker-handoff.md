---
cursor:
  subagentId: "bc-d4258684-1867-512b-954e-a5cc1238ca4f"
---

# A001C006 basic denoise — cloud worker handoff

## Request

Basic denoise test on:

`/Users/Sarah.Fang/Projects/bayer_rgb/data/samples/A001C006_0101_4E4951.npy`

## Blocker (cloud)

- Cloud VM repo is **`sarahfang027/curser`** (empty); **`bayer_rgb` is not in GitHub** and lives only on the Mac worker.
- **No self-hosted workers** connected this run (`list-self-hosted-workers` → empty).
- Sample `.npy` and `runs/bayer-denoise-smoke/model.pt` are **not on the cloud disk**.

## Delivered (Project store)

| Path | Purpose |
|------|---------|
| [docs/inference/a001c006-basic-denoise-test.md](../../docs/inference/a001c006-basic-denoise-test.md) | User runbook |
| [scripts/denoise_sample_a001c006.sh](../../scripts/denoise_sample_a001c006.sh) | Runnable Mac script (copy into `~/Projects/bayer_rgb/scripts/` or run via bash with `BAYER_RGB_ROOT`) |

## Recommended execution (Mac)

```bash
cd ~/Projects/bayer_rgb
source .venv/bin/activate
bash /path/to/project/store/scripts/denoise_sample_a001c006.sh
```

Or sync script into repo and `bash scripts/denoise_sample_a001c006.sh`.

Uses **512×512 center crop** + `runs/bayer-denoise-smoke/model.pt` → `runs/sample-tests/A001C006_denoised512.npy` + PNG previews.

## Verification pending

Re-run on Mac with self-hosted worker connected, or user pastes infer log / PNG paths after local run.

## Update (Gaussian GT pipeline)

User wants Gaussian-denoised `.npy` as synthetic GT for pipeline testing. Added:

- [docs/inference/gaussian-gt-pipeline-test.md](../../docs/inference/gaussian-gt-pipeline-test.md)
- [scripts/make_bayer_gaussian_gt.py](../../scripts/make_bayer_gaussian_gt.py)
- [scripts/overfit_bayer_denoise_pair.py](../../scripts/overfit_bayer_denoise_pair.py)
- [scripts/gaussian_gt_pipeline_test.sh](../../scripts/gaussian_gt_pipeline_test.sh)
