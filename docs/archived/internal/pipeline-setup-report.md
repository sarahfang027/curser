---
cursor:
  subagentId: "bc-dabf708b-49d6-58e3-a714-9b79e160a84d"
---

# Pipeline setup report (synthetic / no production plates)

**Date:** 2026-09-28  
**Checkout:** `~/Projects/bayer_rgb`  
**Agent:** `bc-dabf708b-49d6-58e3-a714-9b79e160a84d`

## Environment

```bash
cd ~/Projects/bayer_rgb
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-raw.txt
```

Verified: **imageio 2.37.2**, **torch MPS available**, `default_device_string()` → **`mps`**.

## Code changes (repo)

| Change | Purpose |
|--------|---------|
| `bayer_rgb/device_util.py` | `default_device_string()` — cuda → mps → cpu |
| `train.py`, `infer.py` | Use default above (no `--device` needed on Mac) |
| `docs/PIPELINE.md` | User-facing synthetic pipeline steps |
| `scripts/pipeline_smoke.sh` | One-command smoke |
| `docs/MANUAL.md` | Link + MPS default note in §7.1 |

## Commands run (verification)

```bash
./scripts/pipeline_smoke.sh
```

Equivalent manual train flags:

- `--model gcp --epochs 2 --batch-size 2 --image-size 64 --synthetic-length 16 --noise-std 0.02 --output-dir runs/pipeline-smoke`

Infer (defaults to MPS):

- `--checkpoint runs/pipeline-smoke/model.pt --input runs/pipeline-smoke/mosaic.npy --output runs/pipeline-smoke/out_rgb.png`

## Smoke artifacts (local, gitignored)

| Path | Role |
|------|------|
| `runs/pipeline-smoke/model.pt` | Checkpoint from synthetic train |
| `runs/pipeline-smoke/history.json` | Epoch loss / PSNR |
| `runs/pipeline-smoke/mosaic.npy` | 64×64 Bayer mosaic fixture |
| `runs/pipeline-smoke/out_rgb.png` | Infer output |

**Result:** exit 0; train logged `device=mps`; epoch 2 ~13.34 dB PSNR on synthetic batches.

## Flags users should know

| Flag | When |
|------|------|
| `--device cpu` | Force CPU (debug or MPS issues) |
| `--device cuda` | Linux/NVIDIA |
| `--synthetic-length`, `--image-size` | Synthetic smoke sizing |
| `--model gcp` (default) vs `demosaic` | Architecture |
| `--baseline` on infer | Bilinear only (checkpoint still required by CLI) |
| `--bayer-dir` / `--rgb-dir` | Real paired training when `data/` exists |

## Git

Initial commit on `main` with pipeline-ready source (no `runs/`, no `*.pt`, no `.venv`).
