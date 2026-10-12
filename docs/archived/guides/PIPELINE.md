# Pipeline without production data

Short path to verify **train → checkpoint → infer** on synthetic data while plates are still in preparation. Full ingest and paired training: [MANUAL.md](MANUAL.md).

**Prerequisites** (from repo root):

```bash
cd ~/Projects/bayer_denoiser
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-raw.txt   # EXR / imageio for when paired RGB arrives
```

On Apple Silicon, `train` and `infer` default to **MPS** when CUDA is unavailable. Override with `--device cpu` if needed.

## Bayer-space denoise (current path)

Train and infer stay in **packed RGGB**; no RGB export. Step-by-step training: [training-user-guide.md](training-user-guide.md). Model summary: [BAYER_DENOISE.md](BAYER_DENOISE.md).

Primary plumbing: [static-noisy-sequence.md](static-noisy-sequence.md) (`python -m bayer_denoiser.core.train_static_noisy_sequence`). Legacy synthetic smoke: `scripts/archived/bayer_denoise_smoke.sh`.

## GCP / RGB demosaic (deferred reference)

Still GCP-Net (`--model gcp`) remains in-tree for future demosaic+denoise experiments. Smoke: `scripts/archived/pipeline_smoke.sh` or steps below.

## One-shot smoke script (GCP)

```bash
./scripts/archived/pipeline_smoke.sh
```

Writes under `runs/pipeline-smoke/` (gitignored; **created by the script** — no need to keep an old copy on disk): checkpoint, mosaic `.npy`, and RGB PNG.

## Manual steps (same flow)

**1. Synthetic train** (default model `gcp`):

```bash
python -m bayer_denoiser.archived.core.train \
  --model gcp \
  --epochs 2 \
  --batch-size 2 \
  --image-size 64 \
  --synthetic-length 16 \
  --noise-std 0.02 \
  --output-dir runs/pipeline-smoke
```

**2. Fixture mosaic for infer** (avoids strict full-plate `.raw` contract):

```bash
python - <<'PY'
import numpy as np, torch
from pathlib import Path
from bayer_denoiser.io.bayer import rgb_to_bayer
out = Path("runs/pipeline-smoke/mosaic.npy")
out.parent.mkdir(parents=True, exist_ok=True)
np.save(out, rgb_to_bayer(torch.rand(1, 3, 64, 64)).squeeze().numpy())
print(out)
PY
```

**3. Infer**

```bash
python -m bayer_denoiser.core.infer \
  --checkpoint runs/pipeline-smoke/model.pt \
  --input runs/pipeline-smoke/mosaic.npy \
  --output runs/pipeline-smoke/out_rgb.png
```

**4. Optional decode preview**

```bash
python -m bayer_denoiser.core.infer \
  --baseline \
  --checkpoint runs/pipeline-smoke/model.pt \
  --input runs/pipeline-smoke/mosaic.npy \
  --output runs/pipeline-smoke/out_bilinear.png
```

(`--checkpoint` is still required by the parser; weights are ignored with `--baseline`.)

## When real data is ready

1. `data/bayer/{stem}.raw` + sidecar, `data/rgb/{stem}.exr` (or `pairs.csv`).
2. `python -m bayer_denoiser.archived.qa.validate_raw` / `archived.qa.validate_pairs` / `archived.qa.visualize_crops`.
3. Train with `--bayer-dir` and `--rgb-dir` per [MANUAL.md §7.2](MANUAL.md#72-production-style--paired-full-frame-raw).
