# Bayer-space denoising

**Current focus:** denoise in **Bayer domain** — network input and output are packed RGGB `(4, H/2, W/2)` (same representation as demosaic/GCP training loaders). No RGB target during train or infer.

**Deferred:** joint demosaic + denoise to RGB via `GCPStillNet` — see [GCP_STILL.md](GCP_STILL.md).

## Model

| CLI | Class | I/O |
|-----|-------|-----|
| `--model bayer_denoise` | `BayerDenoiseNet` | packed noisy Bayer → packed clean Bayer |

Small U-Net (`--nf` sets base channel width via `max(16, nf//2)`, default 32 when `nf=64`).

## Synthetic train

```bash
python -m bayer_denoiser.archived.core.train \
  --model bayer_denoise \
  --epochs 5 \
  --batch-size 4 \
  --image-size 128 \
  --synthetic-length 64 \
  --noise-std 0.05 \
  --output-dir runs/bayer-denoise
```

Loss: L1 on packed planes. PSNR in logs is computed in packed space vs clean target.

## Paired RAW (optional)

Same flags as RGB training (`--bayer-dir`, `--rgb-dir` or `pairs.csv`). RGB is used only for crop alignment; target is **clean packed Bayer** from the plate (noise injected with `--noise-std`).

## Infer

```bash
python -m bayer_denoiser.core.infer \
  --checkpoint runs/bayer-denoise/model.pt \
  --input path/to/mosaic.npy \
  --output path/to/denoised.npy
```

Output must be `.npy` (float32 H×W mosaic in [0, 1]). Input: `.npy`, `.raw` + sidecar, or other paths supported by `load_bayer`.

## Smoke

Use [static-noisy-sequence.md](static-noisy-sequence.md) (`python -m bayer_denoiser.core.train_static_noisy_sequence`). Legacy: `scripts/archived/bayer_denoise_smoke.sh` → `runs/bayer-denoise-smoke/`.
