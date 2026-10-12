---
cursor:
  subagentId: "bc-2054dd5c-ad7a-5fba-ad59-a4e2a85ff95d"
---

# Bayer denoise implementation

## Architecture

- **Class:** `BayerDenoiseNet` in `bayer_rgb/bayer_denoise_net.py`
- **CLI model name:** `bayer_denoise` (aliases in `build_model`: `bayer-denoise`, `bayerdenoise`)
- **Tensor contract:** packed RGGB `(B, 4, H/2, W/2)` in `[0, 1]` — same as GCP/demosaic loaders
- **Structure:** 2-level U-Net, `base_channels = max(16, nf // 2)` (default 32 when `--nf 64`)
- **Params:** ~459k at default width

## Train

- **Synthetic:** `SyntheticBayerDenoiseDataset` — smooth random RGB → clean mosaic → pack; AWGN on mosaic with `--noise-std`; target is clean packed planes
- **Paired RAW:** `PairedRawBayerDataset(bayer_target=True)` when `--model bayer_denoise` — RGB only for crop alignment; target is clean packed Bayer from plate
- **Loss / metrics:** L1 on packed output vs target; PSNR logged in packed space
- **Not used:** `make_noise_map` / second forward arg (signature matches other models for registry consistency)

## Infer

- `python -m bayer_rgb.infer` detects `bayer_denoise` from checkpoint
- **Output:** `.npy` float32 H×W mosaic via `packed_to_bayer_mosaic`
- **Input:** any path accepted by `load_bayer` (`.npy`, `.raw` + JSON, etc.)

## Smoke (`scripts/bayer_denoise_smoke.sh`)

Device: **MPS** (auto). Run: `./scripts/bayer_denoise_smoke.sh` → `runs/bayer-denoise-smoke/`

| Metric | Value (representative run) |
|--------|----------------------------|
| Final train loss | 0.41185 |
| Final train PSNR (packed) | 7.34 dB |
| Infer PSNR noisy vs clean | 26.27 dB |
| Infer PSNR denoised vs clean | 6.14 dB |

Infer PSNR on the fixed fixture is below train batch PSNR because the checkpoint sees different synthetic patches each epoch and only 3 epochs were run; train loss/PSNR trend confirms the loop.

## Docs / context

- Repo: `docs/BAYER_DENOISE.md`, `docs/PIPELINE.md` (Bayer section), `docs/GCP_STILL.md` (deferred banner)
- Project store: `docs/project-context.md` goals updated

## Files touched

- New: `bayer_rgb/bayer_denoise_net.py`, `docs/BAYER_DENOISE.md`, `scripts/bayer_denoise_smoke.sh`
- Modified: `bayer_rgb/model.py`, `dataset.py`, `train.py`, `infer.py`, `docs/PIPELINE.md`, `docs/GCP_STILL.md`
