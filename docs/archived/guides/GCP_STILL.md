# Still GCP-Net study system

> **Project status:** Active development focuses on **Bayer-space denoising** ([BAYER_DENOISE.md](BAYER_DENOISE.md)). This GCP still path stays in the repo as the reference for a future **demosaic + denoise → RGB** stack.

Rebuild of the demosaic stack around a **still-adapted GCP-Net** (TIP2021 green-channel prior), keeping the RAW ingest + decode contract.

Paper: Guo et al., *Joint Denoising and Demosaicking with Green Channel Prior for Real-world Burst Images* ([arXiv](https://arxiv.org/abs/2101.09870), [code](https://github.com/GuoShi28/GCP-Net)).

## What we kept vs changed

| Layer | Status |
|-------|--------|
| RAW contract (`raw_spec`, `.raw` + `.raw.json`) | unchanged |
| Decode `load_bayer` → mosaic → pack | unchanged |
| **Model** | default **`gcp`** = still GCP-Net; optional `demosaic` |
| Burst InterF / PCD / DCN | **omitted** (still study = N=1) |

## Still GCP-Net pipeline

```text
packed RGGB (B,4,H/2,W/2) + noise_map
        │
        ├─► GCP branch (G1,G2) → FiLM lam/beta guides
        │
        └─► IntraF (extract + 4× GCA) → merge → ×2 upsample
                    → U-Net DualA → RGB (B,3,H,W)
```

Noise map: constant channel estimate from `--noise-std` (`make_noise_map`). Use `0` for clean RAW.

## Train (synthetic stills)

Output dir `runs/gcp-still-smoke/` is created when you run train; it is not required to exist beforehand (safe to delete from disk when cleaning `runs/`).

```bash
cd /Users/Sarah.Fang/Projects/bayer_denoiser
source .venv/bin/activate

python -m bayer_denoiser.archived.core.train \
  --model gcp \
  --epochs 5 \
  --batch-size 2 \
  --image-size 128 \
  --synthetic-length 64 \
  --noise-std 0.02 \
  --lr 1e-4 \
  --output-dir runs/gcp-still-smoke
```

Paired RAW (same as before):

```bash
python -m bayer_denoiser.archived.core.train \
  --model gcp \
  --bayer-dir data/bayer --rgb-dir data/rgb \
  --image-size 256 --batch-size 1 --noise-std 0 \
  --lr 1e-4 --epochs 30 \
  --output-dir runs/gcp-raw
```

## Infer

```bash
python -m bayer_denoiser.core.infer \
  --checkpoint runs/gcp-still-smoke/model.pt \
  --input path/to/shot_0100.raw \
  --output /tmp/out.png
```

Checkpoint stores `model` + `nf` + `noise_std`; infer restores them automatically.

## Flags

| Flag | Meaning |
|------|---------|
| `--model gcp` | Still GCP-Net (default) |
| `--model demosaic` | Original small conv + pixel_shuffle |
| `--nf 64` | GCP feature width |
| `--noise-std` | Mosaic noise *and* GCP noise-map level |

## Memory note (18K plates)

GCP-Net is much heavier than `DemosaicNet`. Train with **`--image-size 128–256`** and **`--batch-size 1`**. Full-frame infer at 17592×18024 will OOM on most machines — use crops / tiles later.
