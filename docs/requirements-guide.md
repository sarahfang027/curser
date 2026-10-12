# Python dependencies (`requirements.txt`)

Canonical file: **`~/Projects/bayer_denoiser/requirements.txt`** only (synced from Context `files/requirements.txt`). There is **no** `scripts/requirements.txt` in the repo — remove it if an old sync left a copy.

## Packages

| Package | Minimum | Role |
|---------|---------|------|
| `torch` | 2.0 | Training, inference, tensors |
| `torchvision` | 0.15 | Transforms, synthetic RGB→Bayer |
| `numpy` | 1.24 | Arrays, `.npy` I/O, metrics |
| `Pillow` | 10.0 | PNG/JPEG RGB, infer PNG output |
| `tqdm` | 4.65 | Progress bars |
| `imageio` | 2.34.0 | **`.exr`** paired RGB (archived dataset / legacy GCP paths) |
| `matplotlib` | 3.7 | **`plot_training_history`** training curves PNG |

Former **`requirements-raw.txt`** was removed after merge (delete any leftover copy on the Mac).

## Install

```bash
cd ~/Projects/bayer_denoiser
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**PyTorch:** minimum versions only; on Apple Silicon use an MPS-capable build if needed ([pytorch.org](https://pytorch.org)).

## Do you need `imageio`?

| Path | Needs `imageio`? |
|------|------------------|
| `.sky` → `.npy`, [static-noisy-sequence.md](static-noisy-sequence.md), `.npy` infer | No (still installed; harmless) |
| Archived paired RAW + **`.exr`** RGB | Yes |
| Paired RGB as PNG/JPEG | No (Pillow) |

