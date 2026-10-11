# Python dependencies (`requirements*.txt`)

Canonical files live in the repo root: `~/Projects/bayer_rgb/requirements.txt` and `requirements-raw.txt`.

## Package lists

### `requirements.txt` (core)

| Package | Minimum version | Role in this repo |
|---------|-----------------|-------------------|
| `torch` | 2.0 | Training, inference, tensors |
| `torchvision` | 0.15 | Transforms, synthetic RGB→Bayer data |
| `numpy` | 1.24 | Arrays, `.npy` I/O, metrics |
| `Pillow` | 10.0 | PNG/JPEG RGB loads, infer PNG output (GCP path) |
| `tqdm` | 4.65 | Progress bars in train/infer |

### `requirements-raw.txt` (optional add-on)

| Package | Minimum version | Role in this repo |
|---------|-----------------|-------------------|
| `imageio` | 2.34.0 | Reading **`.exr`** paired RGB in `bayer_rgb/dataset.py` (`_load_rgb_full`) |

Nothing else is pinned in `requirements-raw.txt`. Headerless `.raw` plates use the sidecar + NumPy decode path, not `imageio`.

## When to use which

| Workflow | Install |
|----------|---------|
| **Bayer denoise only** — synthetic train, `.npy` infer, smoke script | `requirements.txt` only |
| **Bayer denoise + paired plates** where RGB is PNG/JPEG (or synthetic) | `requirements.txt` only |
| **Any path that loads paired RGB as `.exr`** (GCP still train, paired RAW with EXR refs) | `requirements.txt` **and** `requirements-raw.txt` |
| **Full repo / GCP demosaic pipeline** with production EXR references | Both files (same as MANUAL.md) |

`requirements-raw.txt` is intentionally separate so lightweight Bayer-space work does not pull in EXR support until you need it.

## Install commands

From the repo root with a virtual environment activated:

```bash
cd ~/Projects/bayer_rgb
python3 -m venv .venv
source .venv/bin/activate

# Minimum for Bayer denoise + PyTorch stack
pip install -r requirements.txt
```

When you need EXR paired RGB:

```bash
pip install -r requirements-raw.txt
```

One-shot full setup (matches `docs/MANUAL.md` and Project `project-context.md`):

```bash
pip install -r requirements.txt
pip install -r requirements-raw.txt
```

**PyTorch note:** These files only set minimum versions. On Apple Silicon, install a build with MPS if `pip` does not already give you one (see [pytorch.org](https://pytorch.org) if needed). The repo uses `device_util` to pick CUDA / MPS / CPU.

## Does Bayer-denoise-only work need `requirements-raw.txt`?

**No**, for the paths called out as current focus in `docs/BAYER_DENOISE.md`:

- `--model bayer_denoise` with **`--synthetic-length`** (no real RGB files)
- Infer with **`--input` … `.npy`** (or `.raw` + sidecar via `load_bayer`, not EXR)
- **`./scripts/bayer_denoise_smoke.sh`** — synthetic train + `.npy` infer only

**Yes**, add `requirements-raw.txt` if Bayer denoise training uses **paired RAW + RGB** and RGB paths are **`.exr`**. RGB is then used only for crop alignment; the target remains packed Bayer, but `_load_rgb_full` still imports `imageio` for EXR.

**No** for paired RAW where RGB is **`.png` / `.jpg`** — Pillow from `requirements.txt` is enough.

## Quick reference

```text
requirements.txt      → always (train / infer / smoke)
requirements-raw.txt  → only when loading .exr RGB pairs
```

Related repo docs: `docs/MANUAL.md` (§ Setup), `docs/PIPELINE.md`, `docs/BAYER_DENOISE.md`.
