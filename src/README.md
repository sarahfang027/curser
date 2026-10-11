# `bayer_denoiser` package layout

Active modules live at **`src/`** root (flat layout, same idea as flat `docs/`). Only **`archived/`** keeps subfolders.

| Module | Purpose |
|--------|---------|
| `bayer_denoise_net.py` | `BayerDenoiseNet` |
| `model.py` | `build_model`, demosaic helpers |
| `infer.py` | Inference CLI |
| `train_static_noisy_sequence.py` | Static-sequence training CLI |
| `sort_bayer.py` | RGGB pack/mosaic (forward `.npy` infer) |
| `preview.py` | Active mosaic QA CLI |
| `device_util.py` | Device selection helper |

## CLIs (`python -m` from repo root, `pip install -e .`)

| Command | Module |
|---------|--------|
| Infer | `python -m bayer_denoiser.infer` |
| Preview | `python -m bayer_denoiser.preview` |
| Static-sequence train | `python -m bayer_denoiser.train_static_noisy_sequence` |

**Training (active):** `python -m bayer_denoiser.train_static_noisy_sequence` — see `docs/static-noisy-sequence.md`.

**Legacy train (archived):** `python -m bayer_denoiser.archived.core.train` — synthetic / paired RGB–Bayer only; see `src/archived/README.md`.

**Headerless `.raw` and pair QA (archived):** `python -m bayer_denoiser.archived.qa.validate_raw`, `make_sidecar`, `validate_pairs`, `visualize_crops`, `export_bayer`, `archived.io.decode` — see `docs/archived/formats/RAW_SPEC.md` and `src/archived/README.md`.

`pyproject.toml` entry point (if used): `bayer_denoiser.infer:main` (remove legacy `bayer_denoiser.archived.core.train` console entry if still listed).

| **`archived/`** | `gcp/`, `io/` (`raw_spec`, `raw_io`, `decode`), `qa/` (`validate_raw`, `make_sidecar`, `pairs`, `validate_pairs`, `visualize_crops`, `export_bayer`), `core/dataset.py`, `core/train.py` |
