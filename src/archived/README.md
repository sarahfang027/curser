# Archived `src` modules

Deferred or legacy paths. **Active training:** `python -m bayer_denoiser.train_static_noisy_sequence` ([static-noisy-sequence.md](../../docs/static-noisy-sequence.md)). **Active QA:** `python -m bayer_denoiser.preview` only.

- `gcp/` — GCPStillNet. See `docs/archived/guides/GCP_STILL.md`.
- `io/` — headerless `.raw` chain: `raw_spec.py` (contract), `raw_io.py` (load/normalize), `decode.py` (`load_bayer`, tiles). Forward training uses `.sky` → `.npy`; active `bayer_denoiser.infer` loads `.npy` directly and delegates other formats here.
- `qa/` — RAW and paired-plate QA: `validate_raw.py`, `make_sidecar.py`, `pairs.py`, `validate_pairs.py`, `visualize_crops.py`, `export_bayer.py` (`python -m bayer_denoiser.archived.qa.*`).
- `core/dataset.py` — paired `.raw`/RGB and synthetic loaders for archived `train` and `archived.qa.visualize_crops` only.
- `core/train.py` — stock CLI (`python -m bayer_denoiser.archived.core.train`): synthetic Bayer denoise and legacy paired `--bayer-dir`/`--rgb-dir` training. Not installed as a public console entry point.

**Layout mistakes:** There must be **no** `src/core/core/` (accidental copy of the whole `core` package). Remove it on Mac/Context if present; `sync_*_context*.sh` prunes it automatically.
