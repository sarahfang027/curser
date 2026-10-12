---
cursor:
  subagentId: "bc-ae1d5926-3c51-5c23-97a9-5df43471b53c"
---

# `src/` subpackage reorganization (2026-10-05)

## Tree (excluding `__pycache__`)

```
src/
├── README.md
├── __init__.py
├── archived/          # untouched
├── core/              # bayer_denoise_net, model, train, infer, dataset
├── io/                # bayer, decode, raw_io, raw_spec
├── qa/                # validate_*, pairs, preview, visualize_crops, export_bayer, make_sidecar
├── utilities/         # device_util
└── <name>.py          # compatibility shims (17 modules)
```

## Shim strategy

Each former top-level module is a thin file at `src/<name>.py`:

- `from bayer_denoiser.<subpackage>.<name> import *`
- CLI modules add `from … import main` and `if __name__ == "__main__": main()`

Preserves `python -m bayer_denoiser.train` (and infer, preview, validate_raw, etc.) and `from bayer_denoiser.model import …` without changing shell scripts.

Implementations live only under `core/`, `io/`, `qa/`, `utilities/` with imports using those prefixes.

## Files updated outside `src/`

| Path | Change |
|------|--------|
| `scripts/train_static_noisy_sequence.py` | `utilities.device_util`, `core.model` |
| `scripts/input_and_infer_crop.py` | same |
| `scripts/sky/io.py` | `io.raw_io` |
| `scripts/convert_npy_folder_to_png.py` | docstring only; still `-m bayer_denoiser.preview` |
| `files/src/` | full mirror of `src/` |
| `files/scripts/` | mirror of `scripts/` |

No `pyproject.toml` in Agent Store; repo entry points unchanged if they pointed at top-level module names.

## Verification

- `compileall` on package via `PYTHONPATH` + `bayer_denoiser` symlink: OK
- `py_compile` on four scripts above: OK

## Broken `-m` paths

**None** for documented CLIs — shims keep legacy module paths.

Optional new paths (no shim required): `bayer_denoiser.core.train`, `bayer_denoiser.qa.preview`, etc.
