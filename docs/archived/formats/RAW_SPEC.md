# RAW ingest contract (v1)

Single-channel **Bayer mosaic** files for `bayer_denoiser` training and inference.  
**Enforcement:** `src/archived/io/raw_spec.py` · **CLI:** `python -m bayer_denoiser.archived.qa.validate_raw`

## Per-frame requirements

| Field | Value |
|--------|--------|
| **Geometry** | **17592 × 18024** (W × H), both **even** |
| **Storage** | **Headerless** binary, **little-endian IEEE float16** |
| **Layout** | **RGGB** CFA, **row-major** (sample `x` at column, `y` at row → index `y * W + x`) |
| **Sidecar** | **`{stem}.raw.json`** beside `{stem}.raw` (canonical name only) |
| **Naming** | Stable ID stem: `shot_0100`, `plate_a001` — `[A-Za-z][A-Za-z0-9_]*` for future RGB pairing |

### Expected file size

```
17592 × 18024 × 2 bytes = 634,156,416 bytes (~604.8 MiB)
```

Any other size is **rejected** before decode.

## Sidecar JSON (`shot_0100.raw.json`)

Required fields:

| Key | Type | Project default |
|-----|------|-----------------|
| `width` | int | `17592` |
| `height` | int | `18024` |
| `dtype` | string | `"float16"` |
| `black_level` | float | `0.0` |
| `white_level` | float | `1.0` (linear **0–1** scene; scale = `(value - black) / (white - black)`) |
| `endian` | string | `"little"` |
| `cfa_pattern` | string | `"RGGB"` |
| `row_major` | bool | `true` |
| `spec_version` | string | `"1"` |

Linear **0–1** half-float data: set `black_level` **0** and `white_level` **1**.  
HDR peaks above 1.0: raise `white_level` to your clip ceiling (or omit auto-peak scaling in loader — see `raw_io.normalize_bayer_array`).

## Tooling

| Command | Purpose |
|---------|---------|
| `python -m bayer_denoiser.archived.qa.make_sidecar path/to/shot_0100.raw` | Write canonical `.raw.json` from project defaults |
| `python -m bayer_denoiser.archived.qa.validate_raw shot_0100.raw` | Fail fast: stem, sidecar, geometry, **byte size** |
| `python -m bayer_denoiser.archived.qa.validate_raw --dir ./bayer` | Validate every `*.raw` in a folder |
| `python -m bayer_denoiser.archived.qa.validate_pairs --bayer-dir … --rgb-dir …` | After RAW passes, check RGB alignment (optional EXR later) |

Loaders (`archived.core.train`, `preview`, `validate_pairs`, legacy `.raw` in `core.infer`) decode through **`bayer_denoiser.archived.io.decode.load_bayer`**, which calls **`archived.io.raw_io.load_bayer_array`** then **`pack_mosaic`**. Forward **`core.infer`** loads **`.npy`** via `io.bayer` only.

## Decode pipeline (no ML)

```text
.raw + .raw.json  →  validate (raw_spec)  →  float32 H×W [0,1]  →  pack  →  (4, H/2, W/2)
```

Optional **`BayerTile`** crop on the mosaic before pack (`preview --tile-size`, `load_bayer(..., tile=...)`). Full-frame tiled inference is not implemented yet.

```bash
python -m bayer_denoiser.qa.preview shot_0100.raw
python -m bayer_denoiser.core.infer --baseline ...   # same decode + bilinear_demosaic
```

## Pairing (later)

RGB reference uses the **same stem**: `shot_0100.exr` ↔ `shot_0100.raw`. Not part of RAW v1 validation unless using `validate_pairs`.

## Out of scope for v1

- Embedded headers, compression, tiles
- CFA patterns other than RGGB
- Big-endian without explicit `endian` in sidecar
