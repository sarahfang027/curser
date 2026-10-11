# `.sky` (csky) → `.npy` conversion

Convert **`.sky`** frames to **float16 H×W Bayer mosaics** in **[0, 1]** for `bayer_rgb` (`preview`, `infer`, training). Policy: [float16-pipeline-standard.md](../guides/float16-pipeline-standard.md). Legacy **float32** `.npy` files still load; re-save as float16 when convenient.

## Format (sample `A001C004_0101_4E490B.sky`)

| Item | Value |
|------|--------|
| Magic | `csky`, little-endian |
| Metadata | Embedded `json` chunk (`clip.settings.width`, `height`, `bit_depth`, …) |
| Payload | `data` chunk: **uint16** row-major mosaic |
| Geometry | From JSON (sample: **18400 × 17712** W×H → numpy **(17712, 18400)**). The payload stores **four 4428-row banks back-to-back**; decode **deinterleaves** so output row `y` comes from bank `y % 4` (not four stacked copies). |
| Row stride | `payload_bytes / height` (sample: **36864** B/row, 64 B row padding) |
| Bit depth | JSON **12** → default **>> 2** before normalize; white **4095** |
| CFA | **RGGB** (assumed) |

No resize, transpose, or crop to project `.raw` geometry — output matches **full JSON height** from the file.

## Setup

Run from the **repo root** (`~/Projects/bayer_rgb`), not from `src/sky/` — otherwise Python cannot import the package.

```bash
cd ~/Projects/bayer_rgb
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .    # once: makes `bayer_rgb` importable from any working directory
```

After `src/` layout, delete stale **`bayer_rgb.egg-info`** at repo root if left from older installs. Rerun **`pip install -e .`** after layout changes. Egg-info dirs are gitignored (`*.egg-info/`).

## Command

Use **`python -m bayer_rgb.sky`** only. Implementation: **`src/sky/convert.py`** (+ **`src/sky/io.py`**).

```bash
cd ~/Projects/bayer_rgb
python -m bayer_rgb.sky \
  --input "/path/to/clip.sky" \
  --output "data/samples/clip.npy" \
  --write-json "data/samples/clip.sky.json"
```

Example:

```bash
python -m bayer_rgb.sky \
  --input "/Users/Sarah.Fang/Desktop/OneDrive_1_7-28-2026/A001C006_0101_4E4951.sky" \
  --output "data/samples/A001C006_0101_4E4951.npy" \
  --write-json "data/samples/A001C006_0101_4E4951.sky.json"
```

Overrides: `--black-level`, `--white-level`, `--value-shift`, `--no-row-deinterleave`, `--dump-clip-json`.

## Verify

Run from the repo root after conversion. For **`.npy`**, `preview` auto-writes a **grayscale** 8-bit mosaic PNG beside the input (`clip.npy` → `clip.png`) unless you pass **`--write-mosaic`**. Use **`--mosaic-false-color`** only if you want RGGB site tinting (red/green/blue) in the mosaic PNG — not demosaic.

### Center crop (512×512)

Fast spot-check at the plate center (RGGB phase–aligned crop):

```bash
cd ~/Projects/bayer_rgb
python -m bayer_rgb.preview data/samples/clip.npy \
  --tile-size 512 --center \
  --write-mosaic /tmp/clip_mosaic_center.png \
  --write-rgb /tmp/clip_rgb_center.png
```

Mosaic PNG is **grayscale** by default. **`--write-rgb`** is bilinear demosaic — use it to judge scene content in color.

### Full frame

Check tiling, banding, or repeat artifacts on the **entire** mosaic (no `--tile-size`; default **`--tile-size 0`** = full frame):

```bash
cd ~/Projects/bayer_rgb
python -m bayer_rgb.preview data/samples/clip.npy \
  --write-mosaic /tmp/clip_mosaic_full.png \
  --write-rgb /tmp/clip_rgb_full.png
```

Optional RGGB-tinted mosaic (not grayscale):

```bash
python -m bayer_rgb.preview data/samples/clip.npy \
  --tile-size 512 --center \
  --mosaic-false-color \
  --write-mosaic /tmp/clip_mosaic_rggb.png
```

Expect a **large** PNG (~17712×18400 on sample clips). For a quick numeric sanity check without writing images:

```bash
python -m bayer_rgb.preview data/samples/clip.npy
```

Confirm printed **`mosaic:`** shape matches the sidecar height×width (e.g. `(17712, 18400)` → H×W in the log).

Default mosaic PNG beside the `.npy` (when you omit **`--write-mosaic`**) is also **full frame** — same size warning applies.

Code: `src/sky/` (`io.py`, `convert.py`). Import name remains **`bayer_rgb`**.
