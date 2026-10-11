---
cursor:
  subagentId: "bc-7fb4392c-43fb-5b6c-bc37-78c45feed553"
---

# Phase B — paired noisy / clean Bayer `.raw` (desired vs stock)

## User intent (primary layout)

```text
data/
  noisy/
    shot_0100.raw
    shot_0100.raw.json
  clean/
    shot_0100.raw
    shot_0100.raw.json    # denoised export or ground-truth Bayer; same H×W as noisy
```

Optional manifest **`pairs.csv`**:

```csv
noisy,clean
/path/to/data/noisy/shot_0100.raw,/path/to/data/clean/shot_0100.raw
```

Same **stem** per row is recommended so QA scripts can list by name.

## Stock `bayer_rgb.train` today (Mac `~/Projects/bayer_rgb`)

| Behavior | Detail |
|----------|--------|
| Paired mode | `--bayer-dir` + `--rgb-dir`, or `--pairs-csv` with columns **`bayer,rgb`** |
| Input mosaic | Loaded from **`bayer`** path only |
| Training target | With `--model bayer_denoise`, `PairedRawBayerDataset(bayer_target=True)` sets target = **clean packed Bayer from the same `bayer` file** |
| Input noise | **`--noise-std`** adds Gaussian noise on the input mosaic only (on top of whatever is in the file) |
| **`rgb` column** | Used only to pick **aligned crop windows** (same `(top, left, size)` as GCP training). Loader expects **RGB** (`.exr`, `.png`, `.tif` via imageio), **not** a second Bayer `.raw` |

**Conclusion:** Stock Phase B is **one clean plate per stem** + optional synthetic noise, **not** `(noisy.raw → clean.raw)` supervision.

### Workaround check: `pairs.csv` with `rgb` = clean `.raw`

**Not supported.** `validate_pairs` / `visualize_crops` / `PairedRawBayerDataset` treat the second path as an **RGB reference image** for alignment strips, not as a packed-Bayer target. Pointing `rgb` at a `.raw` clean mosaic is not a supported training target path.

## Interim training (until repo change)

1. **Export aligned crops** (same window on noisy and clean), e.g. float32 H×W `.npy` mosaics in `[0, 1]`.
2. Run Context script **`overfit_bayer_denoise_pair.py`** (synced to `~/Projects/bayer_rgb/scripts/`) per patch or small catalog — see [gaussian-gt-pipeline-test.md](../docs/gaussian-gt-pipeline-test.md) for the same pattern with Gaussian pseudo-GT.
3. For **catalog-scale** paired raw training, implement the gap below; do not rely on `--noise-std` alone when the on-disk noisy plate is the real input.

## Implementation gap (repo)

Add first-class **noisy → clean Bayer** paired raw training:

| Area | Change |
|------|--------|
| **CLI** | e.g. `--noisy-dir` + `--clean-dir`, or `--pairs-csv` with **`noisy,clean`** (and keep `bayer,rgb` for GCP / legacy alignment-only path) |
| **`PairedRawBayerDataset` (or sibling)** | Load **input** from noisy path, **target** from clean path (packed RGGB); **same crop window** on both mosaics (no RGB required when both are Bayer) |
| **`bayer_denoise`** | Set **`--noise-std 0`** when noise is already in the noisy file; do not add synthetic noise on top unless explicitly desired |
| **QA** | `validate_paired_bayer` (or extend `validate_pairs`): same stem/geometry, sidecars present, H×W match; optional visualize packed noisy vs clean strips |
| **`train.py`** | Branch: if noisy/clean dirs or `noisy,clean` CSV → new dataset; else existing `bayer`+`rgb` path |

**Non-goals for v1:** using EXR as clean target for `bayer_denoise` (RGB is still wrong domain); auto-inferring clean from noisy in the same folder.

## Related Context

- [training-user-guide.md](../docs/training-user-guide.md) — Phase B (user-facing)
- [real-data-runbook.md](../docs/real-data-runbook.md) — commands and decision tree
- [bayer-denoise-implementation.md](bayer-denoise-implementation.md) — stock `bayer_target=True` behavior
