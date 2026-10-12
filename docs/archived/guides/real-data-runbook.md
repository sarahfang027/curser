# Real data runbook — `BayerDenoiseNet` on your Mac

Run everything **locally** under **`~/Projects/bayer_denoiser`**. This doc is the single entry point when you have production-ish plates (not just synthetic smoke). Commands assume an activated venv:

```bash
cd ~/Projects/bayer_denoiser
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pip install -r requirements-raw.txt   # before EXR / paired RGB QA
```

Copy this doc and the helper script from Project Context after updates:

```bash
STORE="$HOME/Library/Application Support/Cursor/AgentStores/cursor_agent_stores/bc-a64d420e-f2ca-403e-91f6-786d63f9c464/files"
bash "$STORE/scripts/sync_context_to_repo.sh"
```

**Related Context docs:** [Training user guide](training-user-guide.md) (Phase A/B, flags, troubleshooting), [Sky → `.npy`](../formats/sky-format-conversion.md), [Float16 pipeline standard](float16-pipeline-standard.md), [Gaussian GT pipeline test](../inference/static-noisy-singleframe-pipeline-test.md) (plumbing without true clean plates), [Project context](project-context.md). Background on your machine: [local machine audit](../internal/local-machine-audit.md).

---

## 1. Formats the repo accepts today

| Format | Role | Notes |
|--------|------|--------|
| **Headerless `.raw` + sidecar `{stem}.raw.json`** | Primary production ingest | Geometry, dtype, black/white in JSON. See repo `docs/RAW_SPEC.md`. Create sidecar: `python -m bayer_denoiser.archived.qa.make_sidecar path/to/plate.raw`. |
| **Float16 mosaic `.npy` (H×W, [0, 1])** | Infer, preview, Gaussian-GT scripts | Default after `.sky` conversion or export (legacy **float32** still loads). **Not** a 4-plane file on disk — training/infer **pack** to RGGB `(4, H/2, W/2)` internally. |
| **`.sky` (csky)** | Source container | Convert once: `python -m bayer_denoiser.sky --input clip.sky --output data/samples/clip.npy` — [sky-format-conversion.md](../formats/sky-format-conversion.md). |
| **`pairs.csv`** | Manifest | **Target (not in stock `train` yet):** `noisy,clean` paths to paired `.raw`. **Today:** `bayer,rgb` for legacy layout (absolute paths OK). |
| **Paired noisy + clean `.raw`** | **Primary Phase B supervision** | `data/noisy/{stem}.raw` + `data/clean/{stem}.raw` (+ sidecars), same geometry. **Stock `train` does not load this pair** — see §3 and [training-user-guide Phase B](training-user-guide.md#phase-b--real-plates-noisy--clean-bayer-raw). |
| **RGB reference (`.exr`, `.png`, `.tif`)** | **Crop alignment only** (legacy `bayer` + `rgb` dirs) | Same **stem** and **H×W** as the **bayer** mosaic. **Not** the denoise target. Requires `requirements-raw.txt` for EXR. |
| **Float16 `.npy` mosaic pairs** | Interim train / plumbing | **`python -m bayer_denoiser.core.train_static_noisy_sequence`** on aligned noisy/clean crops; [Gaussian GT pipeline](../inference/static-noisy-singleframe-pipeline-test.md) for pseudo-GT. |

**Infer inputs:** `.npy` mosaic, `.raw` + sidecar, or other paths `load_bayer` accepts (`python -m bayer_denoiser.core.infer -h`).

**Infer outputs (`bayer_denoise`):** `--output` must end in **`.npy`** (or `.raw`) — **float16** H×W denoised mosaic ([standard](float16-pipeline-standard.md)).

---

## 2. Decision tree — which path?

```text
Do you already have a trained checkpoint (model.pt)?
│
├─ YES — goal is only to denoise new plates
│     → §4 Infer-only (use crops on ~18K geometry until tiled infer exists)
│     Checkpoint from: your plate train, synthetic smoke, or Gaussian tile train (see limitations below)
│
└─ NO — you need weights first
      │
      ├─ You have noisy + clean Bayer `.raw` (or `.npy` mosaics)
      │     → §3 primary layout `data/noisy` + `data/clean`
      │     • Stock `train`: **gap** — use `python -m bayer_denoiser.core.train_static_noisy_sequence` on exported crops until `train` accepts `noisy,clean`
      │     • Legacy: one **clean** plate in `data/bayer` + `--noise-std` (+ optional `data/rgb` EXR for crop QA only)
      │     • No clean reference → cannot learn true denoise; Gaussian / smoke plumbing only
      │
      ├─ You only have noisy plates (no clean reference)
      │     → Train cannot learn true sensor noise without a target
      │     • Plumbing: [Gaussian GT pipeline](../inference/static-noisy-singleframe-pipeline-test.md)
      │     • Denoise anyway: infer-only with **smoke** or **Gaussian tile-train** checkpoint (§5 limitations)
      │
      └─ No real data yet
            → Phase 1 plumbing: `python -m bayer_denoiser.core.train_static_noisy_sequence` — [static-noisy-sequence.md](static-noisy-sequence.md)
```

### Infer with smoke / Gaussian tile-train checkpoint (limitations)

| Checkpoint source | Good for | Not good for |
|-------------------|----------|----------------|
| `runs/bayer-denoise-smoke/model.pt` | Pipeline check on a **512×512** crop | Production look; trained on random synthetic patches, not your sensor |
| `runs/static-noisy-singleframe-test/model.pt` | Matching **that** patch to a Gaussian-smoothed target | Real texture / true denoise; biased soft target |
| `runs/bayer-denoise-plates/model.pt` (legacy `bayer`+`rgb` train) | Clean plates + **`--noise-std`** regime you trained | Real noisy plates or mismatched clean target |
| `runs/.../model.pt` from **`python -m bayer_denoiser.core.train_static_noisy_sequence`** | That **noisy/clean `.npy`** patch (or small set) | Full-catalog generalization until paired-raw `train` exists |

---

## 3. Folder layout under `data/`

Use a stable tree; large catalogs often keep plates on an external volume and point `pairs.csv` at absolute paths.

### Primary paired train (Phase B) — noisy + clean Bayer

```text
~/Projects/bayer_denoiser/
  data/
    noisy/
      shot_0100.raw
      shot_0100.raw.json
    clean/
      shot_0100.raw
      shot_0100.raw.json     # denoised or GT Bayer; same H×W as noisy
  pairs.csv                  # optional (desired): noisy,clean columns — not in stock train yet
```

Example **`pairs.csv`** (for future / spec; map to CLI when implemented):

```csv
noisy,clean
/Volumes/plates/noisy/shot_0100.raw,/Volumes/plates/clean/shot_0100.raw
```

**Interim command** (aligned `.npy` crops):

```bash
mkdir -p runs/crops/shot_0100_seq
cp runs/crops/shot_0100_clean256.npy runs/crops/shot_0100_seq/clean_mosaic.npy
cp runs/crops/shot_0100_noisy256.npy runs/crops/shot_0100_seq/frame_0000.npy

python -m bayer_denoiser.core.train_static_noisy_sequence \
  --sequence-dir runs/crops/shot_0100_seq \
  --output-dir runs/bayer-denoise-pair-train \
  --frames 1
```

### Legacy stock layout — clean `bayer` + RGB alignment + `--noise-std`

```text
  data/
    bayer/
      shot_0100.raw          # treated as CLEAN mosaic by stock train
      shot_0100.raw.json
    rgb/
      shot_0100.exr          # same stem, same H×W (alignment only)
```

### Sky sample + infer-only (no EXR)

```text
  data/
    samples/
      A001C006_0101_4E4951.npy
      A001C006_0101_4E4951.sky.json   # optional, from --write-json
```

### Many plates (~18K **frames**, not pixel size)

```text
  data/
    bayer/                   # or symlink farm to /Volumes/plates/bayer/
      plate_00001.raw
      plate_00001.raw.json
      ...
    rgb/
      plate_00001.exr
      ...
  pairs.csv                  # optional: explicit paths if dirs differ
```

Example `pairs.csv`:

```csv
bayer,rgb
/Volumes/plates/bayer/plate_00001.raw,/Volumes/plates/rgb/plate_00001.exr
/Volumes/plates/bayer/plate_00002.raw,/Volumes/plates/rgb/plate_00002.exr
```

### Infer-only staging (convention)

```text
  data/
    infer_in/                # noisy .raw or .npy you will denoise
      night_042.raw
      night_042.raw.json
  runs/
    infer_out/
      night_042_denoised.npy
```

---

## 4. Command sequences (`--model bayer_denoise`)

### 4a. QA before training

**Noisy + clean `.raw` (primary layout):**

```bash
python -m bayer_denoiser.archived.qa.validate_raw --dir data/noisy
python -m bayer_denoiser.archived.qa.validate_raw --dir data/clean
# Match stems and H×W manually until validate_noisy_clean_pairs exists in repo
```

Or with the helper script: `NOISY_DIR=data/noisy CLEAN_DIR=data/clean RUN_QA=1 RUN_TRAIN=0 bash scripts/run_real_bayer_denoise.sh`

**Legacy `bayer` + `rgb`:**

```bash
python -m bayer_denoiser.archived.qa.validate_raw --dir data/bayer

python -m bayer_denoiser.archived.qa.validate_pairs \
  --bayer-dir data/bayer \
  --rgb-dir data/rgb
# or:
python -m bayer_denoiser.archived.qa.validate_pairs --pairs-csv pairs.csv
```

Fix `[MISMATCH]` on size or missing sidecars before long runs.

**Visual crop alignment** (RGB strip vs bilinear — catches shifted EXR exports):

```bash
python -m bayer_denoiser.archived.qa.visualize_crops \
  --bayer-dir data/bayer \
  --rgb-dir data/rgb \
  --pair-index 0 \
  --crop-size 256 \
  --num-crops 8 \
  --seed 42 \
  --output-dir runs/crop_visual_check
```

Open PNGs under `runs/crop_visual_check/`. Shifted strips → fix pairing or export, not the denoiser.

### 4b. Train on real plates

**Paired noisy/clean `.raw`:** not available in stock `python -m bayer_denoiser.archived.core.train` — use §3 interim `python -m bayer_denoiser.core.train_static_noisy_sequence` or wait for `noisy,clean` support ([internal spec](../internal/phase-b-paired-raw-spec.md)).

**Legacy stock train** (`data/bayer` = clean mosaic, `--noise-std` on input, `data/rgb` for crop alignment). Crop windows only when `--image-size` > 0 (default), so **17592×18024**-class plates do not load fully each step.

```bash
python -m bayer_denoiser.archived.core.train \
  --model bayer_denoise \
  --bayer-dir data/bayer \
  --rgb-dir data/rgb \
  --raw-dtype float16 \
  --black-level 0 \
  --white-level 1 \
  --image-size 256 \
  --crops-per-frame 16 \
  --batch-size 1 \
  --noise-std 0.02 \
  --epochs 30 \
  --lr 1e-4 \
  --nf 64 \
  --output-dir runs/bayer-denoise-plates
```

Manifest instead of dirs:

```bash
python -m bayer_denoiser.archived.core.train \
  --model bayer_denoise \
  --pairs-csv pairs.csv \
  --image-size 256 \
  --batch-size 1 \
  --crops-per-frame 16 \
  --noise-std 0.02 \
  --epochs 30 \
  --output-dir runs/bayer-denoise-plates
```

Match decode flags to your plates (`--raw-width`, `--raw-height`, etc.) — same as training guide Phase B.

**Synthetic baseline (no files):**

```bash
python -m bayer_denoiser.core.train_static_noisy_sequence --help   # see static-noisy-sequence.md
# or longer: python -m bayer_denoiser.archived.core.train --model bayer_denoise --epochs 10 --synthetic-length 512 ...
```

### 4c. Infer — full plate vs crop

**Crop (recommended first on ~18K×18K geometry):**

```bash
python - <<'PY'
import numpy as np
from pathlib import Path
src = np.load("data/samples/A001C006_0101_4E4951.npy")
ts = 512
h, w = src.shape
top = (h - ts) // 2; top -= top % 2
left = (w - ts) // 2; left -= left % 2
crop = src[top:top+ts, left:left+ts].astype(np.float16)
Path("runs/infer_in").mkdir(parents=True, exist_ok=True)
np.save("runs/infer_in/center512.npy", crop)
PY

python -m bayer_denoiser.core.infer \
  --checkpoint runs/bayer-denoise-plates/model.pt \
  --input runs/infer_in/center512.npy \
  --output runs/infer_out/center512_denoised.npy

python -m bayer_denoiser.qa.preview runs/infer_out/center512_denoised.npy \
  --write-mosaic runs/infer_out/center512_denoised.png
```

**Full `.raw` plate** (same sidecar flags as train):

```bash
python -m bayer_denoiser.core.infer \
  --checkpoint runs/bayer-denoise-plates/model.pt \
  --input data/bayer/shot_0100.raw \
  --output runs/infer_out/shot_0100_denoised.npy \
  --raw-dtype float16 \
  --black-level 0 \
  --white-level 1
```

**Full-frame `.npy` at production resolution:** `infer` loads the **entire** array — expect **MPS/GPU OOM** on ~17K×18K. Use crops, or decode QA via `preview --tile-size 512` without running the full U-Net forward.

### 4d. Memory and scale notes

| Concern | Guidance |
|---------|----------|
| **~18K × ~18K pixels** | Training: safe with **`--image-size` 128–256**, **`--batch-size 1`**, **`--crops-per-frame` 8–32**. Avoid **`--full-frame`**. Infer: **crop** or accept OOM risk. |
| **~18K plates (many files)** | Dataset iterates per file; you do **not** need all plates in RAM. Use **`pairs.csv`**, validate once, start with a **subset** epoch (`--epochs` low) to time one epoch. |
| **Mac MPS** | `train` / `infer` pick **cuda → mps → cpu**; override with `--device cpu` if unstable. |
| **EXR** | `pip install -r requirements-raw.txt` before `validate_pairs` / `visualize_crops` on `.exr`. |

---

## 5. Helper script (env vars)

Template in Context: `scripts/run_real_bayer_denoise.sh`. After sync:

```bash
# Infer-only
export CHECKPOINT=runs/bayer-denoise-plates/model.pt
export BAYER_DIR=data/infer_in          # or set INPUT_NPY=path/to/crop.npy
export OUTPUT_DIR=runs/infer_out
bash scripts/run_real_bayer_denoise.sh

# Legacy train + QA (stock CLI: clean bayer + rgb alignment)
unset CHECKPOINT
export BAYER_DIR=data/bayer
export RGB_DIR=data/rgb    # or PAIRS_CSV=pairs.csv with bayer,rgb columns
export OUTPUT_DIR=runs/bayer-denoise-plates
export RUN_TRAIN=1
bash scripts/run_real_bayer_denoise.sh

# Paired noisy/clean .raw (primary layout) — validate only today:
export NOISY_DIR=data/noisy
export CLEAN_DIR=data/clean
export RUN_TRAIN=0
export RUN_QA=1
bash scripts/run_real_bayer_denoise.sh
# Then train via overfit on exported .npy crops (see §3) until train.py supports NOISY_DIR/CLEAN_DIR
```

See script comments for `NOISY_DIR`, `CLEAN_DIR`, `CROP_SIZE`, `EPOCHS`, decode flags.

---

## 6. What to tell the coordinator (checklist)

Copy this block into chat when asking for help tuning trains or debugging ingest:

```text
- Repo path: ~/Projects/bayer_denoiser (venv OK? pip install -e . ?)
- Goal: [ ] infer only  [ ] train + infer  [ ] plumbing / smoke
- Input format: [ ] .raw+JSON  [ ] .npy mosaic  [ ] .sky (converted?)  [ ] other
- Geometry: H×W = ______ ; dtype / black / white = ______
- Plate count: ______ ; example paths: ______
- Paired clean target? [ ] no  [ ] yes — layout: [ ] data/noisy + data/clean `.raw`  [ ] legacy clean in data/bayer + --noise-std  [ ] ``.npy` crops + static-sequence train  [ ] Gaussian pseudo-GT only
- RGB/EXR for alignment? [ ] no  [ ] yes — path pattern: ______
- Checkpoint (if infer): path ______ or "need smoke / new train"
- Constraints: crop size ______ ; full-frame required? [ ] yes [ ] no
```

---

## Quick reference

| Step | Command |
|------|---------|
| `.sky` → `.npy` | `python -m bayer_denoiser.sky --input … --output …` |
| Raw validate | `python -m bayer_denoiser.archived.qa.validate_raw --dir data/bayer` |
| Pair validate | `python -m bayer_denoiser.archived.qa.validate_pairs …` |
| Crop QA | `python -m bayer_denoiser.archived.qa.visualize_crops …` |
| Train | `python -m bayer_denoiser.archived.core.train --model bayer_denoise --bayer-dir … --rgb-dir …` |
| Infer | `python -m bayer_denoiser.core.infer --checkpoint … --input … --output ….npy` |
| Plumbing without clean plates | [static-noisy-singleframe-pipeline-test.md](../inference/static-noisy-singleframe-pipeline-test.md) |
