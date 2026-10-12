#!/usr/bin/env bash
# Static noisy sequence: build frames under train/, train at RUN root, infer + previews in compare/.
# Mirrors compare/ semantics of scripts/static_noisy_singleframe_pipeline_test.sh (canonical denoised_infer.npy + PNGs).
#
# Layout under RUN (default: $ROOT/runs/static-noisy-sequence-test):
#   train/
#     noisy_input_crop.npy     — center crop of sample (infer input for tile mode)
#     gaussian_GT_crop.npy     — synthetic clean (sequence source)
#     static_sequence/         — clean_mosaic.npy + frame_*.npy
#   compare/                   — denoised_infer.npy, noisy.png, clean_gaussian_gt.png, denoised.png
#   RUN root                   — model.pt, model_epoch*.pt, history.json
#
# Overrides:
#   RUN, TRAIN_DIR, CHECKPOINT_DIR, COMPARE_DIR — same as static_noisy_singleframe_pipeline_test.sh
#   SAMPLE, TILE, SIGMA — build GT crop when train/gaussian_GT_crop.npy is missing
#   FRAMES, NOISE_STD, SEED — passed to make_static_noisy_sequence.py
#   CROP_SIZE — training crop (0 = full mosaic per frame; default 512 for full-plate samples)
#   NUM_CROPS — aligned noisy/GT crop pairs under train/crops/ (default 1 = center crop only)
#   CROP_GRID — e.g. 2x2 or 3x3 (sets grid; must match NUM_CROPS when both set)
#   CROP_SEED — RNG for random multi-crop placement (default 42)
#   SKIP_BUILD_GT=1 — require existing train/gaussian_GT_crop.npy
#   SKIP_SEQUENCE=1   — require existing train/static_sequence/
#   DEBUG_DENOISED_MOSAIC=1 — training-forward denoised_mosaic.npy → compare/
#   PLOT_HISTORY=1 (default) — after train, write compare/training_curves.png from history.json
#   FULL_INFER=1 or INFER_TILED=1 — tiled infer on full SAMPLE (not the 512 sequence tile)
#   INFER_TILE_SIZE (default CROP_SIZE or TILE); INFER_OVERLAP optional

set -euo pipefail

ROOT="${BAYER_DENOISER_ROOT:-${BAYER_RGB_ROOT:-$HOME/Projects/bayer_denoiser}}"
SAMPLE="${1:-$ROOT/data/samples/A001C006_0101_4E4951.npy}"
TILE="${TILE:-512}"
SIGMA="${SIGMA:-1.0}"
RUN="${RUN:-$ROOT/runs/static-noisy-sequence-test}"
TRAIN_DIR="${TRAIN_DIR:-$RUN/train}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$RUN}"
COMPARE_DIR="${COMPARE_DIR:-$RUN/compare}"
SEQ_DIR="${SEQ_DIR:-$TRAIN_DIR/static_sequence}"
FRAMES="${FRAMES:-20}"
NOISE_STD="${NOISE_STD:-0.02}"
SEED="${SEED:-42}"
CROP_SIZE="${CROP_SIZE:-512}"
NUM_CROPS="${NUM_CROPS:-1}"
CROP_SEED="${CROP_SEED:-42}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$ROOT"
if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi

mkdir -p "$TRAIN_DIR" "$COMPARE_DIR" "$CHECKPOINT_DIR" "$SEQ_DIR"

CLEAN_GT="$TRAIN_DIR/gaussian_GT_crop.npy"
NOISY_CROP="$TRAIN_DIR/noisy_input_crop.npy"

_crop_grid_args=()
if [[ -n "${CROP_GRID:-}" ]]; then
  _crop_grid_args=(--grid "$CROP_GRID")
fi

if [[ "${SKIP_BUILD_GT:-0}" != "1" ]]; then
  if [[ "$NUM_CROPS" != "1" || -n "${CROP_GRID:-}" ]]; then
    CLEAN_FULL="$TRAIN_DIR/gaussian_GT_full.npy"
    if [[ ! -f "$CLEAN_FULL" ]]; then
      echo "== Gaussian GT full frame (sigma=$SIGMA) for multi-crop =="
      python "$SCRIPT_DIR/make_bayer_gaussian_gt.py" \
        --input "$SAMPLE" \
        --output "$CLEAN_FULL" \
        --sigma "$SIGMA"
    fi
    echo "== training crop pairs (NUM_CROPS=$NUM_CROPS tile=$TILE) =="
    python "$SCRIPT_DIR/generate_training_crop_pairs.py" \
      --noisy "$SAMPLE" \
      --clean "$CLEAN_FULL" \
      --train-dir "$TRAIN_DIR" \
      --tile-size "$TILE" \
      --num-crops "$NUM_CROPS" \
      --seed "$CROP_SEED" \
      "${_crop_grid_args[@]}"
    _SEQ_CLEAN="$CLEAN_FULL"
  elif [[ ! -f "$CLEAN_GT" ]]; then
    echo "== center crop $TILE (noisy fixture) =="
    python - <<PY
import numpy as np
src = np.load("$SAMPLE")
h, w = src.shape
ts = int("$TILE")
top = (h - ts) // 2
top -= top % 2
left = (w - ts) // 2
left -= left % 2
crop = src[top : top + ts, left : left + ts].astype(np.float16)
np.save("$NOISY_CROP", crop)
print("noisy_input_crop", crop.shape)
PY
    echo "== Gaussian GT tile (sigma=$SIGMA) =="
    python "$SCRIPT_DIR/make_bayer_gaussian_gt.py" \
      --input "$NOISY_CROP" \
      --output "$CLEAN_GT" \
      --sigma "$SIGMA"
    _SEQ_CLEAN="$CLEAN_GT"
  elif [[ ! -f "$NOISY_CROP" ]]; then
    echo "== copy noisy fixture from sample crop (GT exists) =="
    python - <<PY
import numpy as np
src = np.load("$SAMPLE")
h, w = src.shape
ts = int("$TILE")
top = (h - ts) // 2
top -= top % 2
left = (w - ts) // 2
left -= left % 2
crop = src[top : top + ts, left : left + ts].astype(np.float16)
np.save("$NOISY_CROP", crop)
print("noisy_input_crop", crop.shape)
PY
    _SEQ_CLEAN="$CLEAN_GT"
  else
    _SEQ_CLEAN="$CLEAN_GT"
  fi
else
  [[ -f "$CLEAN_GT" ]] || { echo "missing $CLEAN_GT (SKIP_BUILD_GT=1)" >&2; exit 1; }
  _SEQ_CLEAN="$CLEAN_GT"
fi

if [[ "${SKIP_SEQUENCE:-0}" != "1" ]]; then
  echo "== static noisy sequence ($FRAMES frames) → $SEQ_DIR =="
  python "$SCRIPT_DIR/make_static_noisy_sequence.py" \
    --clean "${_SEQ_CLEAN:-$CLEAN_GT}" \
    --output-dir "$SEQ_DIR" \
    --frames "$FRAMES" \
    --noise-std "$NOISE_STD" \
    --seed "$SEED" \
    --write-manifest
fi

_infer_input=""
_train_crop_args=()
if [[ "$CROP_SIZE" != "0" ]]; then
  _train_crop_args=(--crop-size "$CROP_SIZE")
fi

echo "== train on static sequence =="
_train_argv=(
  python -m bayer_denoiser.train_static_noisy_sequence
  --sequence-dir "$SEQ_DIR"
  --output-dir "$CHECKPOINT_DIR"
  --epochs "${EPOCHS:-8000}"
)
if [[ ${#_train_crop_args[@]} -gt 0 ]]; then
  _train_argv+=("${_train_crop_args[@]}")
fi
if [[ "${DEBUG_DENOISED_MOSAIC:-0}" == "1" ]]; then
  _train_argv+=(--write-denoised-mosaic)
fi
"${_train_argv[@]}"

if [[ -f "$CHECKPOINT_DIR/denoised_mosaic.npy" ]]; then
  mv -f "$CHECKPOINT_DIR/denoised_mosaic.npy" "$COMPARE_DIR/denoised_mosaic.npy"
fi

if [[ "${PLOT_HISTORY:-1}" == "1" && -f "$CHECKPOINT_DIR/history.json" ]]; then
  echo "== training curves from history.json =="
  python -m bayer_denoiser.plot_training_history \
    --history "$CHECKPOINT_DIR/history.json" \
    --output "$COMPARE_DIR/training_curves.png"
fi

# Infer input: tile-sized sequence → frame_0000; cropped training → center crop of frame_0000 (matches train patch).
_frame0="$SEQ_DIR/frame_0000.npy"
if [[ ! -f "$_frame0" ]]; then
  echo "missing $_frame0" >&2
  exit 1
fi

if [[ "$CROP_SIZE" != "0" ]]; then
  _infer_crop="$COMPARE_DIR/infer_input_crop.npy"
  python - <<PY
import numpy as np
from pathlib import Path
frame = np.load("$_frame0")
h, w = frame.shape
cs = int("$CROP_SIZE")
top = (h - cs) // 2
top -= top % 2
left = (w - cs) // 2
left -= left % 2
crop = frame[top : top + cs, left : left + cs].astype(np.float16)
out = Path("$_infer_crop")
out.parent.mkdir(parents=True, exist_ok=True)
np.save(out, crop)
print("infer_input_crop", crop.shape, "at", top, left)
PY
  _infer_input="$_infer_crop"
  _noisy_preview="$_infer_input"
else
  _infer_input="$_frame0"
  _noisy_preview="$_frame0"
fi

_infer_extra=()
if [[ "${FULL_INFER:-0}" == "1" || "${INFER_TILED:-0}" == "1" ]]; then
  if [[ ! -f "$COMPARE_DIR/infer_input_crop.npy" ]]; then
    _patch_tile="${CROP_SIZE:-$TILE}"
    if [[ "$_patch_tile" == "0" ]]; then
      _patch_tile="$TILE"
    fi
    python - <<PY
import numpy as np
from pathlib import Path
src = np.load("$SAMPLE")
h, w = src.shape
ts = int("$_patch_tile")
top = (h - ts) // 2
top -= top % 2
left = (w - ts) // 2
left -= left % 2
crop = src[top : top + ts, left : left + ts].astype(np.float16)
out = Path("$COMPARE_DIR/infer_input_crop.npy")
out.parent.mkdir(parents=True, exist_ok=True)
np.save(out, crop)
print("infer_input_crop", crop.shape)
PY
  fi
  _infer_input="$SAMPLE"
  _noisy_preview="$SAMPLE"
  _infer_tile="${INFER_TILE_SIZE:-${CROP_SIZE:-$TILE}}"
  if [[ "$_infer_tile" == "0" ]]; then
    _infer_tile="$TILE"
  fi
  _infer_extra=(--tiled --tile-size "$_infer_tile")
  if [[ -n "${INFER_OVERLAP:-}" ]]; then
    _infer_extra+=(--overlap "$INFER_OVERLAP")
  fi
  echo "== tiled infer on full plate ($_infer_input, tile=$_infer_tile) =="
  python -m bayer_denoiser.infer \
    --checkpoint "$CHECKPOINT_DIR/model.pt" \
    --input "$_infer_input" \
    --output "$COMPARE_DIR/denoised_infer.npy" \
    "${_infer_extra[@]}"
  python -m bayer_denoiser.infer \
    --checkpoint "$CHECKPOINT_DIR/model.pt" \
    --input "$COMPARE_DIR/infer_input_crop.npy" \
    --output "$COMPARE_DIR/denoised_infer_patch.npy"
  python -m bayer_denoiser.preview "$COMPARE_DIR/denoised_infer_patch.npy" \
    --write-mosaic "$COMPARE_DIR/denoised_patch.png"
else
  echo "== official infer (should match training forward on same patch) =="
  python -m bayer_denoiser.infer \
    --checkpoint "$CHECKPOINT_DIR/model.pt" \
    --input "$_infer_input" \
    --output "$COMPARE_DIR/denoised_infer.npy"
fi

_clean_preview="$SEQ_DIR/clean_mosaic.npy"
if [[ "$CROP_SIZE" != "0" ]]; then
  _clean_crop="$COMPARE_DIR/clean_mosaic_crop.npy"
  python - <<PY
import numpy as np
from pathlib import Path
clean = np.load("$_clean_preview")
h, w = clean.shape
cs = int("$CROP_SIZE")
top = (h - cs) // 2
top -= top % 2
left = (w - cs) // 2
left -= left % 2
crop = clean[top : top + cs, left : left + cs].astype(np.float16)
np.save("$_clean_crop", crop)
PY
  _clean_preview="$_clean_crop"
fi

python -m bayer_denoiser.preview "$_noisy_preview" \
  --write-mosaic "$COMPARE_DIR/noisy.png"
python -m bayer_denoiser.preview "$_clean_preview" \
  --write-mosaic "$COMPARE_DIR/clean_gaussian_gt.png"
python -m bayer_denoiser.preview "$COMPARE_DIR/denoised_infer.npy" \
  --write-mosaic "$COMPARE_DIR/denoised.png"

echo "Done → train: $TRAIN_DIR  sequence: $SEQ_DIR  checkpoints: $CHECKPOINT_DIR  compare: $COMPARE_DIR"
