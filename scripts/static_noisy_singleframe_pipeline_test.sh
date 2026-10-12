#!/usr/bin/env bash
# Static noisy single-frame pipeline test: Gaussian GT tile (+ optional full) → train (frames=1) → infer + PSNR.
# Pipeline plumbing smoke — Gaussian GT is not a true noise-free reference. Multi-frame: static_noisy_sequence_pipeline_test.sh
#
# Layout under RUN (default: $ROOT/runs/static-noisy-singleframe-test):
#   train/   — data only: noisy_input_crop.npy, gaussian_GT_*.npy, tile_train_seq/
#   compare/ — denoised_infer.npy + preview PNGs (canonical); optional denoised_mosaic.npy when DEBUG_DENOISED_MOSAIC=1
#   (multi-frame static sequence: scripts/static_noisy_sequence_pipeline_test.sh → runs/static-noisy-sequence-test/train/static_sequence/)
#   RUN root — model.pt, model_epoch*.pt, history.json (outside train/)
#
# Overrides:
#   TRAIN_DIR      — training data only (default: $RUN/train)
#   CHECKPOINT_DIR — model.pt, model_epoch*.pt, history.json (default: $RUN)
#   COMPARE_DIR — infer / preview artifacts (default: $RUN/compare)
#   RUN         — parent run folder (default: $ROOT/runs/static-noisy-singleframe-test)
#   DEBUG_DENOISED_MOSAIC=1 — pass --write-denoised-mosaic to train; move denoised_mosaic.npy to compare/
#   PLOT_HISTORY=1 (default) — after train, write compare/training_curves.png from history.json
#   FULL_INFER=1 or INFER_TILED=1 — tiled infer on full SAMPLE → compare/denoised_infer.npy + full denoised.png
#   INFER_TILE_SIZE — tile edge (default: TILE); INFER_OVERLAP — blend overlap (default: infer CLI 64)
#   NUM_CROPS — aligned crop pairs under train/crops/ (default 1); CROP_GRID e.g. 2x2; CROP_SEED (default 42)

set -euo pipefail

ROOT="${BAYER_DENOISER_ROOT:-${BAYER_RGB_ROOT:-$HOME/Projects/bayer_denoiser}}"
SAMPLE="${1:-$ROOT/data/samples/A001C006_0101_4E4951.npy}"
TILE="${TILE:-512}"
SIGMA="${SIGMA:-1.0}"
SIGMA_FULL="${SIGMA_FULL:-$SIGMA}"
BUILD_FULL_GT="${BUILD_FULL_GT:-1}"
RUN="${RUN:-$ROOT/runs/static-noisy-singleframe-test}"
TRAIN_DIR="${TRAIN_DIR:-$RUN/train}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$RUN}"
COMPARE_DIR="${COMPARE_DIR:-$RUN/compare}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NUM_CROPS="${NUM_CROPS:-1}"
CROP_SEED="${CROP_SEED:-42}"

cd "$ROOT"
source .venv/bin/activate

mkdir -p "$TRAIN_DIR" "$COMPARE_DIR" "$CHECKPOINT_DIR"

_crop_grid_args=()
if [[ -n "${CROP_GRID:-}" ]]; then
  _crop_grid_args=(--grid "$CROP_GRID")
fi

if [[ "$NUM_CROPS" != "1" || -n "${CROP_GRID:-}" ]]; then
  if [[ "${BUILD_FULL_GT:-1}" != "1" ]]; then
    echo "NUM_CROPS>1 requires BUILD_FULL_GT=1 (full Gaussian GT)" >&2
    exit 1
  fi
  echo "== Gaussian GT full frame (sigma=$SIGMA_FULL) for multi-crop =="
  python "$SCRIPT_DIR/make_bayer_gaussian_gt.py" \
    --input "$SAMPLE" \
    --output "$TRAIN_DIR/gaussian_GT_full.npy" \
    --sigma "$SIGMA_FULL"
  echo "== training crop pairs (NUM_CROPS=$NUM_CROPS tile=$TILE) =="
  python "$SCRIPT_DIR/generate_training_crop_pairs.py" \
    --noisy "$SAMPLE" \
    --clean "$TRAIN_DIR/gaussian_GT_full.npy" \
    --train-dir "$TRAIN_DIR" \
    --tile-size "$TILE" \
    --num-crops "$NUM_CROPS" \
    --seed "$CROP_SEED" \
    "${_crop_grid_args[@]}"
  _SEQ_CLEAN="$TRAIN_DIR/gaussian_GT_full.npy"
  _SEQ_NOISY="$SAMPLE"
else
  echo "== center crop $TILE =="
  python - <<PY
import numpy as np
from pathlib import Path
src = np.load("$SAMPLE")
h, w = src.shape
ts = int("$TILE")
top = (h - ts) // 2
top -= top % 2
left = (w - ts) // 2
left -= left % 2
crop = src[top : top + ts, left : left + ts].astype(np.float16)
np.save("$TRAIN_DIR/noisy_input_crop.npy", crop)
print("noisy_input_crop", crop.shape)
PY

  echo "== Gaussian GT tile (sigma=$SIGMA) =="
  python "$SCRIPT_DIR/make_bayer_gaussian_gt.py" \
    --input "$TRAIN_DIR/noisy_input_crop.npy" \
    --output "$TRAIN_DIR/gaussian_GT_crop.npy" \
    --sigma "$SIGMA"
  _SEQ_CLEAN="$TRAIN_DIR/gaussian_GT_crop.npy"
  _SEQ_NOISY="$TRAIN_DIR/noisy_input_crop.npy"
fi

if [[ "$BUILD_FULL_GT" == "1" ]]; then
  echo "== Gaussian GT full frame (sigma=$SIGMA_FULL) — may take several minutes =="
  python "$SCRIPT_DIR/make_bayer_gaussian_gt.py" \
    --input "$SAMPLE" \
    --output "$TRAIN_DIR/gaussian_GT_full.npy" \
    --sigma "$SIGMA_FULL"
  echo "noisy full: $SAMPLE"
  echo "clean full: $TRAIN_DIR/gaussian_GT_full.npy"
fi

SEQ="$TRAIN_DIR/tile_train_seq"
mkdir -p "$SEQ"
if [[ "$NUM_CROPS" != "1" || -n "${CROP_GRID:-}" ]]; then
  python "$SCRIPT_DIR/make_static_noisy_sequence.py" \
    --clean "$_SEQ_CLEAN" \
    --output-dir "$SEQ" \
    --frames 1 \
    --noise-std 0 \
    --seed 0
  cp "$_SEQ_NOISY" "$SEQ/frame_0000.npy"
else
  cp "$_SEQ_CLEAN" "$SEQ/clean_mosaic.npy"
  cp "$_SEQ_NOISY" "$SEQ/frame_0000.npy"
fi

echo "== train on single tile pair (noisy -> Gaussian GT) =="
_train_argv=(
  python -m bayer_denoiser.train_static_noisy_sequence
  --sequence-dir "$SEQ"
  --output-dir "$CHECKPOINT_DIR"
  --epochs "${EPOCHS:-100}"
  --frames 1
)
if [[ "$NUM_CROPS" != "1" || -n "${CROP_GRID:-}" ]]; then
  _train_argv+=(--crop-size "$TILE")
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

_infer_input="$TRAIN_DIR/noisy_input_crop.npy"
_noisy_preview="$_infer_input"
_clean_preview="$TRAIN_DIR/gaussian_GT_crop.npy"
_infer_extra=()
if [[ "${FULL_INFER:-0}" == "1" || "${INFER_TILED:-0}" == "1" ]]; then
  _infer_input="$SAMPLE"
  _noisy_preview="$SAMPLE"
  _infer_tile="${INFER_TILE_SIZE:-$TILE}"
  _infer_extra=(--tiled --tile-size "$_infer_tile")
  if [[ -n "${INFER_OVERLAP:-}" ]]; then
    _infer_extra+=(--overlap "$INFER_OVERLAP")
  fi
  if [[ -f "$TRAIN_DIR/gaussian_GT_full.npy" ]]; then
    _clean_preview="$TRAIN_DIR/gaussian_GT_full.npy"
  fi
  echo "== tiled infer on full plate ($_infer_input, tile=$_infer_tile) =="
else
  echo "== official infer (should match training forward on tile) =="
fi

python -m bayer_denoiser.infer \
  --checkpoint "$CHECKPOINT_DIR/model.pt" \
  --input "$_infer_input" \
  --output "$COMPARE_DIR/denoised_infer.npy" \
  "${_infer_extra[@]}"

if [[ "${FULL_INFER:-0}" == "1" || "${INFER_TILED:-0}" == "1" ]]; then
  python -m bayer_denoiser.infer \
    --checkpoint "$CHECKPOINT_DIR/model.pt" \
    --input "$TRAIN_DIR/noisy_input_crop.npy" \
    --output "$COMPARE_DIR/denoised_infer_patch.npy"
  python -m bayer_denoiser.preview "$COMPARE_DIR/denoised_infer_patch.npy" \
    --write-mosaic "$COMPARE_DIR/denoised_patch.png"
fi

python -m bayer_denoiser.preview "$_noisy_preview" \
  --write-mosaic "$COMPARE_DIR/noisy.png"
python -m bayer_denoiser.preview "$_clean_preview" \
  --write-mosaic "$COMPARE_DIR/clean_gaussian_gt.png"
python -m bayer_denoiser.preview "$COMPARE_DIR/denoised_infer.npy" \
  --write-mosaic "$COMPARE_DIR/denoised.png"

echo "Done → train: $TRAIN_DIR  checkpoints: $CHECKPOINT_DIR  compare: $COMPARE_DIR"
