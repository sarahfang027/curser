#!/usr/bin/env bash
# Static noisy single-frame pipeline test: Gaussian GT tile (+ optional full) → train (frames=1) → infer + PSNR.
# Pipeline plumbing smoke — Gaussian GT is not a true noise-free reference. Multi-frame: static_noisy_sequence_pipeline_test.sh
#
# Layout under RUN (default: $ROOT/runs/static-noisy-singleframe-test):
#   train/   — data only: noisy_input_crop.npy, gaussian_GT_*.npy, tile_train_seq/
#   compare/ — denoised_infer.npy + preview PNGs (canonical); optional denoised_mosaic.npy when DEBUG_DENOISED_MOSAIC=1
#   (multi-frame static sequence: scripts/static_noisy_sequence_pipeline_test.sh → runs/static-noisy-test/train/static_sequence/)
#   RUN root — model.pt, model_epoch*.pt, history.json (outside train/)
#
# Overrides:
#   TRAIN_DIR      — training data only (default: $RUN/train)
#   CHECKPOINT_DIR — model.pt, model_epoch*.pt, history.json (default: $RUN)
#   COMPARE_DIR — infer / preview artifacts (default: $RUN/compare)
#   RUN         — parent run folder (default: $ROOT/runs/static-noisy-singleframe-test)
#   DEBUG_DENOISED_MOSAIC=1 — pass --write-denoised-mosaic to train; move denoised_mosaic.npy to compare/

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

cd "$ROOT"
source .venv/bin/activate

mkdir -p "$TRAIN_DIR" "$COMPARE_DIR" "$CHECKPOINT_DIR"

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
cp "$TRAIN_DIR/gaussian_GT_crop.npy" "$SEQ/clean_mosaic.npy"
cp "$TRAIN_DIR/noisy_input_crop.npy" "$SEQ/frame_0000.npy"

echo "== train on single tile pair (noisy -> Gaussian GT) =="
_train_argv=(
  python -m bayer_denoiser.train_static_noisy_sequence
  --sequence-dir "$SEQ"
  --output-dir "$CHECKPOINT_DIR"
  --epochs "${EPOCHS:-80}"
  --frames 1
)
if [[ "${DEBUG_DENOISED_MOSAIC:-0}" == "1" ]]; then
  _train_argv+=(--write-denoised-mosaic)
fi
"${_train_argv[@]}"

if [[ -f "$CHECKPOINT_DIR/denoised_mosaic.npy" ]]; then
  mv -f "$CHECKPOINT_DIR/denoised_mosaic.npy" "$COMPARE_DIR/denoised_mosaic.npy"
fi

echo "== official infer (should match training forward) =="
python -m bayer_denoiser.infer \
  --checkpoint "$CHECKPOINT_DIR/model.pt" \
  --input "$TRAIN_DIR/noisy_input_crop.npy" \
  --output "$COMPARE_DIR/denoised_infer.npy"

python -m bayer_denoiser.preview "$TRAIN_DIR/noisy_input_crop.npy" \
  --write-mosaic "$COMPARE_DIR/noisy.png"
python -m bayer_denoiser.preview "$TRAIN_DIR/gaussian_GT_crop.npy" \
  --write-mosaic "$COMPARE_DIR/clean_gaussian_gt.png"
python -m bayer_denoiser.preview "$COMPARE_DIR/denoised_infer.npy" \
  --write-mosaic "$COMPARE_DIR/denoised.png"

echo "Done → train: $TRAIN_DIR  checkpoints: $CHECKPOINT_DIR  compare: $COMPARE_DIR"
