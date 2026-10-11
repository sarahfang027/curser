#!/usr/bin/env bash
# Bayer-safe Gaussian GT on the full mosaic (no center crop).
#
#   bash scripts/make_full_frame_gaussian_gt.sh
#   SIGMA=1.0 bash scripts/make_full_frame_gaussian_gt.sh /path/to/plate.npy
#
# Overrides:
#   RUN       — parent run folder (default: $ROOT/runs/static-noisy-singleframe-test)
#   TRAIN_DIR — training data root (default: $RUN/train); writes gaussian_GT_full.npy here
#   OUT       — alias for TRAIN_DIR when set (legacy); else use TRAIN_DIR

set -euo pipefail

ROOT="${BAYER_DENOISER_ROOT:-${BAYER_RGB_ROOT:-$HOME/Projects/bayer_denoiser}}"
SAMPLE="${1:-$ROOT/data/samples/A001C006_0101_4E4951.npy}"
SIGMA="${SIGMA:-1.0}"
RUN="${RUN:-$ROOT/runs/static-noisy-singleframe-test}"
TRAIN_DIR="${TRAIN_DIR:-${OUT:-$RUN/train}}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$ROOT"
source .venv/bin/activate

mkdir -p "$TRAIN_DIR"

echo "== full-frame Gaussian GT (sigma=$SIGMA) =="
python "$SCRIPT_DIR/make_bayer_gaussian_gt.py" \
  --input "$SAMPLE" \
  --output "$TRAIN_DIR/gaussian_GT_full.npy" \
  --sigma "$SIGMA"

echo "noisy (full): $SAMPLE"
echo "clean (full): $TRAIN_DIR/gaussian_GT_full.npy"

if [[ "${PREVIEW_FULL:-0}" == "1" ]]; then
  echo "== preview PNG (full frame — large file) =="
  python -m bayer_denoiser.preview "$TRAIN_DIR/gaussian_GT_full.npy" \
    --write-mosaic "$TRAIN_DIR/clean_gaussian_gt_full.png"
fi
