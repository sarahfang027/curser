#!/usr/bin/env bash
# 20-frame static noisy sequence from the Gaussian GT test run.
#
#   bash scripts/make_static_noisy_sequence.sh
#   NOISE_STD=0.03 FRAMES=10 SEED=1 bash scripts/make_static_noisy_sequence.sh
#   CLEAN=runs/gaussian-gt-test/train/gaussian_GT_full.npy bash scripts/make_static_noisy_sequence.sh
#
# Overrides:
#   RUN        — Gaussian GT run folder (default: $ROOT/runs/gaussian-gt-test); CLEAN crop lives under $RUN/train
#   RUNS       — top-level runs dir (default: $ROOT/runs)
#   TRAIN_DATA — training data root (default: $RUN/train); default CLEAN crop lives here
#   CLEAN      — clean mosaic .npy (default: $TRAIN_DATA/gaussian_GT_crop.npy)
#   PIPELINE_RUN — static-sequence run root (e.g. $RUNS/static-noisy-test); when set and OUT unset,
#                  writes to $PIPELINE_RUN/train/static_sequence (train/compare layout)
#   OUT        — sequence output dir (default: $RUNS/static_sequence, or $PIPELINE_RUN/train/static_sequence)

set -euo pipefail

ROOT="${BAYER_DENOISER_ROOT:-${BAYER_RGB_ROOT:-$HOME/Projects/bayer_denoiser}}"
RUNS="${RUNS:-$ROOT/runs}"
RUN="${RUN:-$RUNS/gaussian-gt-test}"
TRAIN_DATA="${TRAIN_DATA:-$RUN/train}"
CLEAN="${CLEAN:-$TRAIN_DATA/gaussian_GT_crop.npy}"
if [[ -n "${OUT:-}" ]]; then
  :
elif [[ -n "${PIPELINE_RUN:-}" ]]; then
  OUT="$PIPELINE_RUN/train/static_sequence"
else
  OUT="$RUNS/static_sequence"
fi
FRAMES="${FRAMES:-20}"
NOISE_STD="${NOISE_STD:-0.02}"
SEED="${SEED:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$ROOT"
if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi

python "$SCRIPT_DIR/make_static_noisy_sequence.py" \
  --clean "$CLEAN" \
  --output-dir "$OUT" \
  --frames "$FRAMES" \
  --noise-std "$NOISE_STD" \
  --seed "$SEED" \
  --write-manifest

echo "sequence: $OUT"
