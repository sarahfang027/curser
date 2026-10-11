#!/usr/bin/env bash
# Basic bayer_denoise infer on A001C006 sample (center 512×512 crop).
# Run from ~/Projects/bayer_denoiser with venv active.

set -euo pipefail

ROOT="${BAYER_DENOISER_ROOT:-${BAYER_RGB_ROOT:-$HOME/Projects/bayer_denoiser}}"
SAMPLE="${1:-$ROOT/data/samples/A001C006_0101_4E4951.npy}"
OUT_DIR="$ROOT/runs/sample-tests"
CROP_NPY="$OUT_DIR/A001C006_center512.npy"
DENOISED_NPY="$OUT_DIR/A001C006_denoised512.npy"
CKPT="${CKPT:-$ROOT/runs/bayer-denoise-smoke/model.pt}"

cd "$ROOT"
source .venv/bin/activate

if [[ ! -f "$SAMPLE" ]]; then
  echo "Missing sample: $SAMPLE" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

echo "== preview stats =="
python -m bayer_denoiser.preview "$SAMPLE"

echo "== center 512 crop -> $CROP_NPY =="
python - <<PY
import numpy as np
from pathlib import Path

src = np.load("$SAMPLE")
h, w = src.shape
ts = 512
top = (h - ts) // 2
top -= top % 2
left = (w - ts) // 2
left -= left % 2
crop = src[top : top + ts, left : left + ts].astype(np.float16)
Path("$CROP_NPY").parent.mkdir(parents=True, exist_ok=True)
np.save("$CROP_NPY", crop)
print("crop shape", crop.shape, "min/max", float(crop.min()), float(crop.max()))
PY

echo "== noisy mosaic PNG =="
python -m bayer_denoiser.preview "$CROP_NPY" \
  --write-mosaic "$OUT_DIR/A001C006_noisy512.png"

if [[ ! -f "$CKPT" ]]; then
  echo "Checkpoint missing: $CKPT" >&2
  echo "Train with python -m bayer_denoiser.train_static_noisy_sequence (see docs/static-noisy-sequence.md) or set CKPT=..." >&2
  exit 1
fi

echo "== infer (bayer_denoise) =="
python -m bayer_denoiser.infer \
  --checkpoint "$CKPT" \
  --input "$CROP_NPY" \
  --output "$DENOISED_NPY"

echo "== denoised mosaic PNG =="
python -m bayer_denoiser.preview "$DENOISED_NPY" \
  --write-mosaic "$OUT_DIR/A001C006_denoised512.png"

echo "Done."
echo "  crop:     $CROP_NPY"
echo "  denoised: $DENOISED_NPY"
echo "  PNGs:     $OUT_DIR/A001C006_{noisy,denoised}512.png"
