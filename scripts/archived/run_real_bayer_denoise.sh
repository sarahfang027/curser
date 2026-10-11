#!/usr/bin/env bash
# Template: QA, optional train, and/or infer for real Bayer plates (bayer_denoise).
# Copy from Project Context via sync_context_to_repo.sh — run on Mac only.
#
# Infer-only (needs CHECKPOINT + input):
#   export CHECKPOINT=runs/bayer-denoise-plates/model.pt
#   export INPUT_NPY=runs/infer_in/center512.npy   # or set BAYER_DIR for .raw under one folder
#   export OUTPUT_DIR=runs/infer_out
#   bash scripts/run_real_bayer_denoise.sh
#
# Primary Phase B layout (noisy + clean .raw) — QA only until train.py supports it:
#   export NOISY_DIR=data/noisy
#   export CLEAN_DIR=data/clean
#   export RUN_QA=1 RUN_TRAIN=0
#   bash scripts/run_real_bayer_denoise.sh
#   # Then: export aligned .npy crops and run python -m bayer_denoiser.train_static_noisy_sequence
#
# Legacy stock train (clean bayer + rgb alignment + --noise-std):
#   export BAYER_DIR=data/bayer
#   export RGB_DIR=data/rgb          # EXR/PNG — crop alignment only; not the denoise target
#   # export PAIRS_CSV=pairs.csv     # columns bayer,rgb — not noisy,clean
#   export OUTPUT_DIR=runs/bayer-denoise-plates
#   export RUN_TRAIN=1
#   bash scripts/run_real_bayer_denoise.sh
#
# Optional: RUN_QA=0 to skip validate/visualize; CROP_SIZE=512 for infer crop from INPUT_NPY parent;
# EPOCHS, IMAGE_SIZE, CROPS_PER_FRAME, NOISE_STD, DEVICE, RAW_* mirror training-user-guide.

set -euo pipefail

ROOT="${BAYER_DENOISER_ROOT:-${BAYER_RGB_ROOT:-$HOME/Projects/bayer_denoiser}}"
cd "$ROOT"
source .venv/bin/activate

RUN_QA="${RUN_QA:-1}"
RUN_TRAIN="${RUN_TRAIN:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/bayer-denoise-real}"
BAYER_DIR="${BAYER_DIR:-}"
RGB_DIR="${RGB_DIR:-}"
NOISY_DIR="${NOISY_DIR:-}"
CLEAN_DIR="${CLEAN_DIR:-}"
PAIRS_CSV="${PAIRS_CSV:-}"
CHECKPOINT="${CHECKPOINT:-}"
INPUT_NPY="${INPUT_NPY:-}"
BAYER_DIR_INFER="${BAYER_DIR_INFER:-$BAYER_DIR}"
CROP_SIZE="${CROP_SIZE:-0}"
DEVICE="${DEVICE:-}"
EPOCHS="${EPOCHS:-30}"
IMAGE_SIZE="${IMAGE_SIZE:-256}"
CROPS_PER_FRAME="${CROPS_PER_FRAME:-16}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NOISE_STD="${NOISE_STD:-0.02}"

device_args=()
if [[ -n "$DEVICE" ]]; then
  device_args=(--device "$DEVICE")
fi

raw_args=()
[[ -n "${RAW_DTYPE:-}" ]] && raw_args+=(--raw-dtype "$RAW_DTYPE")
[[ -n "${RAW_WIDTH:-}" ]] && raw_args+=(--raw-width "$RAW_WIDTH")
[[ -n "${RAW_HEIGHT:-}" ]] && raw_args+=(--raw-height "$RAW_HEIGHT")
[[ -n "${BLACK_LEVEL:-}" ]] && raw_args+=(--black-level "$BLACK_LEVEL")
[[ -n "${WHITE_LEVEL:-}" ]] && raw_args+=(--white-level "$WHITE_LEVEL")

pair_args=()
if [[ -n "$PAIRS_CSV" ]]; then
  pair_args=(--pairs-csv "$PAIRS_CSV")
elif [[ -n "$BAYER_DIR" && -n "$RGB_DIR" ]]; then
  pair_args=(--bayer-dir "$BAYER_DIR" --rgb-dir "$RGB_DIR")
fi

if [[ "$RUN_QA" == "1" ]]; then
  if [[ -n "$NOISY_DIR" ]]; then
    echo "== validate_raw (noisy): $NOISY_DIR =="
    python -m bayer_denoiser.archived.qa.validate_raw --dir "$NOISY_DIR"
  fi
  if [[ -n "$CLEAN_DIR" ]]; then
    echo "== validate_raw (clean): $CLEAN_DIR =="
    python -m bayer_denoiser.archived.qa.validate_raw --dir "$CLEAN_DIR"
  fi
  if [[ -n "$NOISY_DIR" && -n "$CLEAN_DIR" ]]; then
    echo "Note: stock train does not load NOISY_DIR+CLEAN_DIR yet; use python -m bayer_denoiser.train_static_noisy_sequence on .npy crops."
  fi
  if [[ -n "$BAYER_DIR" ]]; then
    echo "== validate_raw: $BAYER_DIR =="
    python -m bayer_denoiser.archived.qa.validate_raw --dir "$BAYER_DIR"
  fi
  if [[ ${#pair_args[@]} -gt 0 ]]; then
    echo "== validate_pairs =="
    python -m bayer_denoiser.archived.qa.validate_pairs "${pair_args[@]}"
    echo "== visualize_crops (pair 0) =="
    mkdir -p runs/crop_visual_check
    python -m bayer_denoiser.archived.qa.visualize_crops \
      "${pair_args[@]}" \
      --pair-index 0 \
      --crop-size "${VIS_CROP_SIZE:-256}" \
      --num-crops "${VIS_NUM_CROPS:-8}" \
      --seed 42 \
      --output-dir runs/crop_visual_check
  fi
fi

if [[ "$RUN_TRAIN" == "1" ]]; then
  if [[ -n "$NOISY_DIR" || -n "$CLEAN_DIR" ]]; then
    echo "RUN_TRAIN with NOISY_DIR/CLEAN_DIR is not implemented in stock bayer_denoiser.train." >&2
    echo "Set RUN_TRAIN=0 and use python -m bayer_denoiser.train_static_noisy_sequence on aligned .npy crops." >&2
    exit 1
  fi
  if [[ ${#pair_args[@]} -eq 0 ]]; then
    echo "Train requires BAYER_DIR+RGB_DIR or PAIRS_CSV (legacy Phase B)." >&2
    exit 1
  fi
  mkdir -p "$OUTPUT_DIR"
  echo "== train bayer_denoise -> $OUTPUT_DIR =="
  python -m bayer_denoiser.archived.core.train \
    --model bayer_denoise \
    "${pair_args[@]}" \
    "${raw_args[@]}" \
    --image-size "$IMAGE_SIZE" \
    --crops-per-frame "$CROPS_PER_FRAME" \
    --batch-size "$BATCH_SIZE" \
    --noise-std "$NOISE_STD" \
    --epochs "$EPOCHS" \
    --output-dir "$OUTPUT_DIR" \
    "${device_args[@]}"
fi

if [[ -n "$CHECKPOINT" ]]; then
  mkdir -p "$OUTPUT_DIR"
  infer_input=""
  infer_output=""

  if [[ -n "$INPUT_NPY" ]]; then
    infer_input="$INPUT_NPY"
    if [[ "$CROP_SIZE" != "0" ]]; then
      crop_path="$OUTPUT_DIR/infer_crop_${CROP_SIZE}.npy"
      echo "== center crop $CROP_SIZE from $INPUT_NPY =="
      python - <<PY
import numpy as np
from pathlib import Path
src = np.load("$INPUT_NPY")
ts = int("$CROP_SIZE")
h, w = src.shape
top = (h - ts) // 2; top -= top % 2
left = (w - ts) // 2; left -= left % 2
crop = src[top:top+ts, left:left+ts].astype(np.float16)
Path("$OUTPUT_DIR").mkdir(parents=True, exist_ok=True)
np.save("$crop_path", crop)
print("saved", crop.shape)
PY
      infer_input="$crop_path"
    fi
    base="$(basename "$infer_input" .npy)"
    infer_output="$OUTPUT_DIR/${base}_denoised.npy"
  elif [[ -n "$BAYER_DIR_INFER" ]]; then
    first_raw="$(find "$BAYER_DIR_INFER" -maxdepth 1 -name '*.raw' | head -1 || true)"
    if [[ -z "$first_raw" ]]; then
      echo "Infer needs INPUT_NPY or a .raw under BAYER_DIR_INFER ($BAYER_DIR_INFER)" >&2
      exit 1
    fi
    infer_input="$first_raw"
    stem="$(basename "$first_raw" .raw)"
    infer_output="$OUTPUT_DIR/${stem}_denoised.npy"
  else
    echo "Infer needs CHECKPOINT and INPUT_NPY or BAYER_DIR/BAYER_DIR_INFER" >&2
    exit 1
  fi

  echo "== infer: $infer_input -> $infer_output =="
  python -m bayer_denoiser.infer \
    --checkpoint "$CHECKPOINT" \
    --input "$infer_input" \
    --output "$infer_output" \
    "${raw_args[@]}" \
    "${device_args[@]}"
  echo "Done: $infer_output"
fi

if [[ "$RUN_TRAIN" != "1" && -z "$CHECKPOINT" ]]; then
  echo "Nothing to do: set RUN_TRAIN=1 and pair dirs, or CHECKPOINT for infer." >&2
  exit 1
fi
