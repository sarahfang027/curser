#!/usr/bin/env bash
# Static noisy sequence: build frames under train/, train at RUN root, infer + previews in compare/.
# Mirrors compare/ semantics of scripts/static_noisy_singleframe_pipeline_test.sh (canonical denoised_infer.npy + PNGs).
#
# Layout under RUN (default: $ROOT/runs/static-noisy-sequence-test):
#   train/
#     noisy_input_crop.png / gaussian_GT_crop.png — smoke crops (SMOKE_CROP_PNG=1)
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
#   CROP_Y, CROP_X — even top-left of one TILE window on full plate (NUM_CROPS=1; not with CROP_GRID)
#   CROP_SEED — RNG for random multi-crop placement (default 42)
#   DATA_ROOT — paired dataset on disk: $DATA_ROOT/GT/ + $DATA_ROOT/input/ (no static_sequence assemble)
#   GT_DIR, INPUT_DIR — override folders when using DATA_ROOT (default: $DATA_ROOT/GT, $DATA_ROOT/input)
#   SKIP_BUILD_GT=1 — require existing train/static_sequence/ (or prior build artifacts)
#   SKIP_SEQUENCE=1   — require existing train/static_sequence/
#   DEBUG_DENOISED_MOSAIC=1 — training-forward denoised_mosaic.npy → compare/
#   PLOT_HISTORY=1 (default) — after train, write compare/training_curves.png from history.json
#   FULL_INFER=1 (default) — tiled infer on full SAMPLE; FULL_INFER=0 — patch-only infer
#   INFER_TILED=1 — same as full-plate tiled infer (alias)
#   INFER_TILE_SIZE (default CROP_SIZE or TILE); INFER_OVERLAP optional
#   SMOKE_CROP_PNG — when 1, write crop QA as compare/*.png and drop crop .npy
#     (auto 1 for EPOCHS=100 + NUM_CROPS=1 + center, CROP_GRID=1x1, or CROP_Y/CROP_X)

set -euo pipefail

_smoke_crop_to_png() {
  local npy="$1" png="$2"
  if [[ ! -f "$npy" ]]; then
    return 0
  fi
  python -m bayer_denoiser.preview "$npy" --write-mosaic "$png"
  rm -f "$npy"
}

ROOT="${BAYER_DENOISER_ROOT:-${BAYER_RGB_ROOT:-$HOME/Projects/bayer_denoiser}}"
DATA_ROOT="${DATA_ROOT:-}"
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
EPOCHS="${EPOCHS:-8000}"

_use_coordinate_crop=0
if [[ -n "${CROP_Y:-}" || -n "${CROP_X:-}" ]]; then
  if [[ -z "${CROP_Y:-}" || -z "${CROP_X:-}" ]]; then
    echo "set both CROP_Y and CROP_X (even integers)" >&2
    exit 1
  fi
  if [[ -n "${CROP_GRID:-}" ]]; then
    echo "use CROP_Y/CROP_X or CROP_GRID, not both" >&2
    exit 1
  fi
  if [[ "$NUM_CROPS" != "1" ]]; then
    echo "CROP_Y/CROP_X requires NUM_CROPS=1" >&2
    exit 1
  fi
  _use_coordinate_crop=1
fi

if [[ -z "${SMOKE_CROP_PNG:-}" ]]; then
  _grid_lc="${CROP_GRID:-}"
  _grid_lc="${_grid_lc,,}"
  if [[ "$EPOCHS" == "100" && "$NUM_CROPS" == "1" && ( -z "${CROP_GRID:-}" || "$_grid_lc" == "1x1" || "$_use_coordinate_crop" == "1" ) ]]; then
    SMOKE_CROP_PNG=1
  else
    SMOKE_CROP_PNG=0
  fi
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$ROOT"
if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi

_use_data_root=0
DATA_ROOT_ABS=""
if [[ -n "$DATA_ROOT" ]]; then
  _use_data_root=1
  if [[ "$DATA_ROOT" = /* ]]; then
    DATA_ROOT_ABS="$DATA_ROOT"
  else
    DATA_ROOT_ABS="$ROOT/$DATA_ROOT"
  fi
  GT_DIR="${GT_DIR:-$DATA_ROOT_ABS/GT}"
  INPUT_DIR="${INPUT_DIR:-$DATA_ROOT_ABS/input}"
  if [[ ! -d "$GT_DIR" || ! -d "$INPUT_DIR" ]]; then
    echo "DATA_ROOT requires GT and input dirs: $GT_DIR  $INPUT_DIR" >&2
    exit 1
  fi
  _first_input="$(ls "$INPUT_DIR"/*.npy 2>/dev/null | sort | head -1 || true)"
  if [[ -z "$_first_input" ]]; then
    echo "no .npy under $INPUT_DIR" >&2
    exit 1
  fi
  if [[ $# -ge 1 && -f "$1" ]]; then
    SAMPLE="$1"
  else
    SAMPLE="${SAMPLE:-$_first_input}"
  fi
  SKIP_BUILD_GT="${SKIP_BUILD_GT:-1}"
  SKIP_SEQUENCE="${SKIP_SEQUENCE:-1}"
fi

mkdir -p "$TRAIN_DIR" "$COMPARE_DIR" "$CHECKPOINT_DIR" "$SEQ_DIR"

CLEAN_GT="$TRAIN_DIR/gaussian_GT_crop.npy"
NOISY_CROP="$TRAIN_DIR/noisy_input_crop.npy"

_use_full_plate_build=0
if [[ "$NUM_CROPS" != "1" || -n "${CROP_GRID:-}" || "$_use_coordinate_crop" == "1" ]]; then
  _use_full_plate_build=1
fi

TRAIN_CROP_Y="${CROP_Y:-}"
TRAIN_CROP_X="${CROP_X:-}"

if [[ "${SKIP_BUILD_GT:-0}" != "1" ]]; then
  if [[ "$_use_full_plate_build" == "1" ]]; then
    CLEAN_FULL="$TRAIN_DIR/gaussian_GT_full.npy"
    if [[ ! -f "$CLEAN_FULL" ]]; then
      echo "== Gaussian GT full frame (sigma=$SIGMA) for multi-crop =="
      python "$SCRIPT_DIR/make_bayer_gaussian_gt.py" \
        --input "$SAMPLE" \
        --output "$CLEAN_FULL" \
        --sigma "$SIGMA"
    fi
    if [[ "$_use_coordinate_crop" == "1" ]]; then
      echo "== training crop at (y=$CROP_Y, x=$CROP_X) tile=$TILE =="
    else
      echo "== training crop pairs (NUM_CROPS=$NUM_CROPS tile=$TILE) =="
    fi
    _gen_crop_argv=(
      python "$SCRIPT_DIR/generate_training_crop_pairs.py"
      --noisy "$SAMPLE"
      --clean "$CLEAN_FULL"
      --train-dir "$TRAIN_DIR"
      --tile-size "$TILE"
      --num-crops "$NUM_CROPS"
      --seed "$CROP_SEED"
    )
    if [[ -n "${CROP_GRID:-}" ]]; then
      _gen_crop_argv+=(--grid "$CROP_GRID")
    fi
    if [[ "$_use_coordinate_crop" == "1" ]]; then
      _gen_crop_argv+=(--crop-y "$CROP_Y" --crop-x "$CROP_X")
    fi
    "${_gen_crop_argv[@]}"
    _SEQ_CLEAN="$CLEAN_FULL"
    if [[ -z "$TRAIN_CROP_Y" && -f "$TRAIN_DIR/crops/manifest.json" ]]; then
      read -r TRAIN_CROP_Y TRAIN_CROP_X < <(
        python -c "import json; c=json.load(open('$TRAIN_DIR/crops/manifest.json'))['crops'][0]; print(c['y'], c['x'])"
      )
    fi
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
  if [[ -f "$SEQ_DIR/clean_mosaic.npy" ]]; then
    _SEQ_CLEAN="$SEQ_DIR/clean_mosaic.npy"
  elif [[ -f "$CLEAN_GT" ]]; then
    _SEQ_CLEAN="$CLEAN_GT"
  else
    echo "missing train data (SKIP_BUILD_GT=1): need $SEQ_DIR/clean_mosaic.npy" >&2
    exit 1
  fi
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
  if [[ "$_use_full_plate_build" == "1" && -f "$SEQ_DIR/clean_mosaic.npy" ]]; then
    rm -f "$TRAIN_DIR/gaussian_GT_full.npy"
  fi
fi

_infer_input=""
_train_crop_args=()
if [[ "$CROP_SIZE" != "0" ]]; then
  _train_crop_args=(--crop-size "$CROP_SIZE")
fi

echo "== train on static sequence =="
_train_argv=(
  python -m bayer_denoiser.train_static_noisy_sequence
  --output-dir "$CHECKPOINT_DIR"
  --epochs "$EPOCHS"
)
if [[ "$_use_data_root" == "1" ]]; then
  _train_argv+=(--data-root "$DATA_ROOT_ABS")
  if [[ -f "$TRAIN_DIR/crops/manifest.json" ]]; then
    _train_argv+=(--crops-manifest "$TRAIN_DIR/crops/manifest.json")
  fi
else
  _train_argv+=(--sequence-dir "$SEQ_DIR")
fi
if [[ ${#_train_crop_args[@]} -gt 0 ]]; then
  _train_argv+=("${_train_crop_args[@]}")
fi
if [[ -n "$TRAIN_CROP_Y" && -n "$TRAIN_CROP_X" ]]; then
  _train_argv+=(--crop-y "$TRAIN_CROP_Y" --crop-x "$TRAIN_CROP_X")
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
if [[ "$_use_data_root" == "1" ]]; then
  _frame0="$(ls "$INPUT_DIR"/*.npy 2>/dev/null | sort | head -1)"
  if [[ -f "$GT_DIR/clean_mosaic.npy" ]]; then
    _seq_clean="$GT_DIR/clean_mosaic.npy"
  else
    _seq_clean="$(ls "$GT_DIR"/*.npy 2>/dev/null | sort | head -1)"
  fi
else
  _frame0="$SEQ_DIR/frame_0000.npy"
  _seq_clean="$SEQ_DIR/clean_mosaic.npy"
fi
if [[ ! -f "$_frame0" ]]; then
  echo "missing noisy frame: $_frame0" >&2
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
ty, tx = "$TRAIN_CROP_Y", "$TRAIN_CROP_X"
if ty and tx:
    top, left = int(ty), int(tx)
else:
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
if [[ "${FULL_INFER:-1}" == "1" || "${INFER_TILED:-0}" == "1" ]]; then
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
ty, tx = "$TRAIN_CROP_Y", "$TRAIN_CROP_X"
if ty and tx:
    top, left = int(ty), int(tx)
else:
    top = (h - ts) // 2
    top -= top % 2
    left = (w - ts) // 2
    left -= left % 2
crop = src[top : top + ts, left : left + ts].astype(np.float16)
out = Path("$COMPARE_DIR/infer_input_crop.npy")
out.parent.mkdir(parents=True, exist_ok=True)
np.save(out, crop)
print("infer_input_crop", crop.shape, "at", top, left)
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

_clean_preview="${_seq_clean:-$SEQ_DIR/clean_mosaic.npy}"
if [[ "$CROP_SIZE" != "0" ]]; then
  _clean_crop="$COMPARE_DIR/clean_mosaic_crop.npy"
  python - <<PY
import numpy as np
from pathlib import Path
clean = np.load("$_clean_preview")
h, w = clean.shape
cs = int("$CROP_SIZE")
ty, tx = "$TRAIN_CROP_Y", "$TRAIN_CROP_X"
if ty and tx:
    top, left = int(ty), int(tx)
else:
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

if [[ "$SMOKE_CROP_PNG" == "1" ]]; then
  echo "== smoke crop previews (PNG, drop crop .npy) =="
  if [[ -f "$TRAIN_DIR/crops/noisy_000.npy" ]]; then
    _smoke_crop_to_png "$TRAIN_DIR/crops/noisy_000.npy" "$TRAIN_DIR/noisy_input_crop.png"
    _smoke_crop_to_png "$TRAIN_DIR/crops/gaussian_GT_000.npy" "$TRAIN_DIR/gaussian_GT_crop.png"
    rm -f "$TRAIN_DIR/crops/noisy_000.npy" "$TRAIN_DIR/crops/gaussian_GT_000.npy"
  else
    _smoke_crop_to_png "$NOISY_CROP" "$TRAIN_DIR/noisy_input_crop.png"
    _smoke_crop_to_png "$CLEAN_GT" "$TRAIN_DIR/gaussian_GT_crop.png"
  fi
  _smoke_crop_to_png "$COMPARE_DIR/infer_input_crop.npy" "$COMPARE_DIR/infer_input_crop.png"
  _smoke_crop_to_png "$COMPARE_DIR/clean_mosaic_crop.npy" "$COMPARE_DIR/clean_mosaic_crop.png"
  _smoke_crop_to_png "$COMPARE_DIR/denoised_infer_patch.npy" "$COMPARE_DIR/denoised_infer_patch.png"
fi

# Redundant after static_sequence/ exists (trainer reads sequence only)
rm -f \
  "$TRAIN_DIR/gaussian_GT_crop.npy" \
  "$TRAIN_DIR/noisy_input_crop.npy" \
  "$TRAIN_DIR/gaussian_GT_full.npy"

echo "Done → train: $TRAIN_DIR  sequence: $SEQ_DIR  checkpoints: $CHECKPOINT_DIR  compare: $COMPARE_DIR"
