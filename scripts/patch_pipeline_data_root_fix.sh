#!/usr/bin/env bash
# One-shot fix when DATA_ROOT smoke fails with:
#   missing train data (SKIP_BUILD_GT=1): need .../train/static_sequence/clean_mosaic.npy
#
#   cd ~/Projects/bayer_denoiser
#   bash scripts/patch_pipeline_data_root_fix.sh
#
# Safe to re-run (no-op if already patched).

set -euo pipefail

ROOT="${BAYER_DENOISER_ROOT:-${BAYER_RGB_ROOT:-$HOME/Projects/bayer_denoiser}}"
if git -C "$ROOT" rev-parse --show-toplevel &>/dev/null; then
  ROOT="$(git -C "$ROOT" rev-parse --show-toplevel)"
fi
PIPE="$ROOT/scripts/static_noisy_sequence_pipeline_test.sh"

if [[ ! -f "$PIPE" ]]; then
  echo "not found: $PIPE" >&2
  exit 1
fi

if grep -q 'or set DATA_ROOT=.../GT+input/' "$PIPE"; then
  echo "already patched: $PIPE"
  exit 0
fi

python3 - "$PIPE" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text()

helper = '''
_resolve_gt_clean() {
  local gt_dir="$1"
  if [[ -f "$gt_dir/clean_mosaic.npy" ]]; then
    echo "$gt_dir/clean_mosaic.npy"
    return 0
  fi
  ls "$gt_dir"/*.npy 2>/dev/null | sort | head -1 || true
}

'''

if "_resolve_gt_clean" not in text:
    needle = 'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n\n'
    if needle not in text:
        raise SystemExit("pipeline layout changed; cannot insert _resolve_gt_clean")
    text = text.replace(needle, needle + helper, 1)

old = '''else
  if [[ -f "$SEQ_DIR/clean_mosaic.npy" ]]; then
    _SEQ_CLEAN="$SEQ_DIR/clean_mosaic.npy"
  elif [[ -f "$CLEAN_GT" ]]; then
    _SEQ_CLEAN="$CLEAN_GT"
  else
    echo "missing train data (SKIP_BUILD_GT=1): need $SEQ_DIR/clean_mosaic.npy" >&2
    exit 1
  fi
fi'''

new = '''else
  if [[ "$_use_data_root" == "1" ]]; then
    _SEQ_CLEAN="$(_resolve_gt_clean "$GT_DIR")"
    if [[ -z "$_SEQ_CLEAN" || ! -f "$_SEQ_CLEAN" ]]; then
      echo "missing clean .npy under $GT_DIR (DATA_ROOT paired layout)" >&2
      exit 1
    fi
  elif [[ -f "$SEQ_DIR/clean_mosaic.npy" ]]; then
    _SEQ_CLEAN="$SEQ_DIR/clean_mosaic.npy"
  elif [[ -f "$CLEAN_GT" ]]; then
    _SEQ_CLEAN="$CLEAN_GT"
  else
    echo "missing train data (SKIP_BUILD_GT=1): need $SEQ_DIR/clean_mosaic.npy or set DATA_ROOT=.../GT+input/" >&2
    exit 1
  fi
fi'''

if old not in text:
    raise SystemExit("SKIP_BUILD_GT block not found; pipeline may already be patched or too old/new")

path.write_text(text.replace(old, new, 1))
print("patched:", path)
PY

echo "Verify: grep 'DATA_ROOT paired layout' $PIPE"
echo "Re-run your DATA_ROOT smoke command."
