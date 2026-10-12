#!/usr/bin/env bash
# Compare key Context markers between Agent Store and repo (Mac).
#
#   cd ~/Projects/bayer_denoiser
#   bash scripts/verify_context_pipeline_sync.sh
#
# Exit 0 when repo matches Context for DATA_ROOT pipeline markers; 1 otherwise.

set -euo pipefail

REPO="${1:-}"
if [[ -z "$REPO" ]]; then
  if git rev-parse --show-toplevel &>/dev/null; then
    REPO="$(git rev-parse --show-toplevel)"
  else
    REPO="$HOME/Projects/bayer_denoiser"
  fi
fi

STORE_ROOT="${CURSOR_AGENT_STORE:-$HOME/Library/Application Support/Cursor/AgentStores/cursor_agent_stores/bc-a64d420e-f2ca-403e-91f6-786d63f9c464}"
if [[ -d "$STORE_ROOT/files" ]]; then
  STORE="$STORE_ROOT/files"
else
  STORE="$STORE_ROOT"
fi

PIPE_REPO="$REPO/scripts/static_noisy_sequence_pipeline_test.sh"
PIPE_STORE="$STORE/scripts/static_noisy_sequence_pipeline_test.sh"

echo "Store: $STORE"
echo "Repo:  $REPO"
echo ""

_markers() {
  local f="$1"
  [[ -f "$f" ]] || { echo "missing:$f"; return; }
  local n
  n="$(grep -c _write_data_root_smoke_crop_npys "$f" 2>/dev/null || echo 0)"
  echo "write_data_root_smoke=$n"
  grep -q 'DATA_ROOT paired layout' "$f" && echo "data_root_skip_build=1" || echo "data_root_skip_build=0"
  wc -c <"$f" | awk '{print "bytes=" $1}'
}

echo "=== static_noisy_sequence_pipeline_test.sh ==="
echo "Context: $(_markers "$PIPE_STORE")"
echo "Repo:    $(_markers "$PIPE_REPO")"
echo ""

if [[ ! -f "$PIPE_STORE" ]]; then
  echo "Context pipeline missing — open Project in Cursor and wait for Context sync." >&2
  exit 1
fi

if grep -q _write_data_root_smoke_crop_npys "$PIPE_STORE" && ! grep -q _write_data_root_smoke_crop_npys "$PIPE_REPO" 2>/dev/null; then
  echo "Repo is BEHIND Context. Run:" >&2
  echo "  cp \"$PIPE_STORE\" \"$PIPE_REPO\"" >&2
  echo "  # or: bash scripts/sync_context_to_repo.sh" >&2
  exit 1
fi

if ! grep -q _write_data_root_smoke_crop_npys "$PIPE_REPO" 2>/dev/null; then
  echo "Repo pipeline missing DATA_ROOT smoke helpers (Context may also be stale)." >&2
  exit 1
fi

echo "Pipeline markers OK."
