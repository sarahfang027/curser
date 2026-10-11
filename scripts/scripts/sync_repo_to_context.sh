#!/usr/bin/env bash
# Mac repo → Context (Agent Store): docs/ and scripts/
#
#   cd ~/Projects/bayer_denoiser
#   bash scripts/sync_repo_to_context.sh
#   bash scripts/sync_repo_to_context.sh /path/to/bayer_denoiser
#
# Does not touch notes.md / preferences.md at the Agent Store root.
# Reverse: bash scripts/sync_context_to_repo.sh

set -euo pipefail

DEFAULT_REPO="$HOME/Projects/bayer_denoiser"
REPO="$DEFAULT_REPO"
if [[ $# -gt 0 && "$1" != \#* ]]; then
  REPO="$1"
elif git rev-parse --show-toplevel &>/dev/null; then
  REPO="$(git rev-parse --show-toplevel)"
fi
STORE_ROOT="${CURSOR_AGENT_STORE:-$HOME/Library/Application Support/Cursor/AgentStores/cursor_agent_stores/bc-a64d420e-f2ca-403e-91f6-786d63f9c464}"

if [[ -d "$STORE_ROOT/files" ]]; then
  STORE="$STORE_ROOT/files"
elif [[ -d "$STORE_ROOT/docs" || -d "$STORE_ROOT/scripts" ]]; then
  STORE="$STORE_ROOT"
else
  echo "Agent Store not found under: $STORE_ROOT" >&2
  exit 1
fi

mirror_tree() {
  local src="$1"
  local dst="$2"
  local label="$3"

  if [[ ! -d "$src" ]]; then
    echo "missing in repo: $src" >&2
    exit 1
  fi

  mkdir -p "$dst"
  if command -v rsync &>/dev/null; then
    rsync -a --exclude '.DS_Store' "$src/" "$dst/"
  else
    cp -R "$src/." "$dst/"
  fi
  echo "mirrored $label: $src → $dst"
}

echo "Repo:          $REPO"
echo "Context store: $STORE"
echo ""

mirror_tree "$REPO/docs" "$STORE/docs" "docs"
mirror_tree "$REPO/scripts" "$STORE/scripts" "scripts"

chmod +x "$STORE/scripts/"*.sh 2>/dev/null || true

echo ""
echo "Done. Reopen Project → Context or wait for cloud sync."
echo "Reverse: bash scripts/sync_context_to_repo.sh"
