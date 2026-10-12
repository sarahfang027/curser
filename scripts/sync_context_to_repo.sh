#!/usr/bin/env bash
# Context (Agent Store) → Mac repo: docs/, scripts/, internal/ → docs/internal/
#
# Run on the Mac after Cursor has synced Project Context to the local Agent Store:
#
#   cd ~/Projects/bayer_denoiser
#   bash scripts/sync_context_to_repo.sh
#   bash scripts/sync_context_to_repo.sh /path/to/bayer_denoiser
#
# Reverse: bash scripts/sync_repo_to_context.sh

set -euo pipefail

# zsh (default on Mac) does not treat "# ..." as a comment unless `setopt interactivecomments`;
# a trailing "# note" can be passed as $1 — ignore comment-like args.
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
  echo "Agent Store not found (expected files/ under): $STORE_ROOT" >&2
  echo "Open the Bayer Denoising CNN project in Cursor so Context syncs, then retry." >&2
  exit 1
fi

if [[ ! -d "$REPO" ]]; then
  echo "Repo not found: $REPO" >&2
  if [[ -d "$HOME/Projects/bayer_rgb" ]]; then
    echo "Found old folder ~/Projects/bayer_rgb — rename first: mv ~/Projects/bayer_rgb ~/Projects/bayer_denoiser" >&2
  fi
  exit 1
fi

SYNCED=0
SKIPPED=0

count_files() {
  local dir="$1"
  if [[ -d "$dir" ]]; then
    find "$dir" -type f ! -name '.DS_Store' 2>/dev/null | wc -l | tr -d ' '
  else
    echo 0
  fi
}

sync_tree() {
  local src_sub="$1"
  local dst_sub="$2"
  local src="$STORE/$src_sub"
  local dst="$REPO/$dst_sub"

  if [[ ! -d "$src" ]]; then
    echo "skip (missing in Context): $src_sub" >&2
    SKIPPED=$((SKIPPED + 1))
    return 0
  fi

  mkdir -p "$dst"
  local n_before
  n_before="$(count_files "$dst")"

  if command -v rsync &>/dev/null; then
    if [[ "$src_sub" == "scripts" ]]; then
      rsync -a --checksum --exclude '.DS_Store' "$src/" "$dst/"
    else
      rsync -a --exclude '.DS_Store' "$src/" "$dst/"
    fi
  else
    cp -R "$src/." "$dst/"
  fi

  local n_after
  n_after="$(count_files "$dst")"
  echo "synced $src_sub → $dst_sub ($n_after files under destination; was $n_before)"
  SYNCED=$((SYNCED + 1))
}

echo "Context store: $STORE"
echo "Repo:          $REPO"
echo ""

sync_tree docs docs
sync_tree scripts scripts
if [[ -d "$STORE/src" ]]; then
  sync_tree src src
fi
sync_tree internal docs/internal

ARCHIVE_DOCS="internal/archived/context-docs-2026-09-29"
if [[ -d "$STORE/$ARCHIVE_DOCS" ]]; then
  sync_tree "$ARCHIVE_DOCS" "docs/internal/archived/context-docs-2026-09-29"
fi

if compgen -G "$REPO/scripts/"*.sh >/dev/null; then
  chmod +x "$REPO/scripts/"*.sh
  echo "chmod +x: $REPO/scripts/*.sh"
fi

echo ""
echo "Summary: $SYNCED tree(s) synced, $SKIPPED skipped (missing in Context)."
echo "Next: cd $REPO && git status"
