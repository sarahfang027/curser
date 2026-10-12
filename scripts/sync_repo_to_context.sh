#!/usr/bin/env bash
# Mac repo → Context (Agent Store): docs/, scripts/, src/
#
#   cd ~/Projects/bayer_denoiser
#   bash scripts/sync_repo_to_context.sh
#
# src uses rsync --delete so Agent Store matches repo (removes old flat modules).
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

RSYNC_EXCLUDES=(--exclude '.DS_Store' --exclude '__pycache__' --exclude '*.pyc')

# Accidental copy during src/ reorg: whole package duplicated under src/core/core/
prune_bad_src_layout() {
  local root="$1"
  local label="$2"
  [[ -d "$root/src/core/core" ]] || return 0
  echo "Removing mistaken nested tree: $label/src/core/core"
  rm -rf "$root/src/core/core"
}

mirror_tree() {
  local src="$1"
  local dst="$2"
  local label="$3"
  local delete_flag="${4:-}"

  if [[ ! -d "$src" ]]; then
    echo "missing in repo: $src" >&2
    exit 1
  fi

  mkdir -p "$dst"
  if command -v rsync &>/dev/null; then
    if [[ "$delete_flag" == "--delete" ]]; then
      rsync -a --delete "${RSYNC_EXCLUDES[@]}" "$src/" "$dst/"
    else
      rsync -a "${RSYNC_EXCLUDES[@]}" "$src/" "$dst/"
    fi
  else
    rm -rf "${dst:?}/"*
    cp -R "$src/." "$dst/"
    find "$dst" -type d -name '__pycache__' -prune -exec rm -rf {} + 2>/dev/null || true
  fi
  echo "mirrored $label: $src → $dst"
}

echo "Repo:          $REPO"
echo "Context store: $STORE"
echo ""

prune_bad_src_layout "$REPO" "repo"

mirror_tree "$REPO/docs" "$STORE/docs" "docs" "--delete"
mirror_tree "$REPO/scripts" "$STORE/scripts" "scripts"
mirror_tree "$REPO/src" "$STORE/src" "src → files/src" "--delete"
prune_bad_src_layout "$STORE" "Context files"

if [[ -f "$REPO/requirements.txt" ]]; then
  cp "$REPO/requirements.txt" "$STORE/requirements.txt"
  echo "mirrored requirements.txt: $REPO → \$STORE/requirements.txt"
fi
# Single deps file: repo root → $STORE/requirements.txt only (not under scripts/)
rm -f "$STORE/scripts/requirements.txt"
prune_stray_requirements_copies() {
  # Old layout / cloud UI sometimes leaves copies at store-root (outside files/)
  if [[ "$STORE" == "$STORE_ROOT/files" ]]; then
    rm -f "$STORE_ROOT/requirements.txt"
    rm -f "$STORE_ROOT/scripts/requirements.txt"
  fi
}
prune_stray_requirements_copies
if [[ -f "$REPO/pyproject.toml" ]]; then
  cp "$REPO/pyproject.toml" "$STORE/pyproject.toml"
  echo "mirrored pyproject.toml: $REPO → \$STORE"
fi

# Store-root mirrors must match files/ (cloud UI + agents); --delete drops stale copies
if [[ "$STORE" == "$STORE_ROOT/files" ]]; then
  mirror_tree "$STORE/docs" "$STORE_ROOT/docs" "docs → store-root (copy of files/docs)" "--delete"
  mirror_tree "$STORE/src" "$STORE_ROOT/src" "src → store-root (copy of files/src)" "--delete"
  prune_bad_src_layout "$STORE_ROOT" "Context store-root"
fi

chmod +x "$STORE/scripts/"*.sh 2>/dev/null || true

echo ""
echo "Canonical Context trees: \$STORE/docs and \$STORE/src (files/docs, files/src)"
echo "docs: $(ls -1 "$STORE/docs"/*.md 2>/dev/null | wc -l | tr -d ' ') top-level .md files"
echo "src:  $(ls -1 "$STORE/src" 2>/dev/null | tr '\n' ' ')"
echo ""
echo "Done. Reopen Project → Context or wait for cloud sync."
echo "Reverse: bash scripts/sync_context_to_repo.sh"
