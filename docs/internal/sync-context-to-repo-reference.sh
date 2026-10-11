#!/usr/bin/env bash
# Reference copy (2026-10-08): known-good sync_context_to_repo.sh — copy to repo if edits break syntax.
#   cp docs/internal/sync-context-to-repo-reference.sh scripts/sync_context_to_repo.sh
# Context (Agent Store) → Mac repo: docs/, scripts/, src/, requirements.txt, internal/ → docs/internal/
#
#   cd ~/Projects/bayer_denoiser
#   bash scripts/sync_context_to_repo.sh
#
# Reverse: bash scripts/sync_repo_to_context.sh

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
  echo "Agent Store not found (expected files/ under): $STORE_ROOT" >&2
  exit 1
fi

if [[ ! -d "$REPO" ]]; then
  echo "Repo not found: $REPO" >&2
  exit 1
fi

RSYNC_EXCLUDES=(--exclude '.DS_Store' --exclude '__pycache__' --exclude '*.pyc')

prune_bad_src_layout() {
  local root="$1"
  local label="$2"
  [[ -d "$root/src/core/core" ]] || return 0
  echo "Removing mistaken nested tree: $label/src/core/core"
  rm -rf "$root/src/core/core"
}

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

sync_tree_from() {
  local src="$1"
  local dst="$2"
  local label="$3"
  local delete_flag="${4:-}"

  if [[ ! -d "$src" ]]; then
    echo "skip (missing in Context): $label" >&2
    SKIPPED=$((SKIPPED + 1))
    return 0
  fi

  mkdir -p "$dst"
  local n_before n_after
  n_before="$(count_files "$dst")"

  if command -v rsync &>/dev/null; then
    if [[ "$delete_flag" == "--delete" ]]; then
      rsync -a --delete "${RSYNC_EXCLUDES[@]}" "$src/" "$dst/"
    else
      rsync -a "${RSYNC_EXCLUDES[@]}" "$src/" "$dst/"
    fi
  else
    if [[ "$delete_flag" == "--delete" ]]; then
      rm -rf "${dst:?}/"*
    fi
    cp -R "$src/." "$dst/"
    find "$dst" -type d -name '__pycache__' -prune -exec rm -rf {} + 2>/dev/null || true
  fi

  n_after="$(count_files "$dst")"
  echo "synced $label ($n_after files under destination; was $n_before)"
  SYNCED=$((SYNCED + 1))
}

sync_tree() {
  sync_tree_from "$STORE/$1" "$REPO/$2" "$1 → $2"
}

resolve_src_dir() {
  # Canonical package tree is always $STORE/src (files/src on Mac). Do not fall back to
  # store-root src — it can lag and resurrect deleted modules (e.g. core/train.py).
  if [[ -d "$STORE/src" ]] && [[ -n "$(find "$STORE/src" -mindepth 1 -maxdepth 1 2>/dev/null | head -1)" ]]; then
    echo "$STORE/src"
  else
    echo ""
  fi
}

echo "Context store: $STORE"
echo "Store root:    $STORE_ROOT"
echo "Repo:          $REPO"
echo "Trees:         docs, scripts, src, requirements.txt, internal → docs/internal"
echo ""

# Mistaken nested Agent Store path (IDE sometimes opens files/files/docs/…)
NESTED="$STORE/files"
if [[ -d "$NESTED" ]]; then
  echo "NOTE: merging nested Context \$STORE/files/ → \$STORE (docs, scripts, src) before repo sync…" >&2
  for tree in docs scripts src; do
    if [[ -d "$NESTED/$tree" ]]; then
      mkdir -p "$STORE/$tree"
      if command -v rsync &>/dev/null; then
        rsync -a "${RSYNC_EXCLUDES[@]}" "$NESTED/$tree/" "$STORE/$tree/"
      else
        cp -R "$NESTED/$tree/." "$STORE/$tree/"
      fi
      echo "  merged $tree: \$STORE/files/$tree → \$STORE/$tree" >&2
    fi
  done
  for f in requirements.txt pyproject.toml preferences.md; do
    if [[ -f "$NESTED/$f" ]]; then
      cp "$NESTED/$f" "$STORE/$f"
      echo "  merged $f" >&2
    fi
  done
  echo "  (remove \$STORE/files when done: rm -rf \"\$STORE/files\")" >&2
  echo "" >&2
fi
if [[ "$STORE" == "$STORE_ROOT/files" && -d "$STORE_ROOT/docs" ]]; then
  if [[ -f "$STORE/docs/project-context.md" && -f "$STORE_ROOT/docs/project-context.md" ]] \
    && [[ "$STORE/docs/project-context.md" -nt "$STORE_ROOT/docs/project-context.md" ]]; then
    : # files/docs is canonical; store-root mirror updated below after docs sync
  elif [[ -f "$STORE/docs/project-context.md" && -f "$STORE_ROOT/docs/project-context.md" ]] \
    && [[ "$STORE_ROOT/docs/project-context.md" -nt "$STORE/docs/project-context.md" ]]; then
    echo "WARNING: store-root docs/ looks newer than files/docs — cloud UI may be stale." >&2
    echo "         Run sync_repo_to_context from repo, or copy store-root docs → files/docs." >&2
    echo "" >&2
  fi
fi

sync_tree docs docs
# Keep store-root docs in sync with files/docs (same as src mirror in repo→Context script)
if [[ "$STORE" == "$STORE_ROOT/files" && -d "$STORE/docs" ]]; then
  sync_tree_from "$STORE/docs" "$STORE_ROOT/docs" "docs → store-root mirror" "--delete"
fi
sync_tree scripts scripts
rm -f "$REPO/scripts/requirements.txt"
if [[ "$STORE" == "$STORE_ROOT/files" ]]; then
  rm -f "$STORE_ROOT/requirements.txt" "$STORE_ROOT/scripts/requirements.txt"
fi

# Package code: files/src on Mac
prune_bad_src_layout "$STORE" "Context files"
if [[ -d "$STORE_ROOT/src" ]]; then
  prune_bad_src_layout "$STORE_ROOT" "Context store-root"
fi

SRC_DIR="$(resolve_src_dir)"
if [[ -n "$SRC_DIR" ]]; then
  sync_tree_from "$SRC_DIR" "$REPO/src" "src → repo/src" "--delete"
  prune_bad_src_layout "$REPO" "repo"
else
  echo "skip (missing in Context): src — expected \$STORE/src or \$STORE_ROOT/src" >&2
  SKIPPED=$((SKIPPED + 1))
fi

sync_tree internal docs/internal

REQ_SRC=""
if [[ -f "$STORE/requirements.txt" ]]; then
  REQ_SRC="$STORE/requirements.txt"
elif [[ -f "$STORE_ROOT/files/requirements.txt" ]]; then
  REQ_SRC="$STORE_ROOT/files/requirements.txt"
elif [[ -f "$STORE_ROOT/requirements.txt" ]]; then
  REQ_SRC="$STORE_ROOT/requirements.txt"
fi
if [[ -n "$REQ_SRC" ]]; then
  cp "$REQ_SRC" "$REPO/requirements.txt"
  echo "synced requirements.txt → repo/requirements.txt (from $REQ_SRC)"
  SYNCED=$((SYNCED + 1))
else
  echo "skip (missing in Context): requirements.txt — expected \$STORE/requirements.txt" >&2
  SKIPPED=$((SKIPPED + 1))
fi

if [[ -f "$STORE/pyproject.toml" ]]; then
  cp "$STORE/pyproject.toml" "$REPO/pyproject.toml"
  echo "synced pyproject.toml → repo/pyproject.toml"
  SYNCED=$((SYNCED + 1))
elif [[ -f "$STORE/scripts/pyproject.toml" ]]; then
  cp "$STORE/scripts/pyproject.toml" "$REPO/pyproject.toml"
  echo "synced pyproject.toml → repo/pyproject.toml (from scripts/)"
  SYNCED=$((SYNCED + 1))
fi

ARCHIVE_DOCS="internal/archived/context-docs-2026-09-29"
if [[ -d "$STORE/$ARCHIVE_DOCS" ]]; then
  sync_tree_from "$STORE/$ARCHIVE_DOCS" "$REPO/docs/internal/archived/context-docs-2026-09-29" \
    "internal/archived → docs/internal/archived/context-docs-2026-09-29"
fi

if compgen -G "$REPO/scripts/"*.sh >/dev/null; then
  chmod +x "$REPO/scripts/"*.sh
  echo "chmod +x: $REPO/scripts/*.sh"
fi

echo ""
echo "Summary: $SYNCED tree(s) synced, $SKIPPED skipped (missing in Context)."
echo "Next: cd $REPO && git status"
