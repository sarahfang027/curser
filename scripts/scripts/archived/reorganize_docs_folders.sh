#!/usr/bin/env bash
# Idempotent docs/ subfolder layout for ~/Projects/bayer_denoiser (Mac).
# Run from repo root after syncing this script from Context:
#   bash scripts/reorganize_docs_folders.sh
#
# Does NOT move: docs/inference/, docs/internal/, docs/DOC_INDEX.md, docs/README.md
# Skips any source file that is missing (already moved or repo-only).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ ! -d docs ]]; then
  echo "No docs/ directory under $REPO_ROOT" >&2
  exit 1
fi

mkdir -p docs/guides docs/formats docs/operations

git_mv_if_exists() {
  local src="$1" dest="$2"
  if [[ -f "$src" ]]; then
    if [[ -f "$dest" ]]; then
      echo "skip (already at dest): $dest"
    else
      git mv "$src" "$dest"
      echo "moved: $src -> $dest"
    fi
  else
    echo "skip (missing): $src"
  fi
}

# Context-synced / user guides
git_mv_if_exists docs/project-context.md docs/guides/project-context.md
git_mv_if_exists docs/real-data-runbook.md docs/guides/real-data-runbook.md
git_mv_if_exists docs/training-user-guide.md docs/guides/training-user-guide.md
git_mv_if_exists docs/float16-pipeline-standard.md docs/guides/float16-pipeline-standard.md

# Repo-native guides (skip if missing)
git_mv_if_exists docs/MANUAL.md docs/guides/MANUAL.md
git_mv_if_exists docs/TRAINING_GUIDE.md docs/guides/TRAINING_GUIDE.md
git_mv_if_exists docs/BAYER_DENOISE.md docs/guides/BAYER_DENOISE.md
git_mv_if_exists docs/GCP_STILL.md docs/guides/GCP_STILL.md
git_mv_if_exists docs/PIPELINE.md docs/guides/PIPELINE.md

# Formats
git_mv_if_exists docs/RAW_SPEC.md docs/formats/RAW_SPEC.md
git_mv_if_exists docs/sky-format-conversion.md docs/formats/sky-format-conversion.md

# Operations
git_mv_if_exists docs/requirements-guide.md docs/operations/requirements-guide.md
git_mv_if_exists docs/licensing-notes.md docs/operations/licensing-notes.md

fix_doc_links() {
  local guides=(docs/guides/*.md)
  [[ -e docs/guides ]] || return 0

  # guides/ — cross-folder references
  for f in docs/guides/*.md; do
    [[ -f "$f" ]] || continue
    perl -pi -e 's|\]\(sky-format-conversion\.md\)|](../formats/sky-format-conversion.md)|g' "$f"
    perl -pi -e 's|\]\(inference/|](../inference/|g' "$f"
    perl -pi -e 's|\]\(RAW_SPEC\.md\)|](../formats/RAW_SPEC.md)|g' "$f"
  done

  if [[ -f docs/formats/sky-format-conversion.md ]]; then
    perl -pi -e 's|\]\(float16-pipeline-standard\.md\)|](../guides/float16-pipeline-standard.md)|g' \
      docs/formats/sky-format-conversion.md
  fi

  for f in docs/inference/*.md; do
    [[ -f "$f" ]] || continue
    perl -pi -e 's|\]\(\.\./sky-format-conversion\.md\)|](../formats/sky-format-conversion.md)|g' "$f"
    perl -pi -e 's|\]\(\.\./training-user-guide\.md\)|](../guides/training-user-guide.md)|g' "$f"
    perl -pi -e 's|\]\(\.\./real-data-runbook\.md\)|](../guides/real-data-runbook.md)|g' "$f"
    perl -pi -e 's|\]\(\.\./project-context\.md\)|](../guides/project-context.md)|g' "$f"
    perl -pi -e 's|\]\(\.\./float16-pipeline-standard\.md\)|](../guides/float16-pipeline-standard.md)|g' "$f"
  done

  echo "Link fix pass complete (guides/, formats/, inference/)."
}

fix_doc_links

cat <<'EOF'

Next steps:
  1. Sync or copy docs/DOC_INDEX.md from Context if needed.
  2. rg for stale paths: docs/project-context.md, ](training-user-guide.md) from docs root, etc.
  3. git diff docs/ && commit when satisfied.

EOF
