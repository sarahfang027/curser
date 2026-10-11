---
cursor:
  subagentId: "bc-57378538-4ded-5460-969e-5c264768f8b3"
---

# Proposed `docs/` folder reorganization (Mac)

**Status:** applied in Context archive (`internal/archived/context-docs-2026-09-29/` mirrors `guides/`, `formats/`, `operations/`, `inference/`); [DOC_INDEX.md](../docs/DOC_INDEX.md) updated in Context. **Run on Mac:** `bash scripts/reorganize_docs_folders.sh` from `~/Projects/bayer_rgb` after syncing the script from Context.

## Target layout

```text
docs/
  DOC_INDEX.md
  README.md
  guides/           # orientation, runbooks, training, policy
  formats/          # ingest contracts & conversions
  inference/        # unchanged
  operations/       # deps & licensing
  internal/         # unchanged (synced from Context)
```

| Category | After move |
|----------|------------|
| **guides/** | `project-context.md`, `real-data-runbook.md`, `training-user-guide.md`, `float16-pipeline-standard.md`, `MANUAL.md`, `TRAINING_GUIDE.md`, `BAYER_DENOISE.md`, `GCP_STILL.md`, `PIPELINE.md` |
| **formats/** | `RAW_SPEC.md`, `sky-format-conversion.md` |
| **inference/** | (no change) |
| **operations/** | `requirements-guide.md`, `licensing-notes.md` |

## `git mv` commands (from repo root)

```bash
cd ~/Projects/bayer_rgb

mkdir -p docs/guides docs/formats docs/operations

# Context-synced / user guides (skip if file missing)
git mv docs/project-context.md docs/guides/
git mv docs/real-data-runbook.md docs/guides/
git mv docs/training-user-guide.md docs/guides/
git mv docs/float16-pipeline-standard.md docs/guides/

# Repo-native (skip if missing)
git mv docs/MANUAL.md docs/guides/
git mv docs/TRAINING_GUIDE.md docs/guides/
git mv docs/BAYER_DENOISE.md docs/guides/
git mv docs/GCP_STILL.md docs/guides/
git mv docs/PIPELINE.md docs/guides/

git mv docs/RAW_SPEC.md docs/formats/
git mv docs/sky-format-conversion.md docs/formats/

git mv docs/requirements-guide.md docs/operations/
git mv docs/licensing-notes.md docs/operations/

# DOC_INDEX.md: sync from Context, then update paths in DOC_INDEX (second pass) or run sed below
```

`docs/inference/`, `docs/internal/`, `docs/DOC_INDEX.md`, and `docs/README.md` stay at `docs/` root.

## Link-update grep (before/after commit)

From repo root, find references to moved basenames:

```bash
cd ~/Projects/bayer_rgb

rg -n 'docs/(project-context|real-data-runbook|training-user-guide|float16-pipeline-standard|requirements-guide|licensing-notes|sky-format-conversion|MANUAL|TRAINING_GUIDE|RAW_SPEC|GCP_STILL|BAYER_DENOISE|PIPELINE)\.md' \
  docs bayer_rgb scripts README.md 2>/dev/null

rg -n '\]\((project-context|real-data-runbook|training-user-guide|float16-pipeline-standard|requirements-guide|licensing-notes|sky-format-conversion|MANUAL|TRAINING_GUIDE|RAW_SPEC|GCP_STILL|BAYER_DENOISE|PIPELINE)\.md\)' \
  docs
```

Common relative-link rewrites inside `docs/guides/` (files that used to sit beside each other at `docs/` root):

| Old relative | New relative (from `docs/guides/`) |
|--------------|-------------------------------------|
| `float16-pipeline-standard.md` | `float16-pipeline-standard.md` (same dir) |
| `training-user-guide.md` | `training-user-guide.md` |
| `sky-format-conversion.md` | `../formats/sky-format-conversion.md` |
| `inference/static-noisy-singleframe-pipeline-test.md` | `../inference/static-noisy-singleframe-pipeline-test.md` |
| `GCP_STILL.md` | `GCP_STILL.md` |
| `RAW_SPEC.md` | `../formats/RAW_SPEC.md` |

From `docs/inference/*.md`:

| Old | New |
|-----|-----|
| `../sky-format-conversion.md` | `../formats/sky-format-conversion.md` |
| `../training-user-guide.md` | `../guides/training-user-guide.md` |
| `../real-data-runbook.md` | `../guides/real-data-runbook.md` |

Bulk sed (review diff before commit):

```bash
# Example: training-user-guide links to formats
perl -pi -e 's|\]\(RAW_SPEC\.md\)|](../formats/RAW_SPEC.md)|g' docs/guides/training-user-guide.md
perl -pi -e 's|\]\(GCP_STILL\.md\)|](GCP_STILL.md)|g' docs/guides/training-user-guide.md
perl -pi -e 's|\]\(float16-pipeline-standard\.md\)|](float16-pipeline-standard.md)|g' docs/guides/*.md
perl -pi -e 's|\]\(\.\./sky-format-conversion\.md\)|](../formats/sky-format-conversion.md)|g' docs/inference/*.md
```

Update [DOC_INDEX.md](../docs/DOC_INDEX.md) path column after moves (e.g. `guides/project-context.md`).

## `sync_context_to_repo.sh`

**No change required** for the index alone: the script mirrors `Context files/docs/` → `repo/docs/` recursively. If you later mirror the subfolder layout in Context `docs/`, rsync will preserve it. Optional follow-up: copy archived user docs back into Context `docs/guides/` etc. so a fresh Mac gets the full tree from sync (today only `DOC_INDEX.md` + `README.md` stub sync from Context).

## Context archive

**2026-09-29:** Archive on disk reorganized to match target subfolders; relative links in archived user docs and `inference/*.md` updated. Live Mac repo still needs `scripts/reorganize_docs_folders.sh`.
