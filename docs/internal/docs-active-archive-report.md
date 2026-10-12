---
cursor:
  subagentId: "bc-d39f8f57-c638-5fa6-acd3-792ebeb0ba06"
---

# Docs active vs archive (2026-10-07)

Source tree: `files/docs/` in the Project Agent Store. Store-root mirror: `docs/` (same layout after flattening a mistaken `docs/docs/` nest).

## Live (`docs/` — 8 markdown files)

| Path | Role |
|------|------|
| `DOC_INDEX.md` | Rewritten: active docs only + link to archive |
| `README.md` | Stub → `DOC_INDEX` and `archived/` |
| `guides/project-context.md` | Project orientation (links to archive for legacy runbooks) |
| `guides/static-noisy-sequence.md` | Static burst training |
| `guides/float16-pipeline-standard.md` | Dtype policy (archive links for runbooks / internal checklist) |
| `formats/sky-format-conversion.md` | `.sky` → `.npy` |
| `operations/requirements-guide.md` | requirements vs requirements-raw |
| `inference/gaussian-gt-pipeline-test.md` | Gaussian pseudo-GT pipeline |

Removed from live tree: `docs/internal/` (entire tree).

## Archive layout (`docs/archived/`)

| Subfolder | Contents |
|-----------|----------|
| `archived/README.md` | Index of all archived material |
| `archived/guides/` | `BAYER_DENOISE`, `GCP_STILL`, `MANUAL`, `PIPELINE`, `real-data-runbook`, `rename-to-bayer-denoiser`, `training-user-guide` |
| `archived/formats/` | `RAW_SPEC.md` |
| `archived/inference/` | `README.md`, `a001c006-basic-denoise-test.md` |
| `archived/operations/` | `licensing-notes.md` |
| `archived/internal/` | Former `docs/internal/*` (audits, handoffs, `denoise/`, `archived/context-docs-2026-09-29/`) |
| `archived/legacy-guides/` | Unchanged: `README.md`, `TRAINING_GUIDE.md` |

**Count:** 42 archived `.md` files under `archived/` (including nested `internal/archived/context-docs-2026-09-29/`).

## Link fixes (live docs only)

- `guides/project-context.md` — archive paths for rename checklist, RAW_SPEC, internal notes, GCP/PIPELINE/training-user-guide.
- `inference/gaussian-gt-pipeline-test.md` — a001c006 + training-user-guide → `archived/`.
- `guides/float16-pipeline-standard.md` — training runbook + internal implementation checklist → `archived/`.
- `DOC_INDEX.md` — full rewrite.
- `files/preferences.md` — rename checklist path → `docs/archived/guides/rename-to-bayer-denoiser.md`.

No mass edits inside archived markdown.

## Mac sync note

`scripts/sync_context_to_repo.sh` (and typical `rsync` without `--delete`) **copies new paths but does not remove** files that used to live at:

- `docs/guides/MANUAL.md`, `docs/guides/training-user-guide.md`, …
- `docs/internal/**`
- `docs/inference/README.md`, `docs/formats/RAW_SPEC.md`, etc.

After syncing Context → Mac repo, either:

1. **Delete stale paths** under `~/Projects/bayer_denoiser/docs/` that are now only under `docs/archived/`, or  
2. Run rsync with **`--delete`** (only if the sync script is updated to target the full `docs/` tree safely), or  
3. Remove the whole `docs/` folder on the Mac and re-run a full sync once.

Same caveat applies to store-root `docs/` if an older mirror left duplicate active copies.

## Store mirror

`files/docs/` was copied to `/cursor/stores/.../docs/` (rsync unavailable in VM; used `cp -a`). Store-root active set matches the 8 live files above.
