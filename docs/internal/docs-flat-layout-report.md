---
cursor:
  subagentId: "bc-275a5936-4f3d-5d15-8943-ada9f0d17b2d"
---

# Docs flat layout (2026-10-07)

## Summary

Moved six live guides from `docs/guides/`, `docs/formats/`, `docs/operations/`, and `docs/inference/` into **`files/docs/` root**. Removed empty top-level subdirs. Rewrote **[DOC_INDEX.md](../files/docs/DOC_INDEX.md)** and **[README.md](../files/docs/README.md)** for flat paths. Fixed cross-links in all eight live docs and updated live references under `files/` (scripts, `src/`, `internal/denoise/`, `preferences.md`).

**`docs/archived/`** tree unchanged except a short note in **[archived/README.md](../files/docs/archived/README.md)** and a fixed legacy pointer in `archived/legacy-guides/README.md`.

## Live files (`files/docs/` root)

| File | Former path |
|------|----------------|
| `DOC_INDEX.md` | (was already root) |
| `README.md` | (was already root) |
| `project-context.md` | `guides/project-context.md` |
| `static-noisy-sequence.md` | `guides/static-noisy-sequence.md` |
| `float16-pipeline-standard.md` | `guides/float16-pipeline-standard.md` |
| `sky-format-conversion.md` | `formats/sky-format-conversion.md` |
| `requirements-guide.md` | `operations/requirements-guide.md` |
| `gaussian-gt-pipeline-test.md` | `inference/gaussian-gt-pipeline-test.md` |

Removed empty dirs: `files/docs/guides/`, `formats/`, `operations/`, `inference/`.

## Mirror

Copied `files/docs/` → store parent **`/cursor/stores/bc-a64d420e-f2ca-403e-91f6-786d63f9c464/docs/`** (no `rsync` in VM; used `cp`). Deleted stale **`docs/guides/`**, **`docs/formats/`**, **`docs/operations/`**, **`docs/inference/`** copies on the parent mirror. Synced **`preferences.md`** to store root.

## Mac sync + delete stale subfolders

`scripts/sync_context_to_repo.sh` is **one-way copy** and does **not** delete obsolete paths on `~/Projects/bayer_denoiser/docs/`.

After pulling Context:

1. Run **`bash scripts/sync_context_to_repo.sh`** (or the Agent Store path in `docs/README.md`).
2. **Manually remove** on the Mac repo if they still exist:
   - `docs/guides/` (old live copies)
   - `docs/formats/sky-format-conversion.md` (and empty `formats/` if only that file)
   - `docs/operations/requirements-guide.md`
   - `docs/inference/gaussian-gt-pipeline-test.md`
3. Confirm flat files exist at `docs/*.md` per table above.
4. Optional: use rsync **`--delete`** only if you extend the sync script deliberately — default script does not prune.

## Reference updates (live only)

- `files/preferences.md` + store `preferences.md`: `docs/static-noisy-sequence.md`; rename checklist `docs/archived/guides/rename-to-bayer-denoiser.md` (parent was wrong `docs/guides/...`).
- Scripts (`bayer_denoise_smoke.sh`, `denoise_sample_a001c006.sh`, `scripts/README.md`): flat doc paths.
- `src/README.md`, `src/core/model.py`: flat / archived GCP path.
- `internal/phase-b-paired-raw-spec.md`, `internal/denoise/*`, `internal/local-machine-audit.md`: flat or `archived/` paths.

Archived markdown under `docs/archived/` was **not** bulk-rewritten (relative links may still assume old live layout); see archived README note.

## Verification

```bash
ls /cursor/stores/bc-a64d420e-f2ca-403e-91f6-786d63f9c464/files/docs/*.md
# eight live markdown files at root + archived/
```
