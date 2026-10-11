---
cursor:
  subagentId: "bc-a3330753-2bb5-503a-b5e6-175ba1ac37e6"
---

# Float16 pipeline docs unify

## Kept

`docs/guides/float16-pipeline-standard.md` (mirrored to `files/docs/guides/float16-pipeline-standard.md`).

## Removed

`docs/formats/float16-pipeline-standard.md` (no redirect stub). Same file gone under `files/docs/formats/`. Archive snapshot `docs/internal/archived/` untouched.

## Unique bits merged

Formats copy was the same policy body. Differences vs guides:

| Formats-only | Action |
|--------------|--------|
| Checklist link `../internal/float16-pipeline-implementation.md` | Merged into guides (replaced broken `../../../float16-pipeline-implementation.md`) |
| Same-folder links `sky-format-conversion.md` / sibling guide names | Not merged; guides already uses `../formats/sky-format-conversion.md` and same-dir guide links |
| `overfit_bayer_denoise_pair.py` | Not merged; guides already has `python -m bayer_denoiser.core.train_static_noisy_sequence` |

No formats-only paragraph needed a stub.

Canonical file already had: `bayer_denoiser`, `.sky` → float16 `.npy`, `python -m bayer_denoiser.{sky,train,infer}`.

## Other live-doc link fixes

`docs/internal/float16-pipeline-implementation.md`:

- Policy link `../docs/float16-pipeline-standard.md` → `../guides/float16-pipeline-standard.md`
- Script table: `overfit_bayer_denoise_pair.py` → `python -m bayer_denoiser.core.train_static_noisy_sequence`

`docs/DOC_INDEX.md`: already listed only `guides/float16-pipeline-standard.md`; no formats-copy mention to strip.

Live `docs/` (excluding archive) has **no** remaining `formats/float16-pipeline-standard` links. Sibling files already pointed at `guides/` (`sky-format-conversion.md`) or same-dir (`training-user-guide.md`, `real-data-runbook.md`, `project-context.md`).

## Mirror

`cp -R docs/. files/docs/` after the delete. `diff -rq docs files/docs` empty. `files/docs/formats/float16-pipeline-standard.md` absent.
