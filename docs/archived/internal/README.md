# Internal / agent notes (`docs/internal` in repo)

This tree holds **agent handoffs, audits, and implementation notes** — not user-facing runbooks.

## Sync

On the Mac, `scripts/sync_context_to_repo.sh` copies:

`Context files/internal/` → `~/Projects/bayer_rgb/docs/internal/`

Agents should read synced copies from the **bayer_rgb** repo when working on the Mac worker. Cloud agents may still read `internal/` in Context for coordination; long-form product docs are only under `docs/` in the repo (see `docs/README.md` in Context stub).

## Archive

Pre-migration Context user docs were moved to `internal/archived/context-docs-2026-09-29/` (2026-09-29). Prefer the repo `docs/` tree after sync.
