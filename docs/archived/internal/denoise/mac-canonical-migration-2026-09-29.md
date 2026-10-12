---
cursor:
  subagentId: "bc-8d395e6a-ece8-5dc0-9f09-614d10159d03"
---

# Mac canonical migration (store-only, 2026-09-29)

## Done

- Rewrote `scripts/sync_context_to_repo.sh` — full `docs/`, `scripts/`, `internal/` → repo (`internal` → `docs/internal/`).
- Archived former Context `docs/` tree to `internal/archived/context-docs-2026-09-29/`; `docs/` now stub `README.md` only.
- `internal/README.md` explains `docs/internal` sync target.
- `preferences.md` / `notes.md` — repo-relative doc refs; no `cursor.agent://project/context/docs/*` for removed files.

## User on Mac

```bash
bash ~/Library/Application\ Support/Cursor/AgentStores/cursor_agent_stores/bc-a64d420e-f2ca-403e-91f6-786d63f9c464/files/scripts/sync_context_to_repo.sh
cd ~/Projects/bayer_rgb && git status
```

Commit synced `docs/`, `scripts/`, and `docs/internal/`.
