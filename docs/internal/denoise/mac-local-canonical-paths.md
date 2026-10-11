---
cursor:
  subagentId: "bc-d4258684-1867-512b-954e-a5cc1238ca4f"
---

# Mac-local canonical paths (user request)

User wants artifacts on **`~/Projects/bayer_rgb`**, not cloud-only Context.

## Cloud blocker

No self-hosted Mac worker connected during this turn — cloud agent cannot write the Mac checkout directly.

## Delivered

- **`scripts/sync_context_to_repo.sh`** — one-shot copy from synced Agent Store `files/` into repo `scripts/` + `docs/` (flat live guides)
- **`docs/archived/inference/README.md`** — archived inference write-ups
- Updated **`docs/static-noisy-singleframe-pipeline-test.md`** — canonical = Mac repo

## User action (once)

```bash
STORE="$HOME/Library/Application Support/Cursor/AgentStores/cursor_agent_stores/bc-a64d420e-f2ca-403e-91f6-786d63f9c464/files"
bash "$STORE/scripts/sync_context_to_repo.sh"
```

If sync script not in Store yet: copy from Context or run manual `cp` from prior handoff.

## Follow-up

With **`cursor worker start`** on Mac, an agent can commit these into `bayer_rgb` git and extend sync to full Context doc mirror if requested.
