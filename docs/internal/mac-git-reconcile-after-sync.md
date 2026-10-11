# Mac git reconcile after Context sync

Use when `git status` shows **staged** old `docs/guides/` moves **and** **unstaged** deletes of the same paths plus new flat `docs/*.md`.

## What happened

1. An earlier **partial commit was staged** (guides layout, old sync script names, `scripts/train_static_noisy_sequence.*`).
2. **`sync_context_to_repo.sh`** then applied the **current** Context layout: flat live docs at `docs/` root, legacy under `docs/archived/`, training in `src/train_static_noisy_sequence.py`, `scripts/input_and_infer_crop.py`, etc.
3. Git now mixes **staged (old)** + **working tree (new)**. Sync did run; the index is just stale.

## Target layout (match Context `files/docs`)

- Live: `docs/DOC_INDEX.md`, `project-context.md`, `static-noisy-sequence.md`, … (no `docs/guides/` for live docs)
- Archive: `docs/archived/**`
- Internal reports: `docs/internal/**`
- **No** `docs/docs/` (delete if present — mistaken nested copy)

## Commands (repo root)

```bash
cd ~/Projects/bayer_denoiser

# 1. Drop stale index; keep working tree from last sync
git reset HEAD

# 2. Junk nested docs mirror (if present)
rm -rf docs/docs

# 3. Re-sync after merging nested Context (if you still use files/files/…)
STORE="$HOME/Library/Application Support/Cursor/AgentStores/cursor_agent_stores/bc-a64d420e-f2ca-403e-91f6-786d63f9c464/files"
cp "$STORE/scripts/sync_context_to_repo.sh" scripts/
bash scripts/sync_context_to_repo.sh

# 4. Stage everything that should ship
git add -A
git status

# 5. Sanity check (no guides/, no docs/docs/, no scripts/train_static_noisy_sequence.*)
test ! -d docs/guides && test ! -d docs/docs && test ! -f scripts/train_static_noisy_sequence.py && echo "layout OK"

git diff --cached --stat
```

Then commit with a message like: `Align repo with Context: flat docs, archived legacy, src layout, sync scripts`.

## `input_and_infer_crop.py` missing in repo

Cloud Context has `scripts/input_and_infer_crop.py`; Mac Agent Store may still have `infer_bayer_mosaic_crop.py`. Sync will not rename automatically.

```bash
git mv scripts/infer_bayer_mosaic_crop.py scripts/input_and_infer_crop.py
cp scripts/input_and_infer_crop.py "$STORE/scripts/"
rm -f "$STORE/scripts/infer_bayer_mosaic_crop.py"
```

## Do not commit

- `docs/docs/` — remove before `git add`
- `scripts/requirements.txt` — sync removes it; only root `requirements.txt`
- `requirements-raw.txt` — deleted; stay deleted
