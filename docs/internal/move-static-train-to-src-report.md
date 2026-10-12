---
cursor:
  subagentId: "bc-e9d32e21-d3e4-5690-90b0-2936d4eadb74"
---

# Move `train_static_noisy_sequence` into package `src/core`

## Summary

Active static-sequence training now lives in **`files/src/core/train_static_noisy_sequence.py`** (mirrored to store-root **`src/core/`**). The old **`scripts/train_static_noisy_sequence.py`** path is a thin deprecation stub (exit 1).

## New command line

From repo root with `pip install -e .` and venv active:

```bash
python -m bayer_denoiser.core.train_static_noisy_sequence --help

python -m bayer_denoiser.core.train_static_noisy_sequence \
  --sequence-dir runs/gaussian-gt-test/static_sequence \
  --output-dir runs/static-seq-plumbing \
  --crop-size 512 \
  --epochs 100
```

Shell wrapper (unchanged path, now calls `-m` internally):

```bash
bash scripts/train_static_noisy_sequence.sh
```

Deprecated (prints message, exit 1):

```bash
python scripts/train_static_noisy_sequence.py
```

## Files changed (Context `files/`)

| Action | Path |
|--------|------|
| **Added** | `src/core/train_static_noisy_sequence.py` |
| **Replaced** | `scripts/train_static_noisy_sequence.py` → deprecation stub |
| **Updated** | `scripts/train_static_noisy_sequence.sh`, `scripts/gaussian_gt_pipeline_test.sh`, `scripts/bayer_denoise_smoke.sh`, `scripts/README.md`, `scripts/denoise_sample_a001c006.sh`, `scripts/run_real_bayer_denoise.sh` |
| **Updated** | `src/README.md`, `src/archived/README.md` |
| **Updated** | `docs/guides/static-noisy-sequence.md`, `docs/guides/training-user-guide.md`, `docs/guides/real-data-runbook.md`, `docs/operations/requirements-guide.md`, `docs/inference/gaussian-gt-pipeline-test.md` |
| **Updated** | `preferences.md` |
| **Not touched** | `src/archived/core/train.py` |

Store-root siblings **`scripts/`**, **`src/`**, **`docs/`**, **`preferences.md`** were updated to match `files/` where those trees exist.

Nested duplicate **`files/scripts/scripts/`** received the same stub and `.sh` `-m` change where present.

## `pyproject.toml`

**Not present in Project Context** (`files/` or store root). On Mac repo:

- No new console script required if docs use `-m` only (same pattern as `bayer_denoiser.core.infer`).
- If `[project.scripts]` still lists `bayer-denoiser-train` → `bayer_denoiser.core.train:main`, remove or point at archived train only; do **not** add a duplicate entry for static-sequence train unless you want a short alias.

## Mac sync steps

1. On Mac, from `~/Projects/bayer_denoiser`:
   ```bash
   bash scripts/sync_context_to_repo.sh
   ```
   (Pulls `docs/`, `scripts/`, `src/` from Context into the git repo.)

2. Verify:
   ```bash
   source .venv/bin/activate
   pip install -e .
   python -m bayer_denoiser.core.train_static_noisy_sequence --help
   python scripts/train_static_noisy_sequence.py   # expect exit 1 + message
   bash scripts/train_static_noisy_sequence.sh     # dry-run with SEQUENCE_DIR if desired
   ```

3. Commit repo changes; optional `sync_repo_to_context.sh` to refresh Context mirrors after local edits.

## Implementation notes

- Imports already used `bayer_denoiser.*`; no path hacks removed.
- `parse_args()` + `main()` + `if __name__ == "__main__"` match `core/infer.py` CLI style.
- `gaussian_gt_pipeline_test.sh` invokes the `-m` module directly.
