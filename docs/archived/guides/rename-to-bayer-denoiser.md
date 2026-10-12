# Rename `bayer_rgb` → `bayer_denoiser` (Mac checklist)

**Why:** the project is a **Bayer-space denoiser on packed RGGB** (`BayerDenoiseNet`, `--model bayer_denoise`). The old name `bayer_rgb` suggested raw→RGB; that path (GCP / joint demosaic) is **deferred** and is not the product name.

| | Old | New |
|---|-----|-----|
| Repo folder | `~/Projects/bayer_rgb` | `~/Projects/bayer_denoiser` |
| Python package / import | `bayer_rgb` | `bayer_denoiser` |
| CLI | `python -m bayer_rgb.train` | `python -m bayer_denoiser.archived.core.train` |
| Sync script | `sync_context_to_repo.sh` | `sync_context_to_repo.sh` (old name = deprecated wrapper) |
| Root env var (scripts) | `BAYER_RGB_ROOT` | `BAYER_DENOISER_ROOT` (old still honoured as fallback) |

**Unchanged:** model/flag names (`BayerDenoiseNet`, `--model bayer_denoise`, `--model gcp`), script names like `bayer_denoise_smoke.sh`, data dirs (`data/noisy`, `data/clean`, legacy `data/bayer` + `data/rgb`), `runs/` layout, the Agent Store path (`…/bc-a64d420e-…`, no project name in it).

---

## Order of operations (do exactly this)

Run on the Mac. Steps 1–8 are the code/repo rename; 9–10 pull docs/scripts; 11 verifies.

1. **Clean tree + safety tag.** `cd ~/Projects/bayer_rgb && git status` (commit or stash everything), then `git tag pre-rename-bayer_rgb`. Close terminals/editors using the repo; `deactivate` any venv.
2. **Branch.** `git switch -c rename/bayer-denoiser`
3. **Package rename (`git mv`).** Inspect first: `grep -n -A6 'package-dir\|packages\|\[project.scripts\]' pyproject.toml`
   - **Case A — `src/` *is* the package** (`package-dir = {"bayer_rgb" = "src"}`; what Context docs describe): no directory to move. Only edit `pyproject.toml` (step 4).
   - **Case B — `src/bayer_rgb/` exists:** `git mv src/bayer_rgb src/bayer_denoiser`
4. **`pyproject.toml`.** Edit (details in [pyproject section](#pyprojecttoml)). Steps 3–5 together become the "pure rename" commit in step 6 (keeps `git log --follow` clean).
5. **Rewrite imports/CLI in code** (code paths only; docs come from sync in step 9):
   ```bash
   git grep -lIz -e 'bayer_rgb' -e 'BAYER_RGB' -e 'bayer-rgb' -- \
     src tests scripts pyproject.toml requirements*.txt README.md Makefile .github 2>/dev/null \
   | xargs -0 perl -pi -e 's/(?<![A-Za-z0-9])bayer_rgb(?![A-Za-z0-9])/bayer_denoiser/g; s/BAYER_RGB_ROOT/BAYER_DENOISER_ROOT/g; s/bayer-rgb/bayer-denoiser/g'
   ```
   (`perl -pi` works on macOS without the BSD `sed -i ''` quirk. The lookarounds also catch `to_bayer_rgb.sh` and `bayer_rgb.train`, but not `bayer_rgbx`.) Then `git diff --stat` and skim.
6. **Commit the code rename:** `git add -A && git commit -m "Rename package bayer_rgb -> bayer_denoiser"`
7. **Rename the repo folder** (from *outside* it): `cd ~/Projects && mv bayer_rgb bayer_denoiser && cd bayer_denoiser`. (Optional: rename the GitHub remote repo too — GitHub redirects the old URL; then `git remote set-url origin …`.)
8. **Recreate the venv and reinstall.** The old `.venv` has absolute paths baked in — do not reuse it:
   ```bash
   rm -rf .venv src/*.egg-info *.egg-info
   python3 -m venv .venv && source .venv/bin/activate
   pip install -U pip
   pip install -r requirements.txt
   pip install -r requirements-raw.txt   # EXR / imageio
   pip install -e .
   python -c "import bayer_denoiser; print(bayer_denoiser.__file__)"
   python -m bayer_denoiser.archived.core.train -h | head -3
   ```
   (Optional: `pip list | grep -i bayer` — expect only `bayer-denoiser`; if `bayer-rgb` appears, `pip uninstall bayer-rgb` in the *old* venv or just rely on the fresh venv.)
9. **Pull updated docs + scripts from Context** (Cursor must have synced the Project store; new script name):
   ```bash
   bash "$HOME/Library/Application Support/Cursor/AgentStores/cursor_agent_stores/bc-a64d420e-f2ca-403e-91f6-786d63f9c464/files/scripts/sync_context_to_repo.sh"
   git status && git diff --stat
   ```
   The script defaults to `~/Projects/bayer_denoiser`; if you still see "Repo not found" it prints the `mv` hint. It overwrites `docs/`, `scripts/`, `docs/internal/` — this is why step 1 needed a clean tree. Then delete leftovers from the old name if present: `git rm -f scripts/sync_context_to_repo.sh` (or keep the wrapper for a week; it just calls the new script).
10. **Commit docs/scripts:** `git add docs scripts && git commit -m "Docs/scripts: bayer_denoiser naming + rename guide"`
11. **Verify** (next section), run the smoke test, then merge: `git switch main && git merge --no-ff rename/bayer-denoiser`. Only delete the tag `pre-rename-bayer_rgb` when satisfied.

Afterwards, repoint anything external: Cursor self-hosted worker / workspace folder → `~/Projects/bayer_denoiser`, shell aliases (`~/.zshrc`, `~/.zprofile`), `.cursor/` rules, Jupyter kernelspecs (`jupyter kernelspec list`), launch configs, `PYTHONPATH`, data symlinks pointing into the old folder (`find data -type l -exec ls -l {} +`).

---

## `pyproject.toml`

Adjust whichever of these exist (values illustrative):

```toml
[project]
name = "bayer_denoiser"            # was bayer_rgb / bayer-rgb

[tool.setuptools]
package-dir = {"bayer_denoiser" = "src"}   # Case A; was {"bayer_rgb" = "src"}
packages = ["bayer_denoiser"]

# Case B (src/bayer_denoiser/): use
# [tool.setuptools.packages.find]
# where = ["src"]

[project.scripts]                  # only if entry points exist
# Remove bayer-denoiser-train entry (train archived: bayer_denoiser.archived.core.train)
bayer-denoiser-infer = "bayer_denoiser.core.infer:main"
```

Also check `[tool.setuptools.package-data]`, `[tool.pytest.ini_options]` (`pythonpath`, `testpaths`), `[tool.coverage]`, `[tool.mypy]`, `[tool.ruff]` (`known-first-party`) for the old name. `setup.py` / `setup.cfg` if present.

Keep the earlier sky fix in mind (drop `src/sky` from packages; see [sky-format-conversion.md](../formats/sky-format-conversion.md)).

## CLI mapping

Every `python -m bayer_rgb.<mod>` becomes `python -m bayer_denoiser.<mod>`: `train`, `infer`, `preview`, `validate_raw`, `validate_pairs`, `visualize_crops`, `make_sidecar`, plus any `sky` module invocation. Scripts under `scripts/` (`bayer_denoise_smoke.sh`, `pipeline_smoke.sh`, `static_noisy_singleframe_pipeline_test.sh`, `python -m bayer_denoiser.core.train_static_noisy_sequence`, …) already use the new import in the Context copies.

## Sync scripts

| Script | Change |
|--------|--------|
| `sync_context_to_repo.sh` | New canonical name; default `REPO=~/Projects/bayer_denoiser`; helpful hint if only the old folder exists |
| `sync_context_to_repo.sh` | Deprecated wrapper (`exec`s the new script); delete when nothing calls it |
| `sync_repo_to_context.sh`, `sync_repo_to_context.sh`, `sync_repo_to_context.sh` | Default `REPO=~/Projects/bayer_denoiser`; behaviour unchanged |

Also update `docs/README.md`, `docs/inference/README.md`, and any personal notes that quote the old script name.

## Checkpoints (`*.pt`) — one thing to test

If a checkpoint stored a pickled object (full model or config class) rather than a `state_dict`, `torch.load` may look for module `bayer_rgb`. Test one: `python -c "import torch; torch.load('runs/bayer-denoise-smoke/model.pt', map_location='cpu', weights_only=False)"`. If it fails with `No module named 'bayer_rgb'`, either retrain the smoke model or use a temporary shim before loading:

```python
import sys, bayer_denoiser
sys.modules["bayer_rgb"] = bayer_denoiser   # plus submodules if needed
```

then re-save as `state_dict`. Plain `state_dict` checkpoints (the usual case) are unaffected.

---

## Verification (all should print nothing unless noted)

```bash
cd ~/Projects/bayer_denoiser

# 1. Old name left in tracked text files (outside archives)
git grep -nI -e 'bayer_rgb' -e 'BAYER_RGB' -e 'bayer-rgb' -- . ':!docs/internal/archived'

# 2. Old CLI/imports specifically (must be empty)
git grep -nIE 'python -m bayer_rgb|from bayer_rgb|import bayer_rgb' 

# 3. Stale build artifacts / old venv
find . -maxdepth 3 \( -name '*bayer_rgb*' -o -name '*bayer-rgb*' \) -not -path './.git/*'

# 4. New name resolves
python -c "import bayer_denoiser, sys; print(bayer_denoiser.__file__)"
python -c "import bayer_rgb" 2>&1 | tail -1     # expect: ModuleNotFoundError

# 5. Smoke
python python -m bayer_denoiser.core.train_static_noisy_sequence ...  # primary; legacy smoke: scripts/archived/bayer_denoise_smoke.sh
```

Also inspect: `git diff --stat pre-rename-bayer_rgb..HEAD | tail -1` (rename detection should show mostly renames/small edits), `pip check`.

**Expected leftovers in check 1** (fine): agent-title links in `docs/guides/project-context.md` ("Audit local bayer_rgb setup", "Map agents to bayer_rgb"), and this guide's explanatory text. Anything else in `src/`, `scripts/`, `pyproject.toml`, `docs/guides/`, `docs/inference/` is a miss — fix it.

To catch un-synced repo-only docs (`MANUAL.md`, `TRAINING_GUIDE.md`, `PIPELINE.md`, `RAW_SPEC.md`, …) if step 9 didn't rewrite them: run the same `perl` one-liner over `docs/guides docs/formats docs/operations docs/inference`, then re-review `git diff docs`.

---

## What NOT to rename

- **Historical agent links/titles** — URLs use `bc-…` ids (no name in them); titles like "Audit local bayer_rgb setup" are records, leave them.
- **`docs/internal/archived/`** and Context `internal/archived/context-docs-2026-09-29/` — frozen history (~1600 hits); do not bulk-replace. Older `internal/` notes may also keep old paths; treat them as history.
- **Git history, tags, old commit messages**; the `pre-rename-bayer_rgb` tag.
- **Paper citations / external names** — GCP-Net (Guo et al., TIP 2021), JDD references, upstream repos (`GuoShi28/GCP-Net`).
- **Model and flag names** — `BayerDenoiseNet`, `--model bayer_denoise`, `--model gcp`, `bayer_denoise_net.py`, `rgb_to_bayer` helpers, `data/bayer` + `data/rgb` (legacy paired-EXR layout, unrelated to the package name).
- **Agent Store path** and Cursor Project name ("Bayer Denoising CNN") — store id is stable.
- Third-party env/venv content — recreate, don't edit in place.

## Optional: branch strategy

- **Default (recommended):** one branch `rename/bayer-denoiser` off `main`, two commits — (1) pure rename: `git mv` + `pyproject.toml` + code import rewrite, no other edits (so `git log --follow` and rename detection work); (2) docs/scripts sync. Merge with `--no-ff` after the smoke passes.
- **Cautious:** keep `main` on `bayer_rgb` until one real training/infer run works on the branch; the tag `pre-rename-bayer_rgb` is your rollback (`git switch main` + `mv` folder back).
- **If other branches are open:** rebase them *after* merging the rename; run the step-5 `perl` on each branch's changed files first, or rebase with `-X rename-threshold=50%` and re-run the grep checks.
- Don't rename the GitHub repo until the branch merges.
