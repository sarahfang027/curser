---
cursor:
  subagentId: "bc-136e7517-8e1f-5e7f-8cc1-c85e71b4af44"
---

# Docs outdated-content audit — `bayer_denoiser` Context `docs/` (2026-10-05)

**Scope:** `/cursor/stores/bc-a64d420e-f2ca-403e-91f6-786d63f9c464/docs/` (51 files). Cross-check vs `files/docs/` (Mac sync source per `sync_context_to_repo.sh`).

**Project truth (audit baseline):** package/repo `bayer_denoiser`; training plumbing via **`python -m bayer_denoiser.core.train_static_noisy_sequence`** (not `overfit_bayer_denoise_pair.py`); **`.sky` → float16 `.npy`** as forward training ingest; Gaussian GT as synthetic clean target; sync only **`sync_context_to_repo.sh`** / **`sync_repo_to_context.sh`**; legacy scripts under **`scripts/archived/`**.

---

## 1. Tree structure

### Top level (`docs/`)

| Entry | Role |
|-------|------|
| `DOC_INDEX.md`, `README.md` | Index + Context stub (current) |
| **Flat legacy copies (stale)** | `project-context.md`, `real-data-runbook.md`, `training-user-guide.md`, `sky-format-conversion.md`, `requirements-guide.md` — pre-`guides/`/`formats/`/`operations/` layout; **not** listed in `DOC_INDEX` canonical paths |
| `formats/` | `RAW_SPEC.md`, `float16-pipeline-standard.md`, `sky-format-conversion.md` |
| `guides/` | User runbooks + repo-git guides (`MANUAL`, `TRAINING_GUIDE`, `PIPELINE`, …) |
| `inference/` | Pipeline test write-ups |
| `operations/` | `requirements-guide.md`, `licensing-notes.md` |
| `internal/` | Agent notes synced from Context `internal/` → repo `docs/internal/` |

### Duplicate / parallel trees

| Tree | Relationship |
|------|----------------|
| **`docs/guides/*` vs flat `docs/*.md`** | Same basenames; **guides/** is canonical per `DOC_INDEX`; flat files are **older** (still cite `overfit_bayer_denoise_pair.py`, `.raw`-first training in `project-context.md`). |
| **`guides/float16-pipeline-standard.md` vs `formats/float16-pipeline-standard.md`** | Both kept; **content diverges** — `formats/` still names `overfit_bayer_denoise_pair.py`; `guides/` names `python -m bayer_denoiser.core.train_static_noisy_sequence`. Index points at **guides/** only. |
| **`requirements-guide.md` (root) vs `operations/requirements-guide.md`** | Appear **identical** (duplicate). |
| **`internal/archived/context-docs-2026-09-29/`** | Full snapshot (`guides/`, `formats/`, `operations/`, `inference/`, old `DOC_INDEX`) — intentionally historical; overlaps live tree. |
| **`inference/`** | Live tests; archived copy under `internal/archived/.../inference/`. |

---

## 2. Stale-term grep (summary)

| Term | Finding |
|------|---------|
| **`bayer_rgb`** | ~104 hits under `docs/` excluding archived rename guide; mostly **OK** in `rename-to-bayer-denoiser.md` and historical **internal/** notes; also agent link titles in `project-context.md`. |
| **`overfit` / `overfit_bayer_denoise_pair.py`** | **Still in flat root docs**, `formats/float16-pipeline-standard.md`, `internal/phase-b-paired-raw-spec.md`, `internal/float16-pipeline-implementation.md`, and **`files/docs/`** copies of `guides/real-data-runbook.md`, `guides/training-user-guide.md`, `inference/gaussian-gt-pipeline-test.md`. **Absent** from updated `guides/training-user-guide.md` (canonical). |
| **`sync_project_artifacts`, `sync_repo_docs`, `sync_repo_scripts`** | **No matches** in `docs/`. |
| **`contract_test`, `examples/`** | **No matches** in `docs/`. |
| **`reorganize_docs`** | `DOC_INDEX.md` + `internal/docs-reorganization-proposal.md` — one-time Mac script; proposal says **applied in Context**; index still reads like a pending step. |
| **`pipeline_smoke`** | Referenced as **GCP** synthetic smoke in `guides/PIPELINE.md`, `internal/pipeline-setup-report.md`, `operations/licensing-notes.md` — script still exists; **not** the Bayer denoise primary path (`bayer_denoise_smoke.sh` / Gaussian GT pipeline). OK if labeled GCP-only. |
| **Headerless `.raw` as primary training** | **Contradiction:** `guides/project-context.md` says `.raw` is **not** forward training; `guides/real-data-runbook.md` §1 still lists `.raw` as **“Primary production ingest”** (reasonable for plates, but easy to misread as train path). Flat `project-context.md` still says train on `.raw` plates. |
| **`src/sky/`** | `formats/sky-format-conversion.md` and root `sky-format-conversion.md` footer — should be **`scripts/sky/`** per migration doc. |
| **Wrong repo paths** | Widespread **`~/Projects/bayer_rgb`** and **`python -m bayer_rgb.*`** in **internal/** audits (expected historical); `internal/README.md` still says sync target **`bayer_rgb/docs/internal/`**. |

---

## 3. Index files — current?

| File | Verdict |
|------|---------|
| **`DOC_INDEX.md`** | **Mostly current** — correct `guides/` layout, `train_static_noisy_sequence`, rename guide, sync script name. **Gaps:** (1) no warning about **flat root duplicates**; (2) “Run `reorganize_docs_folders.sh`” reads **out of date** post-Context reorg; (3) `float16` linked under `guides/` only while `formats/` copy drifts. |
| **`README.md`** | **Current** — `bayer_denoiser`, `sync_context_to_repo.sh`, stub role. |
| **`project-context.md` (root)** | **Stale** — use **`guides/project-context.md`**. |
| **`guides/project-context.md`** | **Mostly current** on product direction (`.sky` train path, `train_static_noisy_sequence`). **Broken links:** `../../../jdd-reference-digest.md`, `../../../pipeline-setup-report.md`, `../../../preferences.md` (should be `../../preferences.md` and `../internal/...`). Still cites `pipeline_smoke.sh` for generic pipeline (GCP script). |

---

## 4. Categorization (actionable)

### Archive or merge

- **Remove or replace flat `docs/{project-context,real-data-runbook,training-user-guide,sky-format-conversion}.md`** with one-line stubs → canonical `guides/` or `formats/` paths (or delete after confirming no external links). These are the main source of **overfit** and **`.raw`-first** drift.
- **Merge `formats/float16-pipeline-standard.md` into `guides/float16-pipeline-standard.md`** (or vice versa) and keep a single file; delete duplicate or make `formats/` a symlink/stub in repo.
- **Drop root `docs/requirements-guide.md`** if identical to `operations/requirements-guide.md`.
- **`internal/archived/context-docs-2026-09-29/`** — **keep** as archive; do not treat as live docs (already noted in `DOC_INDEX`).
- **`guides/TRAINING_GUIDE.md`** — legacy duplicate of `training-user-guide.md`; **keep** with “prefer training-user-guide” banner only (already in index).

### Update in place

- **`guides/real-data-runbook.md`** — Align §1 wording with project-context: `.sky`/`.npy` = **forward training ingest**; `.raw` = production plate / Phase B / infer. Replace any remaining overfit language in §3 (canonical copy already uses `train_static_noisy_sequence` in most places). Fix links: `../../../local-machine-audit.md` → `../internal/local-machine-audit.md`; `../../../phase-b-paired-raw-spec.md` → `../internal/phase-b-paired-raw-spec.md`.
- **`guides/project-context.md`** — Fix internal/preferences links (see §3). Optionally clarify `pipeline_smoke.sh` = **GCP** smoke only.
- **`guides/training-user-guide.md`** — Fix `../../../phase-b-paired-raw-spec.md` → `../internal/phase-b-paired-raw-spec.md`.
- **`guides/float16-pipeline-standard.md`** — Fix `../../../float16-pipeline-implementation.md` → `../internal/float16-pipeline-implementation.md`.
- **`formats/sky-format-conversion.md`** (+ root duplicate if kept) — `src/sky/` → **`scripts/sky/`**; confirm CLI is `python scripts/sky/convert.py` or documented `bayer_denoiser.sky` module per repo.
- **`formats/float16-pipeline-standard.md`** — Replace `overfit_bayer_denoise_pair.py` with **`python -m bayer_denoiser.core.train_static_noisy_sequence`** (match `guides/` copy).
- **`inference/gaussian-gt-pipeline-test.md`** — Ensure Step 2+ references **`python -m bayer_denoiser.core.train_static_noisy_sequence`** only ( **`docs/`** version OK; **`files/docs/`** still says overfit).
- **`internal/README.md`** — `bayer_rgb` → **`bayer_denoiser`** paths.
- **`internal/phase-b-paired-raw-spec.md`** — Rename interim script; `bayer_rgb` → `bayer_denoiser` in “stock train” section.
- **`internal/float16-pipeline-implementation.md`**, **`local-machine-audit.md`**, **`sky-scripts-migration.md`**, **`pipeline-setup-report.md`**, **`bayer-denoise-implementation.md`**, **`runs-folder-inventory.md`** — Add **“Historical (pre-rename / pre-`src/` layout)”** header or refresh paths/CLIs; low priority for users if `internal/` is agent-only.
- **`DOC_INDEX.md`** — Note reorg **done in Context**; add **“do not edit flat root copies”** or list them for deletion; optional: link `formats/float16` only if merged.

### Keep as-is

- **`guides/rename-to-bayer-denoiser.md`**, **`guides/static-noisy-sequence.md`**, **`formats/RAW_SPEC.md`**, **`inference/a001c006-basic-denoise-test.md`**, **`operations/*`**, **`guides/MANUAL.md`**, **`BAYER_DENOISE.md`**, **`GCP_STILL.md`**, **`PIPELINE.md`** (with understanding `pipeline_smoke` = GCP).
- **`README.md`**, **`guides/training-user-guide.md`** (content direction already updated).

---

## 5. `docs/` vs `files/docs/` mismatch

- **Same file count (51)** and **same paths**; **`diff -rq`** shows **5 differing files**:

| Path | `docs/` (Context) | `files/docs/` (sync to Mac) |
|------|-------------------|-----------------------------|
| `guides/real-data-runbook.md` | `train_static_noisy_sequence` + sequence-dir example | Still **`overfit_bayer_denoise_pair.py`** CLI |
| `guides/training-user-guide.md` | Updated interim section | Stale overfit commands |
| `inference/gaussian-gt-pipeline-test.md` | `python -m bayer_denoiser.core.train_static_noisy_sequence` | `overfit_bayer_denoise_pair.py` |
| `internal/local-machine-audit.md` | (differs) | Likely rename/path drift |
| `internal/archived/.../README.md` | (differs) | Minor |

**Action:** After editing canonical **`docs/`**, run **`cp -a docs/* files/docs/`** (or project’s mirror step) so **`sync_context_to_repo.sh`** does not push stale overfit docs to the Mac repo.

---

## 6. Recommended fix order (minimal churn)

1. **Delete or stub flat root duplicates** under `docs/` (biggest stale surface).
2. **Reconcile `files/docs/`** with `docs/` for the five divergent files.
3. **Unify float16 standard** (one file; fix overfit script name).
4. **Fix `guides/*` broken `../../../` links** to `../internal/` and `../../preferences.md`.
5. **Clarify runbook §1** train vs ingest vs infer for `.raw` vs `.sky`/`.npy`.
6. **Refresh `internal/README.md`** + optional historical banners on pre-rename internal audits.
7. **Tweak `DOC_INDEX.md`** — reorg status + “canonical paths only under subfolders” (trivial index edit).

---

## 7. Cross-reference

Scripts audit (overfit removed from `scripts/`): [scripts-outdated-audit.md](scripts-outdated-audit.md).
