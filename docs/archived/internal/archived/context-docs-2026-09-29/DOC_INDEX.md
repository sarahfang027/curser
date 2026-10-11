# Documentation index (`bayer_rgb`)

**Canonical tree:** `~/Projects/bayer_rgb/docs/` on the Mac. Sync new Context docs with `scripts/sync_context_to_repo.sh`.

Paths below are **relative to `docs/`** (`guides/`, `formats/`, `operations/`, `inference/`). This archive mirrors that layout. Optional subfolders: see [docs-reorganization-proposal.md](../../docs-reorganization-proposal.md) in the Agent Store (`internal/`).

## Start here

| Doc | Purpose |
|-----|---------|
| **[guides/project-context.md](guides/project-context.md)** | Goals, models (`bayer_denoise` vs `gcp`), repo map, priorities — read first in this Project. |
| **[guides/real-data-runbook.md](guides/real-data-runbook.md)** | End-to-end local runbook when you have real plates (formats, QA, train/infer commands). |
| **[guides/MANUAL.md](guides/MANUAL.md)** | Original repo operator manual (CLI reference, data layout, production notes). *Repo git only — not in Context archive.* |
| **[README.md](README.md)** | How Context sync relates to this folder (stub after migration). |

---

## Training & models

| Doc | Purpose |
|-----|---------|
| [guides/training-user-guide.md](guides/training-user-guide.md) | **Current** Bayer-space training (`--model bayer_denoise`), Phase A/B, flags, troubleshooting. |
| [guides/TRAINING_GUIDE.md](guides/TRAINING_GUIDE.md) | Legacy / broader training guide in repo git; prefer `training-user-guide.md` for denoise focus. *Repo git only.* |
| [guides/BAYER_DENOISE.md](guides/BAYER_DENOISE.md) | `BayerDenoiseNet` summary and CLI pointers. *Repo git only.* |
| [guides/GCP_STILL.md](guides/GCP_STILL.md) | Deferred GCPStillNet RGB path (`--model gcp`). *Repo git only.* |
| [guides/float16-pipeline-standard.md](guides/float16-pipeline-standard.md) | Dtype policy: float16 on disk, float32 on device for U-Net and metrics. |

---

## Data prep & formats

| Doc | Purpose |
|-----|---------|
| [formats/RAW_SPEC.md](formats/RAW_SPEC.md) | Headerless `.raw` + sidecar JSON contract. *Repo git only.* |
| [formats/sky-format-conversion.md](formats/sky-format-conversion.md) | `.sky` (csky) → float16 H×W `.npy` decode and layout notes. |

---

## Inference & testing

| Doc | Purpose |
|-----|---------|
| [inference/README.md](inference/README.md) | Index of pipeline test write-ups under `inference/`. |
| [inference/gaussian-gt-pipeline-test.md](inference/gaussian-gt-pipeline-test.md) | Gaussian pseudo-GT → overfit → infer (tile and full frame). |
| [inference/a001c006-basic-denoise-test.md](inference/a001c006-basic-denoise-test.md) | Smoke infer on sky-sample crop (`A001C006`). |
| [guides/PIPELINE.md](guides/PIPELINE.md) | Synthetic train → checkpoint → infer smoke (`pipeline_smoke.sh`). *Repo git only.* |

---

## Operations & environment

| Doc | Purpose |
|-----|---------|
| [operations/requirements-guide.md](operations/requirements-guide.md) | When to install `requirements.txt` vs `requirements-raw.txt` (imageio / EXR). |
| [operations/licensing-notes.md](operations/licensing-notes.md) | Dependency license inventory and repo LICENSE gap (not legal advice). |

---

## Internal & agent notes

Synced from Context `internal/` → **`docs/internal/`** (handoffs, audits, implementation checklists — not primary user runbooks).

| Doc | Purpose |
|-----|---------|
| [internal/README.md](internal/README.md) | What belongs in `docs/internal/` and how sync works. |
| [internal/local-machine-audit.md](internal/local-machine-audit.md) | Mac venv, MPS, smoke commands, data-path gaps (2026-09-28). |
| [internal/phase-b-paired-raw-spec.md](internal/phase-b-paired-raw-spec.md) | Spec for noisy/clean paired `.raw` training (stock `train` gap). |
| [internal/float16-pipeline-implementation.md](internal/float16-pipeline-implementation.md) | Code checklist tied to float16 standard. |
| [internal/bayer-denoise-implementation.md](internal/bayer-denoise-implementation.md) | BayerDenoiseNet implementation notes. |
| [internal/pipeline-setup-report.md](internal/pipeline-setup-report.md) | CNN pipeline smoke setup report. |
| [internal/sky-scripts-migration.md](internal/sky-scripts-migration.md) | Move sky tooling to `scripts/sky/`. |
| [internal/jdd-reference-digest.md](internal/jdd-reference-digest.md) | Multi-frame JDD reference (borrow ideas only). |
| [internal/runs-folder-inventory.md](internal/runs-folder-inventory.md) | Local `runs/` artifact inventory. |
| [internal/denoise/](internal/denoise/) | Mac canonical paths, migration, cloud worker handoffs. |

**This folder** (`internal/archived/context-docs-2026-09-29/` in the Agent Store) is a **history mirror** of Context user docs before Mac-canonical migration; prefer the live repo `docs/` tree after sync.

---

## Inventory note (Context vs git)

| Source | Files |
|--------|--------|
| **In this archive** | `guides/project-context.md`, `guides/real-data-runbook.md`, `guides/training-user-guide.md`, `guides/float16-pipeline-standard.md`, `formats/sky-format-conversion.md`, `operations/requirements-guide.md`, `operations/licensing-notes.md`, `inference/*` |
| **Repo git only** | `guides/MANUAL.md`, `guides/TRAINING_GUIDE.md`, `formats/RAW_SPEC.md`, `guides/GCP_STILL.md`, `guides/BAYER_DENOISE.md`, `guides/PIPELINE.md` |
