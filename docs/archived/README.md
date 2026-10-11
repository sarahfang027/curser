# Archived documentation

These files are **kept for reference** but are not part of the active `docs/` tree. Prefer the live docs listed in **[DOC_INDEX.md](../DOC_INDEX.md)**.

Nothing here is deleted — older runbooks were moved under `archived/` (2026-10-07). **Live** user docs now sit flat at `docs/` root (no top-level `guides/` / `formats/` / `operations/` / `inference/`). Links inside this archive may still use pre-flat paths; use **[DOC_INDEX.md](../DOC_INDEX.md)** for current live filenames.

## Guides

| Doc | Notes |
|-----|--------|
| [guides/rename-to-bayer-denoiser.md](guides/rename-to-bayer-denoiser.md) | Mac checklist: `bayer_rgb` → `bayer_denoiser`. |
| [guides/real-data-runbook.md](guides/real-data-runbook.md) | End-to-end local runbook for real plates. |
| [guides/training-user-guide.md](guides/training-user-guide.md) | Bayer-space training (Phase A/B), flags, troubleshooting. |
| [guides/BAYER_DENOISE.md](guides/BAYER_DENOISE.md) | `BayerDenoiseNet` summary and CLI pointers. |
| [guides/GCP_STILL.md](guides/GCP_STILL.md) | Deferred GCPStillNet RGB path (`--model gcp`). |
| [guides/MANUAL.md](guides/MANUAL.md) | Original repo operator manual. |
| [guides/PIPELINE.md](guides/PIPELINE.md) | Synthetic train → checkpoint → infer smoke. |

## Formats

| Doc | Notes |
|-----|--------|
| [formats/RAW_SPEC.md](formats/RAW_SPEC.md) | Headerless `.raw` + sidecar JSON contract. |

## Inference

| Doc | Notes |
|-----|--------|
| [inference/README.md](inference/README.md) | Index of archived pipeline test write-ups. |
| [inference/a001c006-basic-denoise-test.md](inference/a001c006-basic-denoise-test.md) | Smoke infer on sky-sample crop (`A001C006`). |

## Operations

| Doc | Notes |
|-----|--------|
| [operations/licensing-notes.md](operations/licensing-notes.md) | Dependency license inventory (not legal advice). |

## Internal & agent notes

Synced agent handoffs, audits, and implementation checklists (formerly `docs/internal/`):

| Doc | Purpose |
|-----|---------|
| [internal/README.md](internal/README.md) | What belonged in `docs/internal/`. |
| [internal/local-machine-audit.md](internal/local-machine-audit.md) | Mac venv, MPS, smoke commands. |
| [internal/phase-b-paired-raw-spec.md](internal/phase-b-paired-raw-spec.md) | Noisy/clean paired `.raw` training spec. |
| [internal/float16-pipeline-implementation.md](internal/float16-pipeline-implementation.md) | Code checklist tied to float16 standard. |
| [internal/bayer-denoise-implementation.md](internal/bayer-denoise-implementation.md) | BayerDenoiseNet implementation notes. |
| [internal/pipeline-setup-report.md](internal/pipeline-setup-report.md) | CNN pipeline smoke setup report. |
| [internal/sky-scripts-migration.md](internal/sky-scripts-migration.md) | Move sky tooling to `scripts/sky/`. |
| [internal/jdd-reference-digest.md](internal/jdd-reference-digest.md) | Multi-frame JDD reference (borrow ideas only). |
| [internal/runs-folder-inventory.md](internal/runs-folder-inventory.md) | Local `runs/` artifact inventory. |
| [internal/docs-outdated-audit.md](internal/docs-outdated-audit.md) | Docs drift audit. |
| [internal/docs-reorganization-proposal.md](internal/docs-reorganization-proposal.md) | Prior reorg proposal. |
| [internal/float16-pipeline-docs-unify.md](internal/float16-pipeline-docs-unify.md) | Float16 doc unify notes. |
| [internal/scripts-outdated-audit.md](internal/scripts-outdated-audit.md) | Scripts drift audit. |
| [internal/denoise/](internal/denoise/) | Mac canonical paths, migration, cloud worker handoffs. |
| [internal/archived/context-docs-2026-09-29/](internal/archived/context-docs-2026-09-29/) | Historical Context mirror (pre-migration snapshot). |

## Legacy guides (pre–folder layout)

| Doc | Notes |
|-----|--------|
| [legacy-guides/README.md](legacy-guides/README.md) | Pointer to legacy training guide. |
| [legacy-guides/TRAINING_GUIDE.md](legacy-guides/TRAINING_GUIDE.md) | Older training guide. |
