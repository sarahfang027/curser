# Documentation index (`bayer_denoiser`)

**Canonical tree:** `~/Projects/bayer_denoiser/docs/` on the Mac. Sync from Context with `scripts/sync_context_to_repo.sh`.

Paths below are **relative to `docs/`**. Active user docs sit at the **`docs/` root**; historical material is under **[archived/](archived/README.md)**.

**Mac note:** one-way rsync does not delete stale files on the Mac. After this layout change, remove old live subfolders locally if they still exist (`docs/guides/`, `docs/formats/`, `docs/operations/`, `docs/inference/`) and any duplicate copies of the six flat guides. Re-sync with delete if your script supports it.

---

## Start here

| Doc | Purpose |
|-----|---------|
| **[project-context.md](project-context.md)** | Goals, models, repo map, priorities — read first in this Project. |
| **[README.md](README.md)** | Stub: sync and how this folder relates to Context. |

---

## Training & dtype

| Doc | Purpose |
|-----|---------|
| [static-noisy-sequence.md](static-noisy-sequence.md) | Static burst + **`static_noisy_sequence_pipeline_test.sh`** (`runs/static-noisy-sequence-test/` train/compare layout). |
| [static-noisy-singleframe-pipeline-test.md](static-noisy-singleframe-pipeline-test.md) | 1-frame smoke + **`static_noisy_singleframe_pipeline_test.sh`** (`runs/static-noisy-singleframe-test/`). |
| [float16-pipeline-standard.md](float16-pipeline-standard.md) | Dtype policy: float16 on disk, float32 on device for U-Net and metrics. |

---

## Data prep & formats

| Doc | Purpose |
|-----|---------|
| [sky-format-conversion.md](sky-format-conversion.md) | `.sky` (csky) → float16 H×W `.npy` decode and layout notes. |

---

## Operations & environment

| Doc | Purpose |
|-----|---------|
| [requirements-guide.md](requirements-guide.md) | `requirements.txt` at repo root (includes imageio for archived EXR paths). |

---

## Archived documentation

Historical runbooks, repo manuals, extra inference write-ups, and agent working notes:

**[archived/README.md](archived/README.md)** — full archive index (`guides/`, `formats/`, `inference/`, `operations/`, `internal/`, `legacy-guides/` under `archived/`).
