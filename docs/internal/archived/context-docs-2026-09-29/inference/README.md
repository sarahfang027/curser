# Inference & pipeline test docs (Mac)

**Canonical location:** `~/Projects/bayer_rgb/docs/inference/` after you run:

```bash
bash ~/Projects/bayer_rgb/scripts/sync_context_to_repo.sh
```

(or copy from Cursor Project Context once, then use the repo paths below.)

| Doc | Purpose |
|-----|---------|
| [gaussian-gt-pipeline-test.md](gaussian-gt-pipeline-test.md) | Gaussian pseudo-GT (tile + **full frame**) → overfit → `bayer_rgb.infer` |
| [a001c006-basic-denoise-test.md](a001c006-basic-denoise-test.md) | CNN infer smoke on sky sample crop |

Matching scripts live in **`~/Projects/bayer_rgb/scripts/`** (`gaussian_gt_pipeline_test.sh`, etc.).

Project-wide guides in Context (`training-user-guide.md`, `sky-format-conversion.md`) can be copied into `docs/` the same way if you want a full offline mirror.
