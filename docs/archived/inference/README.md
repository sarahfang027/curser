# Inference & pipeline test docs (Mac)

**Canonical location:** `~/Projects/bayer_denoiser/docs/inference/` after you run:

```bash
bash ~/Projects/bayer_denoiser/scripts/sync_context_to_repo.sh
```

(or copy from Cursor Project Context once, then use the repo paths below.)

| Doc | Purpose |
|-----|---------|
| [static-noisy-singleframe-pipeline-test.md](static-noisy-singleframe-pipeline-test.md) | Gaussian pseudo-GT (tile + **full frame**) → overfit → `bayer_denoiser.infer` |
| [a001c006-basic-denoise-test.md](a001c006-basic-denoise-test.md) | CNN infer smoke on sky sample crop |

Matching scripts live in **`~/Projects/bayer_denoiser/scripts/`** (`static_noisy_singleframe_pipeline_test.sh`, etc.).

Project-wide guides in Context (`training-user-guide.md`, `sky-format-conversion.md`) can be copied into `docs/` the same way if you want a full offline mirror.
