# Bayer Denoiser (`bayer_denoiser`) — project context

Living orientation doc for the **Bayer Denoising CNN** Project (repo/package name **`bayer_denoiser`**, formerly `bayer_rgb`). Canonical code lives on Sarah’s Mac at `~/Projects/bayer_denoiser`.

**Scope:** **Bayer-space denoise** on packed **RGGB** (`BayerDenoiseNet`, `--model bayer_denoise`). The GCP / raw→RGB (joint demosaic + denoise) training path is **deferred** — not the product focus and not what the name `bayer_denoiser` means. Rename checklist: [rename-to-bayer-denoiser.md](archived/guides/rename-to-bayer-denoiser.md).

## Goals

- **Current focus:** train and run a **Bayer-space denoiser** (`BayerDenoiseNet`, `--model bayer_denoise`) — noisy → clean packed RGGB. **Training ingest:** **`.sky` (csky)** → float16 `.npy` mosaics ([sky-format-conversion.md](sky-format-conversion.md)); synthetic smoke unchanged. Headerless `.raw` is not the forward training path.
- **Deferred (in repo):** still **GCPStillNet** joint demosaic + denoise to RGB (`docs/GCP_STILL.md`) for production-resolution RGGB float16 plates (17592×18024) when paired EXR is available.
- **Extend** the prior [Bayer Denoising CNN](https://cursor.com/agents/bc-a64d420e-f2ca-403e-91f6-786d63f9c464) implementation — do not replace it with burst JDD code.
- Default development on the **self-hosted Mac worker** (`~/Projects/bayer_denoiser`); use cloud agents only when local context is not needed ([preferences](../preferences.md) — Project Context root, not in git).

## Active model: BayerDenoiseNet (`--model bayer_denoise`)

**Job:** denoise in **Bayer space** — same domain in and out (no RGB demosaic in this path).

| Piece | Detail |
|-------|--------|
| **Code** | `src/bayer_denoise_net.py` (`bayer_denoiser.bayer_denoise_net`) |
| **Topology** | Small **2-level U-Net** on packed RGGB |
| **Width** | `base_channels = max(16, nf // 2)` — default **32** channels when `--nf 64` |
| **Size** | ~**459k** parameters at default width |
| **Input / output** | Packed RGGB tensor `(B, 4, H/2, W/2)`, values in `[0, 1]` |
| **Infer output file** | Unpacked **float16 H×W mosaic** `.npy` (one scalar per Bayer site); [float16 pipeline standard](float16-pipeline-standard.md) |

## Codebase (`~/Projects/bayer_denoiser`)

| Area | What exists |
|------|-------------|
| **Models** | **`bayer_denoise`** = `BayerDenoiseNet` (packed Bayer → packed Bayer); **`gcp`** = `GCPStillNet` (deferred RGB path); optional **`demosaic`** baseline |
| **Data** | Static noisy sequences (`clean_mosaic.npy`, `frame_*.npy`) for plumbing; target real data as paired float16 **`.npy`** per clip/stem. Stock paired **`.raw` / EXR** dataset + `pairs.csv` only on **`archived.core.train`** |
| **Train / infer** | `python -m bayer_denoiser.train_static_noisy_sequence`; infer `python -m bayer_denoiser.infer` (legacy stock train: `python -m bayer_denoiser.archived.core.train`) |
| **QA** | **Active:** `python -m bayer_denoiser.preview` (stats, optional mosaic/RGB PNG). **Archived:** `archived.qa` (`validate_pairs`, `visualize_crops`, `validate_raw`, …) |
| **Docs** | Live (flat `docs/` root): `project-context.md`, `static-noisy-sequence.md`, `float16-pipeline-standard.md`, `sky-format-conversion.md`, `static-noisy-singleframe-pipeline-test.md`, `requirements-guide.md` — index **[DOC_INDEX.md](DOC_INDEX.md)**; older runbooks under `archived/` |
| **Bayer packing** | RGGB pack/unpack is centralized in **`src/sort_bayer.py`** (torch `bayer_mosaic_to_packed` / `packed_to_bayer_mosaic`; NumPy twins `mosaic_to_packed_np` / `packed_to_mosaic_np`). Channel order **R, G1, G2, B**. **Checkpoints trained before this change (swapped G1/G2 in `train_static_noisy_sequence`) need retrain** for infer alignment |
| **Layout** | Flat active modules at **`src/`** root (`infer`, `sort_bayer`, `preview`, …); **`src/archived/`** subfolders only; import name **`bayer_denoiser`** via `pyproject.toml`; repo root = `docs/`, `data/`, `scripts/` |
| **Git** | `pyproject.toml` — **`pip install -e .`**; `runs/` and `*.pt` gitignored |

**Explicitly not in repo:** full burst GCP-Net (InterF / deformable multi-frame alignment), camera ISP blocks.

### Recent local runs (`runs/`, on disk)

Typical Mac layout (gitignored): `bayer-denoise-smoke/`, **`static-noisy-singleframe-test/`** (1-frame smoke) and **`static-noisy-sequence-test/`** (multi-frame burst) for denoise plumbing — each RUN has **`train/`** (data only), **RUN root** checkpoints (`model.pt`, `model_epoch*.pt`, `history.json`), and **`compare/`** with canonical **`denoised_infer.npy`** + preview PNGs (optional **`denoised_mosaic.npy`** only with `--write-denoised-mosaic` or `DEBUG_DENOISED_MOSAIC=1`). Single-frame fixtures: `static-noisy-singleframe-test/train/` (`gaussian_GT_crop.npy`, `tile_train_seq/`); static burst: **`static-noisy-sequence-test/train/static_sequence/`** (`clean_mosaic.npy`, `frame_*.npy`). Scripts: `scripts/static_noisy_singleframe_pipeline_test.sh`, **`scripts/static_noisy_sequence_pipeline_test.sh`**. Legacy **`runs/static_sequence/`** (top-level) and old **`runs/gaussian-gt-test/`** still work if you keep or rename them locally. Optional `crop_test/`, `crop_visual_check/` for crop QA. See [runs folder inventory](archived/internal/runs-folder-inventory.md). No `data/bayer` or `data/rgb` tree in checkout yet.

## Local run (venv, MPS, CLIs)

```bash
cd ~/Projects/bayer_denoiser
source .venv/bin/activate
pip install -r requirements.txt   # includes imageio (legacy EXR / archived paths)
pip install -e .
```

| Topic | Detail |
|-------|--------|
| **Python** | 3.9.x venv; torch 2.8.x, MPS available on Apple Silicon |
| **Device** | `train` / `infer` default to **MPS** on Apple Silicon when CUDA is absent; override with `--device cpu` |
| **Imports** | **`pip install -e .`** then `python -m bayer_denoiser.*` from **repo root** (`~/Projects/bayer_denoiser`) |
| **Plumbing smoke** | `python -m bayer_denoiser.train_static_noisy_sequence` — [static-noisy-sequence.md](static-noisy-sequence.md) (GCP still: [GCP_STILL.md](archived/guides/GCP_STILL.md), `scripts/archived/pipeline_smoke.sh`) |
| **Real plates (target)** | Ingest **`.sky` → `.npy`** (`scripts/sky/convert.py`); build paired noisy/clean **`.npy`** folders; train with static-sequence or future paired loader — see [static-noisy-sequence.md](static-noisy-sequence.md) and archived [training guide](archived/guides/training-user-guide.md) |
| **Real plates (legacy stock)** | Paired **`.raw` / EXR** + `validate_pairs` → **`archived.core.train`** only (deferred for Bayer denoise focus) |

See [local machine audit](archived/internal/local-machine-audit.md) for verified commands and gaps.

## Reference material — Multi-frame JDD (borrow ideas only)

The user’s **“Multi-frame JDD examples”** thread is a **desktop chat** (composer survey + canvases), not a second codebase in `bayer_denoiser`. Related cloud chat: [Multi-frame jdd models](https://cursor.com/agents/bc-bc4d1b47-b461-491c-9fe6-d95af5cede3a) (`bc-bc4d1b47-b461-491c-9fe6-d95af5cede3a`). Local canvas artifacts: `~/.cursor/projects/empty-window/canvases/multi-frame-JDD.canvas.tsx`, `still-video-JDD-roadmap.canvas.tsx`.

**Borrow (patterns):** burst vs still scope; alignment-before-fusion intuition; public method map (GCP-Net, 2StageAlign, RVDD, VJDD); evaluation / explainer video list.

**Out of scope from JDD:** porting burst InterF stacks wholesale; video temporal JDD as the main product; Kodak RGB burst pipelines as default training data.

Detail: [JDD reference digest](archived/internal/jdd-reference-digest.md).

## Pipeline (no dataset) — done

Synthetic **train → checkpoint → infer** (GCP path) is in [PIPELINE.md](archived/guides/PIPELINE.md) with `scripts/archived/pipeline_smoke.sh`. Bayer denoise plumbing: [static-noisy-sequence.md](static-noisy-sequence.md). MPS on Mac; see [pipeline setup report](archived/internal/pipeline-setup-report.md).

## Next development priorities

1. **Bayer denoise quality** — paired **noisy + clean** float16 **`.npy`** (full plate or crops); extend training beyond static-sequence plumbing; infer + **`preview`** QA on patches.
2. **Multi-crop dataset** — training set with **many aligned crops per plate** on both **noisy input** and **GT** (not only one fixed center patch); feeds better coverage before full-plate infer.
3. **Production infer** — tiled sliding-window on full plates via **`bayer_denoiser.infer --tiled`** (see [static-noisy-sequence.md](static-noisy-sequence.md)); train still uses crops on Mac.

## Related agents

| Agent | Link |
|-------|------|
| Base implementation | [Bayer Denoising CNN](https://cursor.com/agents/bc-a64d420e-f2ca-403e-91f6-786d63f9c464) |
| Local machine audit | [Audit local bayer_rgb setup](https://cursor.com/agents/bc-658a3721-f35f-564a-ad4d-5eb2f16171ee) |
| This mapping / context | [Map agents to bayer_rgb](https://cursor.com/agents/bc-fe6e7553-5dfb-59cc-a9ae-0341e87ddbbe) |
| JDD survey (cloud) | [Multi-frame jdd models](https://cursor.com/agents/bc-bc4d1b47-b461-491c-9fe6-d95af5cede3a) |
| CNN pipeline setup | [Set up CNN pipeline](https://cursor.com/agents/bc-dabf708b-49d6-58e3-a714-9b79e160a84d) |
| Bayer-space denoiser | [Add Bayer-space denoiser](https://cursor.com/agents/bc-2054dd5c-ad7a-5fba-ad59-a4e2a85ff95d) |
