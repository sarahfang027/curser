# Bayer Denoising CNN — project context

Living orientation doc for the **Bayer Denoising CNN** Project. Canonical code lives on Sarah’s Mac at `~/Projects/bayer_rgb`.

## Goals

- **Current focus:** train and run a **Bayer-space denoiser** (`BayerDenoiseNet`, `--model bayer_denoise`) — noisy packed RGGB → clean packed RGGB on headerless `.raw` plates and synthetic data.
- **Deferred (in repo):** still **GCPStillNet** joint demosaic + denoise to RGB (`docs/GCP_STILL.md`) for production-resolution RGGB float16 plates (17592×18024) when paired EXR is available.
- **Extend** the prior [Bayer Denoising CNN](https://cursor.com/agents/bc-a64d420e-f2ca-403e-91f6-786d63f9c464) implementation — do not replace it with burst JDD code.
- Default development on the **self-hosted Mac worker** (`~/Projects/bayer_rgb`); use cloud agents only when local context is not needed ([preferences](../../../preferences.md)).

## Active model: BayerDenoiseNet (`--model bayer_denoise`)

**Job:** denoise in **Bayer space** — same domain in and out (no RGB demosaic in this path).

| Piece | Detail |
|-------|--------|
| **Code** | `bayer_rgb/bayer_denoise_net.py` |
| **Topology** | Small **2-level U-Net** on packed RGGB |
| **Width** | `base_channels = max(16, nf // 2)` — default **32** channels when `--nf 64` |
| **Size** | ~**459k** parameters at default width |
| **Input / output** | Packed RGGB tensor `(B, 4, H/2, W/2)`, values in `[0, 1]` |
| **Infer output file** | Unpacked **float16 H×W mosaic** `.npy` (one scalar per Bayer site); [float16 pipeline standard](float16-pipeline-standard.md) |

## Codebase (`~/Projects/bayer_rgb`)

| Area | What exists |
|------|-------------|
| **Models** | **`bayer_denoise`** = `BayerDenoiseNet` (packed Bayer → packed Bayer); **`gcp`** = `GCPStillNet` (deferred RGB path); optional **`demosaic`** baseline |
| **Ingest** | `raw_spec.py` contract; `.raw` + sidecar; **`scripts/sky/`** (`.sky` → `.npy`); `validate_raw`, `make_sidecar` |
| **Decode** | `load_bayer` → mosaic H×W → pack RGGB `(4, H/2, W/2)`; crop-window reads for training |
| **Data** | `PairedRawBayerDataset` (random/center/full-frame crops), synthetic mode, optional `pairs.csv` |
| **Train / infer** | `python -m bayer_rgb.train`, `python -m bayer_rgb.infer` |
| **QA** | `validate_pairs`, `visualize_crops` (strip PNGs + PSNR/MAE vs bilinear), `preview` |
| **Docs** | `docs/MANUAL.md`, `docs/TRAINING_GUIDE.md`, `docs/PIPELINE.md`, `docs/BAYER_DENOISE.md`, `docs/RAW_SPEC.md`, `docs/GCP_STILL.md`; Context `sky-format-conversion.md`, **`float16-pipeline-standard.md`**; smoke scripts |
| **Sidecars** | `python -m bayer_rgb.make_sidecar` + [RAW_SPEC.md](../formats/RAW_SPEC.md) |
| **Layout** | Package root = **`src/`** (import name still **`bayer_rgb`** via `pyproject.toml`); repo root = `docs/`, `data/`, `scripts/` |
| **Git** | `pyproject.toml` — **`pip install -e .`**; `runs/` and `*.pt` gitignored |

**Explicitly not in repo:** full burst GCP-Net (InterF / deformable multi-frame alignment), tiled full-plate infer, camera ISP blocks.

### Recent local runs (`runs/`, on disk)

Typical Mac layout (gitignored): `bayer-denoise-smoke/`, `gaussian-gt-test/` for denoise; optional `crop_test/`, `crop_visual_check/` for crop QA. See [runs folder inventory](../internal/runs-folder-inventory.md). No `data/bayer` or `data/rgb` tree in checkout yet.

## Local run (venv, MPS, CLIs)

```bash
cd ~/Projects/bayer_rgb
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-raw.txt   # EXR / imageio — needed for paired EXR training
```

| Topic | Detail |
|-------|--------|
| **Python** | 3.9.x venv; torch 2.8.x, MPS available on Apple Silicon |
| **Device** | `train` / `infer` default to **MPS** on Apple Silicon when CUDA is absent; override with `--device cpu` |
| **Imports** | **`pip install -e .`** then `python -m bayer_rgb.*` from **repo root** (`~/Projects/bayer_rgb`) |
| **Synthetic smoke** | `./scripts/bayer_denoise_smoke.sh` → `runs/bayer-denoise-smoke/` (GCP still: [GCP_STILL.md](GCP_STILL.md), creates `runs/gcp-still-smoke/` when run) |
| **Real plates (target)** | `data/noisy/{stem}.raw` + `data/clean/{stem}.raw` (+ sidecars) — **stock `train` gap**; interim `.npy` pairs via `overfit_bayer_denoise_pair.py` ([training guide](training-user-guide.md#phase-b--real-plates-noisy--clean-bayer-raw)) |
| **Real plates (legacy stock)** | `data/bayer` (clean) + `data/rgb` (EXR alignment only) → `validate_pairs` → `train` with `--noise-std` |

See [local machine audit](../internal/local-machine-audit.md) for verified commands and gaps.

## Reference material — Multi-frame JDD (borrow ideas only)

The user’s **“Multi-frame JDD examples”** thread is a **desktop chat** (composer survey + canvases), not a second codebase in `bayer_rgb`. Related cloud chat: [Multi-frame jdd models](https://cursor.com/agents/bc-bc4d1b47-b461-491c-9fe6-d95af5cede3a) (`bc-bc4d1b47-b461-491c-9fe6-d95af5cede3a`). Local canvas artifacts: `~/.cursor/projects/empty-window/canvases/multi-frame-JDD.canvas.tsx`, `still-video-JDD-roadmap.canvas.tsx`.

**Borrow (patterns):** burst vs still scope; alignment-before-fusion intuition; public method map (GCP-Net, 2StageAlign, RVDD, VJDD); evaluation / explainer video list.

**Out of scope from JDD:** porting burst InterF stacks wholesale; video temporal JDD as the main product; Kodak RGB burst pipelines as default training data.

Detail: [JDD reference digest](../../../jdd-reference-digest.md).

## Pipeline (no dataset) — done

Synthetic **train → checkpoint → infer** is documented in `docs/PIPELINE.md` with `scripts/pipeline_smoke.sh`. MPS auto-selected on Mac; see [pipeline setup report](../../../pipeline-setup-report.md).

## Next development priorities

1. **Bayer denoise quality** — paired **noisy + clean** `.raw` (or `.npy` crops); implement noisy/clean in `train.py` / dataset; until then overfit scripts + infer QA.
2. **Data on Mac** — `data/noisy` + `data/clean` per stem (or legacy `data/bayer` + `data/rgb`); `validate_raw` on both trees before long runs.
3. **GCP / RGB path (later)** — paired training at `--image-size 256` with `--model gcp` when EXR references are ready; see `docs/GCP_STILL.md`.
4. **Production infer** — memory-safe path for 18K plates (tiled/sliding-window infer or export-crop workflow) — documented gap in MANUAL §8.

## Related agents

| Agent | Link |
|-------|------|
| Base implementation | [Bayer Denoising CNN](https://cursor.com/agents/bc-a64d420e-f2ca-403e-91f6-786d63f9c464) |
| Local machine audit | [Audit local bayer_rgb setup](https://cursor.com/agents/bc-658a3721-f35f-564a-ad4d-5eb2f16171ee) |
| This mapping / context | [Map agents to bayer_rgb](https://cursor.com/agents/bc-fe6e7553-5dfb-59cc-a9ae-0341e87ddbbe) |
| JDD survey (cloud) | [Multi-frame jdd models](https://cursor.com/agents/bc-bc4d1b47-b461-491c-9fe6-d95af5cede3a) |
| CNN pipeline setup | [Set up CNN pipeline](https://cursor.com/agents/bc-dabf708b-49d6-58e3-a714-9b79e160a84d) |
| Bayer-space denoiser | [Add Bayer-space denoiser](https://cursor.com/agents/bc-2054dd5c-ad7a-5fba-ad59-a4e2a85ff95d) |
