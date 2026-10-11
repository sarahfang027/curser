---
cursor:
  subagentId: "bc-fe6e7553-5dfb-59cc-a9ae-0341e87ddbbe"
---

# Multi-frame JDD examples — reference digest (do not merge)

**Status:** reference only for [Bayer Denoising CNN](https://cursor.com/agents/bc-a64d420e-f2ca-403e-91f6-786d63f9c464). Canonical implementation stays in `~/Projects/bayer_rgb`.

## Where this material lives

| Source | ID / path |
|--------|-----------|
| User-facing name | **Multi-frame JDD examples** (desktop chat / composer survey) |
| Cloud agent (related) | [Multi-frame jdd models](https://cursor.com/agents/bc-bc4d1b47-b461-491c-9fe6-d95af5cede3a) — `bc-bc4d1b47-b461-491c-9fe6-d95af5cede3a` |
| Local canvases | `~/.cursor/projects/empty-window/canvases/multi-frame-JDD.canvas.tsx`, `still-video-JDD-roadmap.canvas.tsx` |
| Agent store on disk | No separate `files/` tree for the JDD chat; content is chat + canvas + web survey |

No dedicated **“Multi-frame JDD examples”** `bc-*` cloud agent was found in AgentStores; use the cloud agent above and local canvases as the stable links.

## What that thread covered

- **Landscape survey:** burst **JDD-B** (many RAW Bayer frames → one RGB still) vs **video JDD** (temporally consistent RGB sequence).
- **Methods table (public code):**
  - **GCP-Net** (TIP 2021) — green-channel prior, IntraF / **InterF** deformable alignment, merge; real-world burst RAW.
  - **2StageAlign** (CVPR 2022) — coarse block matching + fine deformable alignment for large inter-frame shift.
  - **RVDD** (WACV 2023) — recurrent video JDD; optical-flow motion compensation baseline.
  - **VJDD** (arXiv 2023) — latent propagation across time; DTC/RPC losses.
- **Videos:** RVDD talk (video JDD); Pixel handheld multi-frame SR + Deep Burst SR (burst align/fuse intuition). Canvas notes **no public talks** for GCP-Net / 2StageAlign / VJDD — use papers + code repos instead.
- **Roadmap canvas:** still vs video JDD decision tree (when burst fusion vs single-frame ISP-style path).

## Worth borrowing for `bayer_rgb`

- **Scope discipline:** this repo is **N=1 still GCP** today; burst InterF is intentionally omitted (`docs/GCP_STILL.md`, MANUAL §13).
- **Alignment narrative:** if capture becomes true multi-frame Bayer later, study **register → align features → merge** from GCP-Net / 2StageAlign — not copy-paste training code.
- **Evaluation habits:** PSNR on crops; side-by-side strips (already mirrored in `visualize_crops`); paper-style method comparison table for design reviews.
- **Noise map channel:** GCP-Net conditions on estimated noise; `bayer_rgb` uses `make_noise_map` + `--noise-std` — keep that parallel when adding real sensor noise models.

## Keep out of this project (unless requirements change)

- Replacing `GCPStillNet` with full **burst GCP-Net** without a burst data contract.
- **Video JDD** (RVDD/VJDD) temporal losses as the primary training objective.
- Treating the JDD canvas survey as a second repo to merge into `bayer_rgb`.
- Kodak / synthetic RGB burst benchmarks as the default dataset (production path is **float16 RGGB plates + EXR** per `RAW_SPEC`).

## Connection to current codebase

`bayer_rgb` already implements the **still fork** of GCP-Net (GCP branch + IntraF + upsample; **no InterF**). Multi-frame JDD reference material informs **future** burst extension and evaluation literacy — not the current default train/infer path.
