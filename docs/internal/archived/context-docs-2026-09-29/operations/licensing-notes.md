# Commercial licensing inventory (bayer_rgb)

Factual inventory for `~/Projects/bayer_rgb` as of 2026-09-28. **Not legal advice** — confirm with counsel before shipping a product.

## Repo license status

| Item | Finding |
|------|---------|
| Root `LICENSE` | **Absent** (not in git) |
| `COPYING` / `NOTICE` | **Absent** |
| `README` with license | **Absent** (no root README in git) |
| Git submodules (`.gitmodules`) | **None** |
| SPDX / per-file headers in `bayer_rgb/*.py` | **None** (only module docstrings) |

**Implication:** The repository does not state terms for **your own project code**. For commercial distribution of this codebase (source or binary), you typically want an explicit license (e.g. proprietary notice or an OSS license) and, if you rely on third-party code, corresponding **NOTICE** attribution files.

Tracked scope (git): Python package `bayer_rgb/`, `docs/`, `scripts/`, `requirements*.txt`. Sidecar geometry: `make_sidecar` + `docs/formats/RAW_SPEC.md`. No vendored third-party source trees.

## Python dependencies

Declared in repo root only (no lockfile). Typical **SPDX-style** license classes (verify at install time with `pip-licenses` or package metadata).

### `requirements.txt` (core)

| Package | Min version | Typical license class | Copyleft? |
|---------|-------------|------------------------|-----------|
| `torch` | 2.0 | BSD-3-Clause (PyTorch) | No |
| `torchvision` | 0.15 | BSD-3-Clause | No |
| `numpy` | 1.24 | BSD-3-Clause | No |
| `Pillow` | 10.0 | HPND (PIL-derived; permissive) | No |
| `tqdm` | 4.65 | MPL-2.0 | Weak (file-level if you modify `tqdm` itself) |

### `requirements-raw.txt` (optional)

| Package | Min version | Typical license class | Copyleft? |
|---------|-------------|------------------------|-----------|
| `imageio` | 2.34.0 | BSD-2-Clause | No |

**Transitive dependencies:** `pip install` pulls many additional wheels (e.g. `typing_extensions`, `filelock`, `sympy`, `networkx`, `jinja2`, and for GPU builds NVIDIA CUDA/cuDNN-related components). Those have their own licenses. For commercial binaries, audit the **exact** environment you ship (CPU vs CUDA wheel, OS).

**Optional code path:** `.exr` RGB loading in `bayer_rgb/dataset.py` requires `imageio`; without it, PNG/JPEG paths use Pillow only.

## Code provenance

| Area | Assessment | Key files / references |
|------|------------|-------------------------|
| RAW ingest, sidecars, pairs, train/infer CLI | Appears **original to this repo** (no upstream file copies found) | `bayer_rgb/raw_spec.py`, `raw_io.py`, `decode.py`, `train.py`, `infer.py`, `dataset.py` |
| Small baselines | **Original** small PyTorch modules | `bayer_rgb/model.py` (`DemosaicNet`, `bilinear_demosaic`), `bayer_rgb/bayer_denoise_net.py` |
| Still GCP-Net | **Paper-inspired reimplementation**, not vendored upstream source | `bayer_rgb/gcp_net.py` — docstring cites Guo et al. TIP2021; burst InterF / PCD / DCN **omitted** (still N=1) |
| Documentation | Describes adaptation vs [GuoShi28/GCP-Net](https://github.com/GuoShi28/GCP-Net) | `docs/GCP_STILL.md`, `docs/MANUAL.md` (links to [arXiv:2101.09870](https://arxiv.org/abs/2101.09870)) |
| Vendored / copied code | **None found** in tracked tree | No `vendor/`, no git submodules, no duplicated third-party file headers |
| Scripts | Thin smoke wrappers | `scripts/pipeline_smoke.sh`, `scripts/bayer_denoise_smoke.sh` |

**Upstream GCP-Net (external):** GitHub reports **Apache-2.0** for [GuoShi28/GCP-Net](https://github.com/GuoShi28/GCP-Net). This repo’s `gcp_net.py` is a separate implementation aligned with the paper architecture; it is **not** a subtree copy of that repository. If you later **port or paste** upstream files or use their **official checkpoints**, Apache-2.0 terms and attribution requirements apply to that material.

**Paper:** The arXiv publication is academic work; implementing a described architecture is separate from copying their code, but patents and publication attribution are outside this inventory.

## Weights and data

| Item | Finding |
|------|---------|
| Pretrained weights in repo | **None** tracked; `.gitignore` excludes `*.pt` and `runs/` |
| Weight download URLs in code | **None** (no `http`/`huggingface`/`drive` fetch logic in `bayer_rgb/`) |
| Training output | Local `runs/.../model.pt` from `bayer_rgb/train.py` — **your** trained weights |
| Official GCP-Net pretrained weights | **Not bundled**; docs point users to the [original repository](https://github.com/GuoShi28/GCP-Net) if they want burst/pretrained assets |
| Datasets in git | No imagery in git; sidecar geometry via `make_sidecar` + `docs/formats/RAW_SPEC.md` |
| Production RAW/RGB plates | Expected **outside** git (`data/bayer`, `data/rgb` per docs) — license is **your data governance**, not this repo |

Using upstream GCP-Net checkpoints in a commercial product would require complying with **their** license and any **dataset / model release** terms posted in that repo (not audited here).

## Practical checklist for commercial use

1. **Add a project license** at repo root (`LICENSE` + optional `NOTICE`) so your code’s redistribution terms are explicit.
2. **Generate a third-party notice** for the exact Python env you deploy (`pip-licenses`, conda export, or SBOM tooling).
3. **Separate stacks:** Bayer-denoise-only (`requirements.txt`) vs EXR/GCP paths (`+ requirements-raw.txt`) — document which you ship.
4. **PyTorch distribution:** If you ship CUDA builds, include NVIDIA/cuDNN license notices as required by those components.
5. **GCP path:** Treat `gcp_net.py` as inspired-by-paper code; if legal review wants alignment with Apache-2.0 upstream, consider explicit attribution in `NOTICE` and counsel review before mixing upstream files or weights.
6. **Training data & checkpoints:** Ensure rights to RAW/RGB training data and clarify whether shipped `model.pt` is proprietary or derived from licensed data.
7. **No automatic compliance** from using permissive deps alone — missing root license on **your** code is the largest repo-level gap for redistribution.

## Disclaimer

This document is an **engineering inventory** based on filesystem and git inspection of `bayer_rgb` and public metadata for [GuoShi28/GCP-Net](https://github.com/GuoShi28/GCP-Net). It does not interpret licenses for your jurisdiction, product model, or data contracts. Consult qualified legal counsel before commercial deployment, especially for model weights, customer data, and GPU stack redistribution.
