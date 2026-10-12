---
cursor:
  subagentId: "bc-88a6dec0-01c4-5e31-a30f-dbb4665094f9"
---

# Archive GCP Python in Context `src/`

Context-only. Mac git repo is canonical; user syncs later.

## Inventory (19 `.py` under live `src/` before move)

GCP-specific: `gcp_net.py` (`GCPStillNet`, `make_noise_map`). Wired via `--model gcp` / aliases in `model.py`, `train.py`, `infer.py`.

Kept (not GCP-only): `bayer_denoise_net.py`, `bayer.py`, `dataset.py`, `decode.py`, `device_util.py`, `export_bayer.py`, `infer.py`, `__init__.py`, `make_sidecar.py`, `model.py` (`DemosaicNet`, `bilinear_demosaic`), `pairs.py`, `preview.py`, `raw_io.py`, `raw_spec.py`, `train.py`, `validate_pairs.py`, `validate_raw.py`, `visualize_crops.py`.

`files/src/` was missing; created as a mirror after edits.

## Moves / new paths

| Action | Path |
|--------|------|
| Moved | `src/gcp_net.py` → `src/archived/gcp/gcp_net.py` |
| New | `src/archived/README.md` (points at `docs/guides/GCP_STILL.md`) |
| New | `src/archived/gcp/README.md` (deferred, not imported) |
| New shim | `src/gcp_net.py` — `ImportError` (archived/deferred), no live net |

`docs/guides/GCP_STILL.md` left in place.

## Imports / CLI

- `import bayer_denoiser` — works (no `gcp_net` load).
- `--model bayer_denoise` — default on `train`; still in `MODEL_NAMES`.
- `--model gcp` (and `gcp-still`, `gcp_net`, `gcpstill`) — argparse accepts, then `raise_if_archived_model` → `SystemExit` / `ValueError` with archived/deferred text. No import of archived `GCPStillNet`.
- `from bayer_denoiser.gcp_net import …` — clear `ImportError` (shim).
- `DemosaicNet` / bilinear preview path unchanged; train/infer no longer call `make_noise_map`.

## Trees match

`src/` and `files/src/` checksum-identical (`diff -rq` clean). `__pycache__` removed.

## Scripts (not moved)

No live `scripts/*.py` is GCP-only. Leftover refs:

- `scripts/archived/pipeline_smoke.sh` — still `--model gcp`
- `scripts/archived/README.md` — GCP smoke note
- `scripts/archived/reorganize_docs_folders.sh` — historical `GCP_STILL.md` path
