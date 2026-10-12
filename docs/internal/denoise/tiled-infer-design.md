---
cursor:
  subagentId: "bc-cfb413e1-8e22-561c-a6c3-a57f263cedc3"
---

# Tiled infer design (denoised_infer full plate)

## Behavior

- **`bayer_denoiser.tiled_infer.run_bayer_tiled_infer`**: sliding windows on H×W mosaic; each window is packed RGGB → U-Net → unpacked patch.
- **Tile origins**: even row/col (RGGB phase); last window along each axis snaps back so width/height stay **`tile_size`** when the mosaic is larger.
- **Stride**: `tile_size - overlap` (default **overlap=64**).
- **Stitch**: weighted average in **`float64`** on CPU. Weights are **1.0** on mosaic borders; **linear ramps** only on interior overlap (avoids zero weight at image corners).
- **Auto-tile**: when input H or W exceeds resolved **`tile_size`** (no single-forward OOM path). **`--tiled`** forces tiling even if input equals tile size (still one window).
- **`tile_size` resolve order**: CLI `--tile-size` → checkpoint **`crop_size`** → **512** (capped by mosaic for tiny inputs).

## Defaults unchanged for 512 smoke

- Input ≤ **`tile_size`**: single forward (same as before).
- Pipeline scripts: default still patch infer; **`FULL_INFER=1`** or **`INFER_TILED=1`** runs full **`SAMPLE`** with **`--tiled`**.

## Outputs

- **`compare/denoised_infer.npy`**: canonical; **`float16`** on save (float16 standard).
- Full mode also writes **`denoised_infer_patch.npy`** + **`denoised_patch.png`** (512 QA).

## Mac example

```bash
cd ~/Projects/bayer_denoiser && source .venv/bin/activate
FULL_INFER=1 bash scripts/static_noisy_singleframe_pipeline_test.sh
# or after train:
python -m bayer_denoiser.infer --checkpoint runs/static-noisy-singleframe-test/model.pt \
  --input data/samples/A001C006_0101_4E4951.npy \
  --output runs/static-noisy-singleframe-test/compare/denoised_infer.npy \
  --tiled --tile-size 512 --overlap 64
python -m bayer_denoiser.preview runs/static-noisy-singleframe-test/compare/denoised_infer.npy \
  --write-mosaic runs/static-noisy-singleframe-test/compare/denoised.png
```
