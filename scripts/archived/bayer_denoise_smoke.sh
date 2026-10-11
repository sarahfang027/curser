#!/usr/bin/env bash
# Synthetic Bayer-space denoise smoke: train → checkpoint → infer (.npy).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
OUT="${OUT:-runs/bayer-denoise-smoke}"

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

python -m bayer_denoiser.archived.core.train \
  --model bayer_denoise \
  --epochs 3 \
  --batch-size 4 \
  --image-size 64 \
  --synthetic-length 32 \
  --noise-std 0.05 \
  --output-dir "$OUT"

python - <<PY
import numpy as np
import torch
from pathlib import Path
from bayer_denoiser.sort_bayer import rgb_to_bayer

out = Path("$OUT") / "noisy_mosaic.npy"
out.parent.mkdir(parents=True, exist_ok=True)
rgb = torch.rand(1, 3, 64, 64)
clean = rgb_to_bayer(rgb).squeeze().numpy()
noisy = (clean + np.random.randn(*clean.shape).astype(np.float32) * 0.05).clip(0, 1)
np.save(out, noisy)
np.save(Path("$OUT") / "clean_mosaic.npy", clean)
print(f"Wrote {out}")
PY

python -m bayer_denoiser.infer \
  --checkpoint "$OUT/model.pt" \
  --input "$OUT/noisy_mosaic.npy" \
  --output "$OUT/denoised_mosaic.npy"

python - <<PY
import json
import numpy as np
from pathlib import Path

out = Path("$OUT")
noisy = np.load(out / "noisy_mosaic.npy")
clean = np.load(out / "clean_mosaic.npy")
den = np.load(out / "denoised_mosaic.npy")
hist = json.loads((out / "history.json").read_text())
last = hist[-1]

def psnr(a, b):
    mse = float(np.mean((a - b) ** 2))
    return 99.0 if mse <= 0 else 10.0 * np.log10(1.0 / mse)

print(f"train_last_loss={last['loss']:.5f} train_last_psnr={last['psnr']:.2f} dB")
print(f"infer_psnr_noisy_vs_clean={psnr(noisy, clean):.2f} dB")
print(f"infer_psnr_denoised_vs_clean={psnr(den, clean):.2f} dB")
PY

echo "OK: $OUT/model.pt → $OUT/denoised_mosaic.npy"
