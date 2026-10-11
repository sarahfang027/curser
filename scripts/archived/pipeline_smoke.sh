#!/usr/bin/env bash
# End-to-end synthetic smoke: train (GCP) → checkpoint → infer.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
OUT="${OUT:-runs/pipeline-smoke}"

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

python -m bayer_denoiser.archived.core.train \
  --model gcp \
  --epochs 2 \
  --batch-size 2 \
  --image-size 64 \
  --synthetic-length 16 \
  --noise-std 0.02 \
  --output-dir "$OUT"

python - <<PY
import numpy as np
import torch
from pathlib import Path
from bayer_denoiser.sort_bayer import rgb_to_bayer

out = Path("$OUT") / "mosaic.npy"
out.parent.mkdir(parents=True, exist_ok=True)
np.save(out, rgb_to_bayer(torch.rand(1, 3, 64, 64)).squeeze().numpy())
print(f"Wrote {out}")
PY

python -m bayer_denoiser.infer \
  --checkpoint "$OUT/model.pt" \
  --input "$OUT/mosaic.npy" \
  --output "$OUT/out_rgb.png"

echo "OK: $OUT/model.pt → $OUT/out_rgb.png"
