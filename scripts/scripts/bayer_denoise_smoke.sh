#!/usr/bin/env bash
# Deprecated: stock synthetic train used bayer_denoiser.core.train (now archived).
set -euo pipefail
echo "bayer_denoise_smoke.sh is retired. Use static-sequence training instead:" >&2
echo "  docs/static-noisy-sequence.md" >&2
echo "  python python -m bayer_denoiser.train_static_noisy_sequence --help" >&2
echo "Legacy smoke (archived train): scripts/archived/bayer_denoise_smoke.sh" >&2
exit 1
