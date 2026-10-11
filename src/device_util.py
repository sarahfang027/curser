"""Default torch device string for train/infer CLIs."""

from __future__ import annotations

import torch


def default_device_string() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
