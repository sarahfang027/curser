"""Model registry: Bayer-space denoiser + small RGB demosaic baseline."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from bayer_denoiser.bayer_denoise_net import BayerDenoiseNet

MODEL_NAMES = ("demosaic", "bayer_denoise")

# Accepted on the CLI so we can raise a clear archived/deferred message
# instead of argparse "invalid choice" or a broken gcp_net import.
ARCHIVED_MODEL_ALIASES = frozenset({"gcp", "gcp-still", "gcp_net", "gcpstill"})

BAYER_SPACE_MODELS = frozenset({"bayer_denoise"})

_GCP_ARCHIVED = (
    "GCPStillNet (--model gcp) is archived/deferred. "
    "Use --model bayer_denoise. Source: src/archived/gcp/; notes: docs/archived/guides/GCP_STILL.md."
)


def is_bayer_space_model(name: str) -> bool:
    return name.lower().strip() in BAYER_SPACE_MODELS


def raise_if_archived_model(name: str) -> None:
    key = (name or "").lower().strip()
    if key in ARCHIVED_MODEL_ALIASES:
        raise ValueError(_GCP_ARCHIVED)


class DemosaicNet(nn.Module):
    """
    Input: packed RGGB (B,4,H/2,W/2)
    Output: RGB (B,3,H,W) in [0, 1]
    """

    def __init__(self, base_channels: int = 64, depth: int = 4):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(4, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        ]
        for _ in range(depth - 1):
            layers.extend(
                [
                    nn.Conv2d(base_channels, base_channels, 3, padding=1),
                    nn.ReLU(inplace=True),
                ]
            )
        layers.append(nn.Conv2d(base_channels, 12, 3, padding=1))
        self.body = nn.Sequential(*layers)

    def forward(self, packed: torch.Tensor, noise_map: torch.Tensor | None = None) -> torch.Tensor:
        del noise_map
        x = self.body(packed)
        x = F.pixel_shuffle(x, 2)
        return x.clamp(0.0, 1.0)


def bilinear_demosaic(packed: torch.Tensor) -> torch.Tensor:
    """Fast baseline: interpolate each plane to full resolution and merge G channels."""
    r = packed[:, 0:1]
    g1 = packed[:, 1:2]
    g2 = packed[:, 2:3]
    b = packed[:, 3:4]
    r_up = F.interpolate(r, scale_factor=2, mode="bilinear", align_corners=False)
    g_up = F.interpolate((g1 + g2) * 0.5, scale_factor=2, mode="bilinear", align_corners=False)
    b_up = F.interpolate(b, scale_factor=2, mode="bilinear", align_corners=False)
    return torch.cat([r_up, g_up, b_up], dim=1).clamp(0.0, 1.0)


def build_model(name: str = "bayer_denoise", nf: int = 64, base_channels: int = 32) -> nn.Module:
    key = name.lower().strip()
    raise_if_archived_model(key)
    if key == "demosaic":
        return DemosaicNet()
    if key in ("bayer_denoise", "bayer-denoise", "bayerdenoise"):
        return BayerDenoiseNet(base_channels=base_channels)
    raise ValueError(f"unknown model {name!r}; choose from {MODEL_NAMES}")
