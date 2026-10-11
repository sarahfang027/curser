"""Bayer-space denoiser: packed RGGB in → packed RGGB out."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BayerDenoiseNet(nn.Module):
    """
    Small U-Net on packed RGGB.

    Input:  (B, 4, H/2, W/2) noisy packed Bayer
    Output: (B, 4, H/2, W/2) denoised packed Bayer in [0, 1]
    """

    def __init__(self, base_channels: int = 32):
        super().__init__()
        c = base_channels
        self.enc1 = nn.Sequential(
            nn.Conv2d(4, c, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(
            nn.Conv2d(c, c * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c * 2, c * 2, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool2d(2)
        self.mid = nn.Sequential(
            nn.Conv2d(c * 2, c * 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c * 4, c * 4, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(c * 4, c * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c * 2, c * 2, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.up1 = nn.ConvTranspose2d(c * 2, c, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(c * 2, c, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, 4, 3, padding=1),
        )

    def forward(
        self,
        packed: torch.Tensor,
        noise_map: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del noise_map
        e1 = self.enc1(packed)
        e2 = self.enc2(self.pool1(e1))
        m = self.mid(self.pool2(e2))
        d2 = self.up2(m)
        if d2.shape[-2:] != e2.shape[-2:]:
            d2 = F.interpolate(d2, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        if d1.shape[-2:] != e1.shape[-2:]:
            d1 = F.interpolate(d1, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        out = self.dec1(torch.cat([d1, e1], dim=1))
        return out.clamp(0.0, 1.0)
