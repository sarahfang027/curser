"""Still-adapted GCP-Net (TIP2021), single-frame demosaic + denoise.

Based on Guo et al. "Joint Denoising and Demosaicking with Green Channel Prior
for Real-world Burst Images". Burst InterF / PCD / DCN alignment is omitted;
N=1 identity merge is used for still study.

Input:  packed RGGB (B, 4, H/2, W/2) and optional noise map (same shape)
Output: RGB (B, 3, H, W) in [0, 1]
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


def _init_conv(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            init.orthogonal_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.ConvTranspose2d):
            init.orthogonal_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0)


class SimpleBlock(nn.Module):
    def __init__(
        self,
        depth: int = 3,
        n_channels: int = 64,
        input_channels: int = 3,
        output_channel: int = 64,
        kernel_size: int = 3,
    ):
        super().__init__()
        padding = kernel_size // 2
        layers: list[nn.Module] = [
            nn.Conv2d(input_channels, n_channels, kernel_size, padding=padding, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
        ]
        for _ in range(max(depth - 2, 0)):
            layers.extend(
                [
                    nn.Conv2d(n_channels, n_channels, kernel_size, padding=padding, bias=False),
                    nn.LeakyReLU(0.1, inplace=True),
                ]
            )
        layers.append(nn.Conv2d(n_channels, output_channel, kernel_size, padding=padding, bias=False))
        self.body = nn.Sequential(*layers)
        _init_conv(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class DualABlock(nn.Module):
    """Channel + spatial attention residual block (DualA in GCP-Net)."""

    def __init__(self, n_channels: int = 64, input_channels: int | None = None):
        super().__init__()
        c = input_channels or n_channels
        self.square_conv = nn.Conv2d(c, n_channels, 3, padding=1, bias=False)
        self.relu = nn.LeakyReLU(0.1, inplace=True)
        self.extract_conv = nn.Conv2d(n_channels, n_channels, 3, padding=1, bias=True)
        self.res_block1 = SimpleBlock(2, n_channels, n_channels, n_channels)
        self.res_block2 = SimpleBlock(2, n_channels, n_channels, n_channels)
        self.down = nn.Conv2d(n_channels, n_channels // 2, 1, bias=True)
        self.up = nn.Conv2d(n_channels // 2, n_channels, 1, bias=True)
        self.spatial_att = nn.Conv2d(n_channels, 1, 7, padding=3, bias=True)
        _init_conv(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_temp = self.relu(self.square_conv(x))
        x_temp = self.extract_conv(x_temp)
        x_temp = x + x_temp
        x_temp = x_temp + self.res_block1(x_temp)
        x_temp = x_temp + self.res_block2(x_temp)

        x_se = F.adaptive_avg_pool2d(x_temp, 1)
        x_se = torch.sigmoid(self.up(self.relu(self.down(x_se))))
        x_sp = torch.sigmoid(self.spatial_att(x_temp))
        return x_temp + x_temp * x_se + x_temp * x_sp


class GCABlock(nn.Module):
    """Green-channel-attention block: FiLM-guided DualA."""

    def __init__(self, n_channels: int = 64):
        super().__init__()
        self.square_conv = nn.Conv2d(n_channels, n_channels, 3, padding=1, bias=False)
        self.relu = nn.LeakyReLU(0.1, inplace=True)
        self.extract_conv = nn.Conv2d(n_channels, n_channels, 3, padding=1, bias=True)
        self.res_block1 = SimpleBlock(2, n_channels, n_channels, n_channels)
        self.res_block2 = SimpleBlock(2, n_channels, n_channels, n_channels)
        self.down = nn.Conv2d(n_channels, n_channels // 2, 1, bias=True)
        self.up = nn.Conv2d(n_channels // 2, n_channels, 1, bias=True)
        self.spatial_att = nn.Conv2d(n_channels, 1, 7, padding=3, bias=True)
        _init_conv(self)

    def forward(
        self,
        x: torch.Tensor,
        guided_lam: torch.Tensor,
        guided_beta: torch.Tensor,
    ) -> torch.Tensor:
        x_temp = self.square_conv(x).mul(guided_lam) + guided_beta
        x_temp = self.relu(x_temp)
        x_temp = self.extract_conv(x_temp)
        x_temp = x + x_temp
        x_temp = x_temp + self.res_block1(x_temp)
        x_temp = x_temp + self.res_block2(x_temp)

        x_se = F.adaptive_avg_pool2d(x_temp, 1)
        x_se = torch.sigmoid(self.up(self.relu(self.down(x_se))))
        x_sp = torch.sigmoid(self.spatial_att(x_temp))
        return x_temp + x_temp * x_se + x_temp * x_sp


class GCPStillNet(nn.Module):
    """
    Still GCP-Net: GCP branch + IntraF (GCA) + U-Net merge.

    Matches paper building blocks for a single packed Bayer frame.
    """

    def __init__(self, nf: int = 64, output_channels: int = 3):
        super().__init__()
        self.nf = nf

        # --- Green Channel Prior branch ---
        self.feature_guided1 = SimpleBlock(3, nf, 4, nf)  # G1,G2 + noise
        self.feature_guided1_lam = nn.Conv2d(nf, nf, 3, padding=1, bias=True)
        self.feature_guided1_beta = nn.Conv2d(nf, nf, 3, padding=1, bias=True)

        self.feature_guided2 = SimpleBlock(3, nf, nf, nf)
        self.feature_guided2_lam = nn.Conv2d(nf, nf, 3, padding=1, bias=True)
        self.feature_guided2_beta = nn.Conv2d(nf, nf, 3, padding=1, bias=True)

        self.feature_guided3 = SimpleBlock(3, nf, nf, nf)
        self.feature_guided3_lam = nn.Conv2d(nf, nf, 3, padding=1, bias=True)
        self.feature_guided3_beta = nn.Conv2d(nf, nf, 3, padding=1, bias=True)

        self.feature_guided4 = SimpleBlock(3, nf, nf, nf)
        self.feature_guided4_lam = nn.Conv2d(nf, nf, 3, padding=1, bias=True)
        self.feature_guided4_beta = nn.Conv2d(nf, nf, 3, padding=1, bias=True)

        self.feature_guided5 = SimpleBlock(3, nf, nf, nf)
        self.feature_guided6 = SimpleBlock(3, nf, nf, nf)
        self.feature_guided6_up = nn.ConvTranspose2d(nf, nf, 2, stride=2, bias=True)
        self.feature_guided6_lam = nn.Conv2d(nf, nf, 3, padding=1, bias=True)
        self.feature_guided6_beta = nn.Conv2d(nf, nf, 3, padding=1, bias=True)

        # --- IntraF ---
        self.feature_extract = SimpleBlock(5, nf, 8, nf)  # RGGB + noise map
        self.gca1 = GCABlock(nf)
        self.gca2 = GCABlock(nf)
        self.gca3 = GCABlock(nf)
        self.gca4 = GCABlock(nf)

        self.merge = nn.Conv2d(nf, nf, 3, padding=1, bias=True)
        self.feature_up = nn.ConvTranspose2d(nf, nf, 2, stride=2, bias=True)

        # --- U-Net reconstruct ---
        self.conv_block_s1 = SimpleBlock(2, nf, nf, nf)
        self.acse_block_s1 = DualABlock(nf)
        self.pool1 = nn.Conv2d(nf, 2 * nf, 3, stride=2, padding=1, bias=True)

        self.conv_block_s2 = SimpleBlock(2, 2 * nf, 2 * nf, 2 * nf)
        self.acse_block_s2 = DualABlock(2 * nf, 2 * nf)
        self.pool2 = nn.Conv2d(2 * nf, 4 * nf, 3, stride=2, padding=1, bias=True)

        self.conv_block_s3 = SimpleBlock(2, 4 * nf, 4 * nf, 4 * nf)
        self.acse_block_s3 = DualABlock(4 * nf, 4 * nf)
        self.conv_block_s3_2 = SimpleBlock(2, 4 * nf, 4 * nf, 4 * nf)

        self.up1 = nn.ConvTranspose2d(4 * nf, 2 * nf, 2, stride=2, bias=True)
        self.conv_block_s4 = SimpleBlock(2, 2 * nf, 4 * nf, 2 * nf)
        self.acse_block_s4 = DualABlock(2 * nf, 2 * nf)

        self.up2 = nn.ConvTranspose2d(2 * nf, nf, 2, stride=2, bias=True)
        self.conv_block_s5 = SimpleBlock(3, nf, 2 * nf, output_channels)

        _init_conv(self)

    def forward(
        self,
        packed: torch.Tensor,
        noise_map: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        packed: (B, 4, H/2, W/2) RGGB
        noise_map: optional (B, 4, H/2, W/2); defaults to zeros
        """
        if packed.dim() != 4 or packed.shape[1] != 4:
            raise ValueError(f"expected packed (B,4,H,W), got {tuple(packed.shape)}")
        if noise_map is None:
            noise_map = torch.zeros_like(packed)
        elif noise_map.shape != packed.shape:
            raise ValueError(
                f"noise_map shape {tuple(noise_map.shape)} != packed {tuple(packed.shape)}"
            )

        # Green planes: G1 (ch1), G2 (ch2) in RGGB packing
        x_gr = packed[:, 1:3]
        x_gr_map = noise_map[:, 1:3]
        temp_g = torch.cat([x_gr, x_gr_map], dim=1)

        x_gr1 = self.feature_guided1(temp_g)
        lam1, beta1 = self.feature_guided1_lam(x_gr1), self.feature_guided1_beta(x_gr1)

        x_gr2 = self.feature_guided2(x_gr1)
        lam2, beta2 = self.feature_guided2_lam(x_gr2), self.feature_guided2_beta(x_gr2)

        x_gr3 = self.feature_guided3(x_gr2)
        lam3, beta3 = self.feature_guided3_lam(x_gr3), self.feature_guided3_beta(x_gr3)

        x_gr4 = self.feature_guided4(x_gr3)
        lam4, beta4 = self.feature_guided4_lam(x_gr4), self.feature_guided4_beta(x_gr4)

        x_gr5 = self.feature_guided5(x_gr4)
        x_gr6 = self.feature_guided6_up(self.feature_guided6(x_gr5))
        lam6, beta6 = self.feature_guided6_lam(x_gr6), self.feature_guided6_beta(x_gr6)

        # IntraF
        x = self.feature_extract(torch.cat([packed, noise_map], dim=1))
        x = self.gca1(x, lam1, beta1)
        x = self.gca2(x, lam2, beta2)
        x = self.gca3(x, lam3, beta3)
        x = self.gca4(x, lam4, beta4)

        # Still: N=1 merge (no InterF alignment)
        x = self.merge(x)
        x = self.feature_up(x)
        x = x.mul(lam6) + beta6

        x = self.acse_block_s1(self.conv_block_s1(x))
        l1 = x

        x = self.acse_block_s2(self.conv_block_s2(self.pool1(x)))
        l2 = x

        x = self.conv_block_s3_2(self.acse_block_s3(self.conv_block_s3(self.pool2(x))))

        x = self.up1(x)
        if x.shape[-2:] != l2.shape[-2:]:
            x = F.interpolate(x, size=l2.shape[-2:], mode="bilinear", align_corners=False)
        x = self.acse_block_s4(self.conv_block_s4(torch.cat([x, l2], dim=1)))

        x = self.up2(x)
        if x.shape[-2:] != l1.shape[-2:]:
            x = F.interpolate(x, size=l1.shape[-2:], mode="bilinear", align_corners=False)
        x = self.conv_block_s5(torch.cat([x, l1], dim=1))
        return x.clamp(0.0, 1.0)


def make_noise_map(packed: torch.Tensor, noise_std: float) -> torch.Tensor:
    """Constant per-channel noise estimate used as GCP-Net noise map input."""
    if noise_std <= 0:
        return torch.zeros_like(packed)
    return torch.full_like(packed, float(noise_std))
