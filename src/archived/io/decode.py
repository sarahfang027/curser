"""
Decode headerless .raw (+ .raw.json) → tensors.

Single path for preview, baseline infer, and training data loaders:

  .raw (+ JSON) → float32 H×W [0, 1] → pack → (4, H/2, W/2)

Normalization and contract checks live in raw_io.load_bayer_array.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from bayer_denoiser.sort_bayer import bayer_mosaic_to_packed
from bayer_denoiser.archived.io.raw_io import BayerLoadOptions, load_bayer_array


@dataclass(frozen=True)
class BayerTile:
    """Even square crop on the mosaic (full-res coordinates)."""

    top: int
    left: int
    size: int

    def __post_init__(self) -> None:
        if self.size <= 0 or self.size % 2:
            raise ValueError("tile.size must be a positive even integer")
        if self.top < 0 or self.left < 0:
            raise ValueError("tile top/left must be non-negative")


@dataclass(frozen=True)
class LoadedBayer:
    path: Path
    mosaic: torch.Tensor  # float32 (H, W), linear [0, 1] after contract normalization
    packed: torch.Tensor | None  # float32 (4, H/2, W/2) when pack=True


def pack_mosaic(mosaic_hw: torch.Tensor) -> torch.Tensor:
    """(H, W) mosaic → (4, H/2, W/2) RGGB planes."""
    if mosaic_hw.dim() != 2:
        raise ValueError(f"expected (H, W) mosaic, got shape {tuple(mosaic_hw.shape)}")
    h, w = mosaic_hw.shape
    if h % 2 or w % 2:
        raise ValueError(f"mosaic H/W must be even, got {h}x{w}")
    return bayer_mosaic_to_packed(mosaic_hw.unsqueeze(0).unsqueeze(0)).squeeze(0)


def load_bayer(
    path: str | Path,
    *,
    options: BayerLoadOptions | None = None,
    pack: bool = True,
    tile: BayerTile | None = None,
) -> LoadedBayer:
    """
    Load one Bayer frame.

    - Applies RAW ingest validation for ``.raw`` (see ``archived.io.raw_spec``).
    - Returns ``mosaic`` always; ``packed`` when ``pack=True``.
    - Optional ``tile`` crops the mosaic before packing.
    """
    path = Path(path)
    mosaic_np = load_bayer_array(path, options)
    mosaic = torch.from_numpy(mosaic_np)

    if tile is not None:
        h, w = mosaic.shape
        if tile.top + tile.size > h or tile.left + tile.size > w:
            raise ValueError(
                f"tile {tile.size} at ({tile.top},{tile.left}) exceeds mosaic {h}x{w}"
            )
        mosaic = mosaic[tile.top : tile.top + tile.size, tile.left : tile.left + tile.size]

    packed = pack_mosaic(mosaic) if pack else None
    return LoadedBayer(path=path, mosaic=mosaic, packed=packed)
