"""Discover paired Bayer + RGB training files."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from bayer_denoiser.archived.io.raw_io import BAYER_EXTENSIONS, RGB_EXTENSIONS


@dataclass(frozen=True)
class BayerRgbPair:
    bayer: Path
    rgb: Path


def find_rgb_for_stem(rgb_dir: Path, stem: str) -> Path | None:
    for ext in RGB_EXTENSIONS:
        candidate = rgb_dir / f"{stem}{ext}"
        if candidate.is_file():
            return candidate
        candidate = rgb_dir / f"{stem}{ext.upper()}"
        if candidate.is_file():
            return candidate
    return None


def discover_pairs(bayer_dir: Path, rgb_dir: Path) -> list[BayerRgbPair]:
    bayer_dir = Path(bayer_dir)
    rgb_dir = Path(rgb_dir)
    if not bayer_dir.is_dir():
        raise FileNotFoundError(bayer_dir)
    if not rgb_dir.is_dir():
        raise FileNotFoundError(rgb_dir)

    pairs: list[BayerRgbPair] = []
    for bayer_path in sorted(bayer_dir.rglob("*")):
        if not bayer_path.is_file():
            continue
        if bayer_path.suffix.lower() not in {e.lower() for e in BAYER_EXTENSIONS}:
            continue
        rgb_path = find_rgb_for_stem(rgb_dir, bayer_path.stem)
        if rgb_path is None:
            raise FileNotFoundError(
                f"No RGB match for {bayer_path.name} under {rgb_dir} "
                f"(expected {bayer_path.stem}{{png,jpg,...}})"
            )
        pairs.append(BayerRgbPair(bayer=bayer_path, rgb=rgb_path))

    if not pairs:
        raise FileNotFoundError(f"No Bayer files found under {bayer_dir}")
    return pairs


def load_pairs_csv(manifest: Path) -> list[BayerRgbPair]:
    manifest = Path(manifest)
    pairs: list[BayerRgbPair] = []
    with manifest.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{manifest}: missing header row")
        for row in reader:
            bayer = Path(row["bayer"])
            rgb = Path(row["rgb"])
            if not bayer.is_file():
                raise FileNotFoundError(bayer)
            if not rgb.is_file():
                raise FileNotFoundError(rgb)
            pairs.append(BayerRgbPair(bayer=bayer, rgb=rgb))
    if not pairs:
        raise ValueError(f"{manifest}: no rows")
    return pairs
