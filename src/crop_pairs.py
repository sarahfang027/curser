"""Aligned Bayer crop windows and optional on-disk crop pair export (float16)."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable

import numpy as np

MANIFEST_VERSION = 1


def snap_even(v: int) -> int:
    return v - (v % 2)


def center_origin(h: int, w: int, tile: int) -> tuple[int, int]:
    top = (h - tile) // 2
    left = (w - tile) // 2
    return snap_even(top), snap_even(left)


def grid_origins(h: int, w: int, tile: int, rows: int, cols: int) -> list[tuple[int, int]]:
    if rows < 1 or cols < 1:
        raise ValueError("grid rows and cols must be >= 1")
    if tile % 2 != 0:
        raise ValueError("tile must be even (Bayer)")
    if tile > h or tile > w:
        raise ValueError(f"tile {tile} exceeds mosaic {h}x{w}")
    ymax = h - tile
    xmax = w - tile
    ys = [0] if rows == 1 else [int(round(ymax * r / (rows - 1))) for r in range(rows)]
    xs = [0] if cols == 1 else [int(round(xmax * c / (cols - 1))) for c in range(cols)]
    out: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for y in ys:
        for x in xs:
            origin = (snap_even(y), snap_even(x))
            if origin not in seen:
                seen.add(origin)
                out.append(origin)
    return out


def random_origins(
    h: int,
    w: int,
    tile: int,
    count: int,
    seed: int,
) -> list[tuple[int, int]]:
    if count < 1:
        raise ValueError("count must be >= 1")
    rng = random.Random(seed)
    seen: set[tuple[int, int]] = set()
    out: list[tuple[int, int]] = []
    max_tries = count * 50
    tries = 0
    while len(out) < count and tries < max_tries:
        tries += 1
        y = snap_even(rng.randrange(0, h - tile + 1, 2))
        x = snap_even(rng.randrange(0, w - tile + 1, 2))
        if (y, x) in seen:
            continue
        seen.add((y, x))
        out.append((y, x))
    if len(out) < count:
        raise ValueError(f"could only place {len(out)} unique crops (requested {count})")
    return out


def crop_mosaic(mosaic: np.ndarray, tile: int, y: int, x: int) -> np.ndarray:
    return mosaic[y : y + tile, x : x + tile]


def resolve_origins(
    h: int,
    w: int,
    tile: int,
    num_crops: int,
    grid: tuple[int, int] | None,
    seed: int,
) -> list[tuple[int, int]]:
    if grid is not None:
        rows, cols = grid
        origins = grid_origins(h, w, tile, rows, cols)
        if num_crops > 0 and len(origins) != num_crops:
            raise ValueError(
                f"CROP_GRID {rows}x{cols} yields {len(origins)} crops, but NUM_CROPS={num_crops}"
            )
        return origins
    if num_crops == 1:
        return [center_origin(h, w, tile)]
    return random_origins(h, w, tile, num_crops, seed)


def write_crop_pair(
    noisy_full: np.ndarray,
    clean_full: np.ndarray,
    out_dir: Path,
    index: int,
    tile: int,
    y: int,
    x: int,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    noisy_name = f"noisy_{index:03d}.npy"
    clean_name = f"gaussian_GT_{index:03d}.npy"
    noisy_crop = crop_mosaic(noisy_full, tile, y, x).astype(np.float16)
    clean_crop = crop_mosaic(clean_full, tile, y, x).astype(np.float16)
    np.save(out_dir / noisy_name, noisy_crop)
    np.save(out_dir / clean_name, clean_crop)
    return {
        "index": index,
        "y": y,
        "x": x,
        "tile_size": tile,
        "noisy": noisy_name,
        "clean": clean_name,
    }


def write_manifest(
    path: Path,
    *,
    tile: int,
    source_shape: tuple[int, int],
    mode: str,
    seed: int,
    crops: list[dict],
    grid: tuple[int, int] | None = None,
) -> None:
    payload = {
        "version": MANIFEST_VERSION,
        "tile_size": tile,
        "source_shape": list(source_shape),
        "mode": mode,
        "seed": seed,
        "crops": crops,
    }
    if grid is not None:
        payload["grid"] = list(grid)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def load_manifest(path: Path) -> dict:
    data = json.loads(path.read_text())
    if data.get("version") != MANIFEST_VERSION:
        raise ValueError(f"unsupported crops manifest version: {data.get('version')}")
    return data


def manifest_origins(manifest: dict) -> list[tuple[int, int, int]]:
    tile = int(manifest["tile_size"])
    out: list[tuple[int, int, int]] = []
    for entry in manifest["crops"]:
        out.append((int(entry["y"]), int(entry["x"]), tile))
    return out
