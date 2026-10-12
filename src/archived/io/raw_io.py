"""Load single-channel Bayer mosaics from disk and normalize to float [0, 1]."""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from bayer_denoiser.sort_bayer import load_bayer_from_array
from bayer_denoiser.archived.io.raw_spec import (
    RawIngestError,
    expected_raw_bytes,
    parse_raw_dtype,
    validate_raw_file,
)

try:
    import rawpy
except ImportError:
    rawpy = None  # type: ignore[assignment,misc]

BAYER_EXTENSIONS = (".npy", ".raw", ".bin", ".dng", ".DNG", ".png", ".tif", ".tiff")
RGB_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".exr")


@dataclass
class BayerLoadOptions:
    """Normalization and decoding options for raw Bayer files."""

    black_level: float = 0.0
    white_level: float | None = None
    raw_width: int | None = None
    raw_height: int | None = None
    raw_dtype: str = "uint16"
    endian: str = "little"


@dataclass
class BayerFileMeta:
    width: int | None = None
    height: int | None = None
    dtype: str = "uint16"
    black_level: float = 0.0
    white_level: float | None = None
    endian: str = "little"


def _sidecar_paths(path: Path) -> list[Path]:
    return [
        path.with_suffix(f"{path.suffix}.json"),
        path.with_name(f"{path.stem}.meta.json"),
        path.with_name(f"{path.stem}.json"),
    ]


def load_bayer_sidecar(path: Path) -> BayerFileMeta | None:
    for sidecar in _sidecar_paths(path):
        if not sidecar.is_file():
            continue
        data = json.loads(sidecar.read_text())
        known = {f.name for f in fields(BayerFileMeta)}
        filtered = {k: v for k, v in data.items() if k in known}
        return BayerFileMeta(**filtered)
    return None


def sidecar_meta_to_bayer(meta_sidecar) -> BayerFileMeta:
    from bayer_denoiser.archived.io.raw_spec import SidecarMeta

    if isinstance(meta_sidecar, SidecarMeta):
        return BayerFileMeta(
            width=meta_sidecar.width,
            height=meta_sidecar.height,
            dtype=meta_sidecar.dtype,
            black_level=meta_sidecar.black_level,
            white_level=meta_sidecar.white_level,
            endian=meta_sidecar.endian,
        )
    return meta_sidecar


def normalize_bayer_array(
    arr: np.ndarray,
    *,
    black_level: float,
    white_level: float | None,
) -> np.ndarray:
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D Bayer mosaic, got shape {arr.shape}")
    x = arr.astype(np.float32)

    if np.issubdtype(arr.dtype, np.floating):
        if white_level is not None:
            scale = max(float(white_level) - float(black_level), 1e-6)
            return np.clip((x - black_level) / scale, 0.0, 1.0)
        if black_level == 0.0 and x.min() >= 0.0 and x.max() <= 1.0:
            return np.clip(x, 0.0, 1.0)
        peak = max(float(x.max()) - float(black_level), 1e-6)
        return np.clip((x - black_level) / peak, 0.0, 1.0)

    if x.max() <= 1.0 and white_level is None and black_level == 0.0:
        return np.clip(x, 0.0, 1.0)
    if white_level is None:
        if np.issubdtype(arr.dtype, np.integer):
            white_level = float(np.iinfo(arr.dtype).max)
        elif x.max() > 1.0:
            white_level = 65535.0 if x.max() > 255 else 255.0
        else:
            white_level = 1.0
    scale = max(float(white_level) - float(black_level), 1e-6)
    return np.clip((x - black_level) / scale, 0.0, 1.0)


def _dtype_max(dtype: str) -> float:
    dt = parse_raw_dtype(dtype)
    if np.issubdtype(dt, np.integer):
        return float(np.iinfo(dt).max)
    return 1.0


def load_flat_raw(path: Path, meta: BayerFileMeta) -> np.ndarray:
    if meta.width is None or meta.height is None:
        raise ValueError(f"{path}: flat .raw requires width and height (sidecar or CLI)")
    dt = parse_raw_dtype(meta.dtype)
    if meta.endian == "big":
        dt = dt.newbyteorder(">")
    count = meta.width * meta.height
    expected = count * dt.itemsize
    actual = path.stat().st_size
    if actual != expected:
        raise ValueError(
            f"{path}: file size {actual} bytes, expected {expected} "
            f"({meta.width}x{meta.height} {meta.dtype}, {dt.itemsize} bytes/sample)"
        )
    data = np.fromfile(path, dtype=dt, count=count)
    if data.size != count:
        raise ValueError(f"{path}: expected {count} samples, got {data.size}")
    return data.reshape(meta.height, meta.width)


def load_flat_raw_crop(
    path: Path,
    meta: BayerFileMeta,
    *,
    top: int,
    left: int,
    height: int,
    width: int,
) -> np.ndarray:
    """
    Read a rectangular crop from a headerless row-major .raw without loading the full plate.

    Coordinates are full-frame mosaic pixels (H×W). Prefer even top/left for RGGB.
    """
    if meta.width is None or meta.height is None:
        raise ValueError(f"{path}: flat .raw requires width and height")
    fw, fh = meta.width, meta.height
    if top < 0 or left < 0 or height <= 0 or width <= 0:
        raise ValueError(f"invalid crop top={top} left={left} size={height}x{width}")
    if top + height > fh or left + width > fw:
        raise ValueError(
            f"crop [{top}:{top + height}, {left}:{left + width}] exceeds frame {fh}x{fw}"
        )
    if top % 2 or left % 2:
        raise ValueError(f"crop origin ({top},{left}) must be even for RGGB")
    if height % 2 or width % 2:
        raise ValueError(f"crop size {height}x{width} must be even for RGGB")

    dt = parse_raw_dtype(meta.dtype)
    if meta.endian == "big":
        dt = dt.newbyteorder(">")
    item = dt.itemsize
    row_bytes = fw * item
    expected = fw * fh * item
    actual = path.stat().st_size
    if actual != expected:
        raise ValueError(
            f"{path}: file size {actual} bytes != expected {expected} "
            f"({fw}x{fh} {meta.dtype})"
        )

    crop = np.empty((height, width), dtype=dt)
    with path.open("rb") as f:
        for i in range(height):
            f.seek((top + i) * row_bytes + left * item)
            row = np.fromfile(f, dtype=dt, count=width)
            if row.size != width:
                raise ValueError(f"{path}: short read at row {top + i}")
            crop[i] = row
    return crop


def load_rawpy_mosaic(path: Path) -> np.ndarray:
    if rawpy is None:
        raise ImportError(
            "Reading DNG/camera RAW requires rawpy: pip install rawpy"
        )
    with rawpy.imread(str(path)) as raw:
        mosaic = raw.raw_image_visible.copy()
    return mosaic


def load_bayer_array(path: Path, options: BayerLoadOptions | None = None) -> np.ndarray:
    """Return H×W float32 mosaic in [0, 1]."""
    options = options or BayerLoadOptions()
    path = Path(path)
    suffix = path.suffix.lower()

    sidecar = load_bayer_sidecar(path)
    black = sidecar.black_level if sidecar else options.black_level
    white = sidecar.white_level if sidecar and sidecar.white_level is not None else options.white_level

    if suffix == ".npy":
        arr = np.load(path)
        return normalize_bayer_array(arr, black_level=black, white_level=white)

    if suffix in (".raw", ".bin"):
        if suffix == ".raw":
            try:
                contract = validate_raw_file(path, strict_spec=True, check_stem=False)
            except RawIngestError as exc:
                raise ValueError(str(exc)) from exc
            meta = sidecar_meta_to_bayer(contract)
            sidecar = meta
        else:
            sidecar = load_bayer_sidecar(path)
            meta = sidecar or BayerFileMeta(
                width=options.raw_width,
                height=options.raw_height,
                dtype=options.raw_dtype,
                black_level=options.black_level,
                white_level=options.white_level,
                endian=options.endian,
            )
            if sidecar is None:
                meta.black_level = options.black_level
                meta.white_level = options.white_level
        arr = load_flat_raw(path, meta)
        bl = meta.black_level if sidecar else black
        wl = meta.white_level if sidecar and meta.white_level is not None else white
        if wl is None:
            wl = _dtype_max(meta.dtype)
        return normalize_bayer_array(arr, black_level=bl, white_level=wl)

    if suffix == ".dng":
        arr = load_rawpy_mosaic(path)
        if sidecar and sidecar.black_level:
            black = sidecar.black_level
        elif arr.dtype != np.float32:
            black = float(np.min(arr)) if options.black_level == 0.0 else options.black_level
        wl = white
        if wl is None and np.issubdtype(arr.dtype, np.integer):
            wl = float(np.iinfo(arr.dtype).max)
        return normalize_bayer_array(arr, black_level=black, white_level=wl)

    img = Image.open(path)
    if img.mode not in ("L", "I;16", "I"):
        img = img.convert("L")
    arr = np.array(img)
    return normalize_bayer_array(arr, black_level=black, white_level=white)


def load_bayer_tensor(path: Path, options: BayerLoadOptions | None = None) -> torch.Tensor:
    """Bayer mosaic as (1, 1, H, W) float tensor."""
    arr = load_bayer_array(path, options)
    return load_bayer_from_array(arr)


def merge_bayer_options(sidecar: BayerFileMeta | None, options: BayerLoadOptions) -> BayerLoadOptions:
    if sidecar is None:
        return options
    return BayerLoadOptions(
        black_level=sidecar.black_level,
        white_level=sidecar.white_level,
        raw_width=sidecar.width or options.raw_width,
        raw_height=sidecar.height or options.raw_height,
        raw_dtype=sidecar.dtype or options.raw_dtype,
        endian=sidecar.endian or options.endian,
    )
