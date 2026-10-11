"""Decode proprietary ``.sky`` (magic ``csky``) cinema RAW containers."""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from bayer_denoiser.archived.io.raw_io import normalize_bayer_array

CSKY_MAGIC = b"csky"
KNOWN_CHUNKS = frozenset({b"json", b"meta", b"data"})
# Sample clips: JSON height = 4 row banks × 4428 lines; banks are contiguous in the file
# but consecutive mosaic rows cycle through banks (not four stacked copies).
CSKY_ROW_BANKS = 4


@dataclass(frozen=True)
class SkyFrameInfo:
    width: int
    height: int
    bit_depth: int
    row_stride_bytes: int
    data_offset: int
    payload_bytes: int
    clip_json: dict


@dataclass(frozen=True)
class SkyDecodeOptions:
    """How to map stored uint16 samples to linear mosaic floats."""

    black_level: float = 0.0
    white_level: float | None = None
    value_shift: int | None = None
    cfa_pattern: str = "RGGB"
    row_deinterleave: bool = True


class SkyFormatError(ValueError):
    pass


def _scan_chunk(data: bytes, tag: bytes, start: int, *, search_limit: int = 1 << 20) -> tuple[int, int]:
    end = min(len(data), start + search_limit)
    pos = start
    while pos + 12 <= end:
        if data[pos : pos + 4] != tag:
            pos += 1
            continue
        l1, l2 = struct.unpack_from("<II", data, pos + 4)
        size = _chunk_payload_size(tag, l1, l2)
        if size <= 0:
            pos += 1
            continue
        chunk_end = pos + 12 + size
        if chunk_end > len(data):
            if tag == b"data" and pos + 12 < len(data):
                size = len(data) - (pos + 12)
                chunk_end = len(data)
            else:
                pos += 1
                continue
        if chunk_end > len(data):
            pos += 1
            continue
        return pos, size
    raise SkyFormatError(f"Missing {tag.decode()} chunk")


def _chunk_payload_size(tag: bytes, l1: int, l2: int) -> int:
    if tag == b"json":
        return l2
    return l1


def parse_sky_file(path: Path) -> SkyFrameInfo:
    data = Path(path).read_bytes()
    if len(data) < 16 or data[:4] != CSKY_MAGIC:
        raise SkyFormatError(f"Not a csky file: {path}")

    json_start_at = 256 if len(data) > 256 and data[256:260] == b"json" else 0
    json_off, payload_len = _scan_chunk(data, b"json", json_start_at)
    json_start = json_off + 12
    try:
        clip = json.loads(data[json_start : json_start + payload_len].decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise SkyFormatError(f"Invalid JSON chunk in {path}") from exc

    settings = clip.get("clip", {}).get("settings", {})
    width = int(settings["width"])
    height = int(settings["height"])
    bit_depth = int(settings.get("bit_depth", 16))

    meta_off, _meta_len = _scan_chunk(data, b"meta", json_off + 12 + payload_len)
    data_off, _payload_bytes = _scan_chunk(data, b"data", meta_off + 12, search_limit=1 << 20)
    mosaic_off = data_off + 12
    header_payload = struct.unpack_from("<I", data, data_off + 4)[0]
    if header_payload <= 0:
        raise SkyFormatError("Invalid data chunk size")
    if header_payload % height != 0:
        raise SkyFormatError(
            f"Header payload {header_payload} is not divisible by height {height}"
        )
    row_stride = header_payload // height
    min_stride = width * 2
    if row_stride < min_stride:
        raise SkyFormatError(
            f"Row stride {row_stride} too small for width {width} (need {min_stride})"
        )
    available = len(data) - mosaic_off
    if available < (height - 1) * row_stride + min_stride:
        raise SkyFormatError(
            f"File truncated: need at least "
            f"{(height - 1) * row_stride + min_stride} bytes, have {available}"
        )

    return SkyFrameInfo(
        width=width,
        height=height,
        bit_depth=bit_depth,
        row_stride_bytes=row_stride,
        data_offset=mosaic_off,
        payload_bytes=header_payload,
        clip_json=clip,
    )


def _default_value_shift(bit_depth: int) -> int:
    if bit_depth <= 0 or bit_depth > 16:
        return 0
    if bit_depth == 12:
        return 2
    return max(0, 16 - bit_depth)


def _default_white_level(bit_depth: int, shift: int) -> float:
    if bit_depth > 0 and bit_depth <= 16:
        return float((1 << bit_depth) - 1)
    return float((1 << (16 - shift)) - 1)


def deinterleave_row_banks(
    mosaic: np.ndarray, row_banks: int = CSKY_ROW_BANKS
) -> np.ndarray:
    """Reorder block-stored row banks into scanline order (17712 unique rows)."""
    h, w = mosaic.shape
    if row_banks <= 1 or h % row_banks != 0:
        return mosaic
    band_h = h // row_banks
    banks = mosaic.reshape(row_banks, band_h, w)
    out = np.empty_like(mosaic)
    for band in range(row_banks):
        out[band::row_banks] = banks[band]
    return out


def read_sky_mosaic_uint16(path: Path, info: SkyFrameInfo | None = None) -> np.ndarray:
    info = info or parse_sky_file(path)
    raw = Path(path).read_bytes()
    h, w = info.height, info.width
    stride = info.row_stride_bytes
    base = info.data_offset
    rows = np.ndarray(
        (h, w),
        dtype="<u2",
        buffer=raw,
        offset=base,
        strides=(stride, 2),
    )
    return np.ascontiguousarray(rows)


def decode_sky_mosaic(
    path: Path,
    *,
    info: SkyFrameInfo | None = None,
    options: SkyDecodeOptions | None = None,
) -> np.ndarray:
    options = options or SkyDecodeOptions()
    info = info or parse_sky_file(path)
    mosaic = read_sky_mosaic_uint16(path, info).astype(np.float32)
    mosaic = mosaic.astype(np.float16)
    if options.row_deinterleave:
        mosaic = deinterleave_row_banks(mosaic)
    shift = options.value_shift
    if shift is None:
        shift = _default_value_shift(info.bit_depth)
    if shift:
        mosaic = np.floor_divide(mosaic, 2**shift)
    white = options.white_level
    if white is None:
        white = _default_white_level(info.bit_depth, shift)
    return normalize_bayer_array(
        mosaic,
        black_level=options.black_level,
        white_level=white,
    )
