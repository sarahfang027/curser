"""
RAW ingest contract — single source of truth for headerless Bayer .raw frames.

Human-readable summary: docs/formats/RAW_SPEC.md (module archived under ``archived/io/``).
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np

SPEC_VERSION = "1"
CFA_PATTERN_RGGB = "RGGB"
CANONICAL_SIDECAR_SUFFIX = ".raw.json"  # plate.raw → plate.raw.json

# Stable stem for pairing: shot_0100, plate_a001 (letter first, then alnum/underscore)
RAW_STEM_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


class RawIngestError(ValueError):
    """Raised when a frame violates the ingest contract."""


@dataclass(frozen=True)
class RawIngestSpec:
    """Project default geometry and storage (override per sidecar when not strict)."""

    width: int = 17592
    height: int = 18024
    dtype: str = "float16"
    endian: str = "little"
    black_level: float = 0.0
    white_level: float = 1.0
    cfa_pattern: str = CFA_PATTERN_RGGB
    row_major: bool = True


PROJECT_RAW_SPEC = RawIngestSpec()


@dataclass
class SidecarMeta:
    width: int
    height: int
    dtype: str
    black_level: float = 0.0
    white_level: float | None = 1.0
    endian: str = "little"
    cfa_pattern: str = CFA_PATTERN_RGGB
    row_major: bool = True
    spec_version: str = SPEC_VERSION

    @classmethod
    def from_dict(cls, data: dict) -> SidecarMeta:
        known = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in known}
        required = ("width", "height", "dtype")
        for key in required:
            if key not in filtered:
                raise RawIngestError(f"sidecar missing required field {key!r}")
        return cls(**filtered)

    def to_dict(self) -> dict:
        return asdict(self)


def parse_raw_dtype(name: str) -> np.dtype:
    aliases = {
        "half": "float16",
        "f16": "float16",
        "fp16": "float16",
        "float16": "float16",
        "single": "float32",
        "f32": "float32",
        "float32": "float32",
    }
    key = name.strip().lower()
    return np.dtype(aliases.get(key, key))


def expected_raw_bytes(width: int, height: int, dtype: str) -> int:
    dt = parse_raw_dtype(dtype)
    return width * height * dt.itemsize


def canonical_sidecar_path(raw_path: Path) -> Path:
    raw_path = Path(raw_path)
    return raw_path.with_suffix(f"{raw_path.suffix}.json")


def validate_stem(raw_path: Path) -> None:
    stem = raw_path.stem
    if not RAW_STEM_PATTERN.match(stem):
        raise RawIngestError(
            f"{raw_path.name}: stem {stem!r} must match {RAW_STEM_PATTERN.pattern} "
            "(stable ID for future EXR pairing, e.g. shot_0100)"
        )


def validate_geometry(width: int, height: int) -> None:
    if width <= 0 or height <= 0:
        raise RawIngestError(f"invalid geometry {width}x{height}")
    if width % 2 or height % 2:
        raise RawIngestError(
            f"geometry {width}x{height} (W×H) must be even for {CFA_PATTERN_RGGB}"
        )


def load_canonical_sidecar(raw_path: Path) -> SidecarMeta:
    sidecar = canonical_sidecar_path(raw_path)
    if not sidecar.is_file():
        raise RawIngestError(
            f"{raw_path.name}: missing canonical sidecar {sidecar.name} "
            f"(expected next to .raw as *{CANONICAL_SIDECAR_SUFFIX})"
        )
    try:
        data = json.loads(sidecar.read_text())
    except json.JSONDecodeError as exc:
        raise RawIngestError(f"{sidecar}: invalid JSON: {exc}") from exc
    return SidecarMeta.from_dict(data)


def sidecar_matches_spec(meta: SidecarMeta, spec: RawIngestSpec) -> list[str]:
    errors: list[str] = []
    if meta.width != spec.width or meta.height != spec.height:
        errors.append(
            f"geometry {meta.width}x{meta.height} != contract {spec.width}x{spec.height}"
        )
    if meta.dtype.lower() not in {spec.dtype.lower(), "half", "f16", "fp16"}:
        if parse_raw_dtype(meta.dtype) != parse_raw_dtype(spec.dtype):
            errors.append(f"dtype {meta.dtype!r} != contract {spec.dtype!r}")
    if meta.endian != spec.endian:
        errors.append(f"endian {meta.endian!r} != contract {spec.endian!r}")
    if meta.cfa_pattern != spec.cfa_pattern:
        errors.append(f"cfa_pattern {meta.cfa_pattern!r} != contract {spec.cfa_pattern!r}")
    if not meta.row_major:
        errors.append("row_major must be true")
    if meta.black_level != spec.black_level:
        errors.append(f"black_level {meta.black_level} != contract {spec.black_level}")
    if meta.white_level != spec.white_level:
        errors.append(f"white_level {meta.white_level} != contract {spec.white_level}")
    return errors


def validate_raw_file(
    raw_path: Path,
    *,
    spec: RawIngestSpec = PROJECT_RAW_SPEC,
    strict_spec: bool = True,
    check_stem: bool = True,
) -> SidecarMeta:
    """
    Fail fast on contract violations. Returns parsed sidecar meta on success.
    """
    raw_path = Path(raw_path)
    if raw_path.suffix.lower() != ".raw":
        raise RawIngestError(f"{raw_path}: expected .raw file")

    if not raw_path.is_file():
        raise RawIngestError(f"{raw_path}: file not found")

    if check_stem:
        validate_stem(raw_path)

    meta = load_canonical_sidecar(raw_path)
    validate_geometry(meta.width, meta.height)

    if strict_spec:
        for msg in sidecar_matches_spec(meta, spec):
            raise RawIngestError(f"{raw_path.name}: {msg}")

    expected = expected_raw_bytes(meta.width, meta.height, meta.dtype)
    actual = raw_path.stat().st_size
    if actual != expected:
        raise RawIngestError(
            f"{raw_path.name}: file size {actual:,} bytes != expected {expected:,} "
            f"({meta.width}×{meta.height} {meta.dtype}, "
            f"{parse_raw_dtype(meta.dtype).itemsize} bytes/sample, row-major)"
        )

    return meta


def write_canonical_sidecar(
    raw_path: Path,
    spec: RawIngestSpec = PROJECT_RAW_SPEC,
) -> Path:
    """Write plate.raw.json from the project contract defaults."""
    raw_path = Path(raw_path)
    out = canonical_sidecar_path(raw_path)
    payload = SidecarMeta(
        width=spec.width,
        height=spec.height,
        dtype=spec.dtype,
        black_level=spec.black_level,
        white_level=spec.white_level,
        endian=spec.endian,
        cfa_pattern=spec.cfa_pattern,
        row_major=spec.row_major,
        spec_version=SPEC_VERSION,
    )
    out.write_text(json.dumps(payload.to_dict(), indent=2) + "\n")
    return out
