"""Training data: synthetic, simulated Bayer from RGB, or paired raw Bayer + RGB."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

from bayer_denoiser.sort_bayer import bayer_mosaic_to_packed, rgb_to_bayer
from bayer_denoiser.archived.io.decode import load_bayer, pack_mosaic
from bayer_denoiser.archived.qa.pairs import BayerRgbPair, discover_pairs, load_pairs_csv
from bayer_denoiser.archived.io.raw_io import (
    BayerFileMeta,
    BayerLoadOptions,
    load_flat_raw_crop,
    normalize_bayer_array,
)
from bayer_denoiser.archived.io.raw_spec import validate_raw_file


def _load_rgb_image(path: Path, image_size: int) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    img = TF.resize(img, [image_size, image_size], antialias=True)
    t = TF.to_tensor(img)
    return t


def _load_rgb_full(path: Path) -> torch.Tensor:
    suffix = path.suffix.lower()
    if suffix == ".exr":
        try:
            import imageio.v3 as iio
        except ImportError as exc:
            raise ImportError("Reading .exr RGB requires: pip install imageio") from exc
        import numpy as np

        arr = np.asarray(iio.imread(path), dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[..., None]
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        t = torch.from_numpy(arr[..., :3].copy()).permute(2, 0, 1)
        if t.max() > 1.0:
            t = t / t.max().clamp(min=1e-6)
        return t.clamp(0.0, 1.0)

    img = Image.open(path).convert("RGB")
    return TF.to_tensor(img)


def choose_crop(
    height: int,
    width: int,
    crop_size: int,
    *,
    random_crop: bool = True,
) -> tuple[int, int]:
    """
    Pick a square crop origin (top, left). Origins are forced even so RGGB phase is preserved.
    """
    if crop_size <= 0:
        return 0, 0
    if crop_size % 2:
        raise ValueError(f"crop_size must be even, got {crop_size}")
    if crop_size > height or crop_size > width:
        raise ValueError(f"crop_size {crop_size} exceeds frame {height}x{width}")
    max_top = height - crop_size
    max_left = width - crop_size
    if not random_crop:
        top = (max_top // 2) & ~1
        left = (max_left // 2) & ~1
        return top, left
    # Sample on the even lattice only
    top_choices = max_top // 2 + 1
    left_choices = max_left // 2 + 1
    top = int(torch.randint(0, top_choices, (1,)).item()) * 2
    left = int(torch.randint(0, left_choices, (1,)).item()) * 2
    return top, left


def _apply_spatial_crop(
    bayer_hw: torch.Tensor,
    rgb: torch.Tensor,
    crop_size: int,
    random_crop: bool,
    index: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    del index  # kept for call-site compatibility; crops are freshly random each call
    _, h, w = rgb.shape
    if crop_size <= 0 or (h == crop_size and w == crop_size):
        return bayer_hw, rgb
    top, left = choose_crop(h, w, crop_size, random_crop=random_crop)
    bayer = bayer_hw[top : top + crop_size, left : left + crop_size]
    rgb = rgb[:, top : top + crop_size, left : left + crop_size]
    return bayer, rgb


def _frame_size_for_pair(pair: BayerRgbPair, options: BayerLoadOptions) -> tuple[int, int]:
    """Return (H, W) without decoding the mosaic when a .raw sidecar is present."""
    path = pair.bayer
    if path.suffix.lower() == ".raw":
        meta = validate_raw_file(path, strict_spec=False, check_stem=False)
        return meta.height, meta.width
    loaded = load_bayer(path, options=options, pack=False)
    return int(loaded.mosaic.shape[0]), int(loaded.mosaic.shape[1])


def _load_bayer_crop(
    path: Path,
    options: BayerLoadOptions,
    *,
    top: int,
    left: int,
    size: int,
) -> torch.Tensor:
    """Load only a mosaic crop when path is contract .raw; otherwise full decode + slice."""
    if path.suffix.lower() == ".raw":
        contract = validate_raw_file(path, strict_spec=False, check_stem=False)
        meta = BayerFileMeta(
            width=contract.width,
            height=contract.height,
            dtype=contract.dtype,
            black_level=contract.black_level,
            white_level=contract.white_level,
            endian=contract.endian,
        )
        arr = load_flat_raw_crop(
            path, meta, top=top, left=left, height=size, width=size
        )
        arr = normalize_bayer_array(
            arr, black_level=meta.black_level, white_level=meta.white_level
        )
        return torch.from_numpy(arr)

    loaded = load_bayer(path, options=options, pack=False)
    return loaded.mosaic[top : top + size, left : left + size]


def _load_rgb_crop(path: Path, *, top: int, left: int, size: int) -> torch.Tensor:
    """Load RGB and return a square crop (full decode for EXR/PNG; window when possible)."""
    suffix = path.suffix.lower()
    if suffix in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
        img = Image.open(path).convert("RGB")
        box = (left, top, left + size, top + size)
        img = img.crop(box)
        return TF.to_tensor(img)

    rgb = _load_rgb_full(path)
    return rgb[:, top : top + size, left : left + size]


class SyntheticBayerDataset(Dataset):
    """Random smooth RGB patches → noisy Bayer mosaic + clean RGB target."""

    def __init__(
        self,
        length: int,
        image_size: int = 128,
        noise_std: float = 0.02,
        seed: int | None = None,
    ):
        if image_size % 2:
            raise ValueError("image_size must be even")
        self.length = length
        self.image_size = image_size
        self.noise_std = noise_std
        self._rng = torch.Generator()
        if seed is not None:
            self._rng.manual_seed(seed)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        g = torch.Generator()
        g.manual_seed(int(torch.randint(0, 2**31 - 1, (1,), generator=self._rng).item()) + index)
        rgb = torch.rand(3, self.image_size, self.image_size, generator=g)
        rgb = TF.gaussian_blur(rgb, kernel_size=[5, 5], sigma=[1.0, 2.5])
        bayer = rgb_to_bayer(rgb.unsqueeze(0)).squeeze(0)
        if self.noise_std > 0:
            bayer = (bayer + torch.randn_like(bayer) * self.noise_std).clamp(0.0, 1.0)
        packed = bayer_mosaic_to_packed(bayer.unsqueeze(0)).squeeze(0)
        return packed, rgb


class SyntheticBayerDenoiseDataset(Dataset):
    """Random smooth RGB → clean packed Bayer; input is clean + AWGN in mosaic space."""

    def __init__(
        self,
        length: int,
        image_size: int = 128,
        noise_std: float = 0.02,
        seed: int | None = None,
    ):
        if image_size % 2:
            raise ValueError("image_size must be even")
        self.length = length
        self.image_size = image_size
        self.noise_std = noise_std
        self._rng = torch.Generator()
        if seed is not None:
            self._rng.manual_seed(seed)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        g = torch.Generator()
        g.manual_seed(int(torch.randint(0, 2**31 - 1, (1,), generator=self._rng).item()) + index)
        rgb = torch.rand(3, self.image_size, self.image_size, generator=g)
        rgb = TF.gaussian_blur(rgb, kernel_size=[5, 5], sigma=[1.0, 2.5])
        bayer = rgb_to_bayer(rgb.unsqueeze(0)).squeeze(0)
        clean_packed = bayer_mosaic_to_packed(bayer.unsqueeze(0)).squeeze(0)
        noisy = bayer
        if self.noise_std > 0:
            noisy = (bayer + torch.randn_like(bayer) * self.noise_std).clamp(0.0, 1.0)
        noisy_packed = bayer_mosaic_to_packed(noisy.unsqueeze(0)).squeeze(0)
        return noisy_packed, clean_packed


class RgbFolderDataset(Dataset):
    """RGB images on disk; Bayer mosaic is created when each sample is loaded."""

    def __init__(
        self,
        root: str | Path,
        image_size: int = 256,
        noise_std: float = 0.02,
        extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".exr"),
    ):
        root = Path(root)
        if not root.is_dir():
            raise FileNotFoundError(root)
        self.paths = sorted(
            p for p in root.rglob("*") if p.suffix.lower() in extensions and p.is_file()
        )
        if not self.paths:
            raise FileNotFoundError(f"No images found under {root}")
        self.image_size = image_size
        self.noise_std = noise_std
        self.resize = transforms.Resize([image_size, image_size], antialias=True)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        rgb = _load_rgb_image(self.paths[index], self.image_size)
        bayer = rgb_to_bayer(rgb.unsqueeze(0)).squeeze(0)
        if self.noise_std > 0:
            bayer = (bayer + torch.randn_like(bayer) * self.noise_std).clamp(0.0, 1.0)
        packed = bayer_mosaic_to_packed(bayer.unsqueeze(0)).squeeze(0)
        return packed, rgb


class PairedRawBayerDataset(Dataset):
    """
    Real Bayer mosaic files paired with reference RGB (same frame / alignment).

    Keep full-resolution plates on disk. The loader auto-crops square patches:

    - ``crop_size`` / train ``--image-size``: patch edge length (even, RGGB-safe)
    - ``crops_per_frame``: virtual samples per plate per epoch (random locations)
    - ``.raw``: seeks and reads only the crop region (no full 18K decode)

    Bayer paths: .npy, .raw (+ sidecar JSON), .dng (needs rawpy), grayscale .tif/.png
    RGB paths: matched by filename stem in rgb_dir, or listed in a CSV manifest.
    """

    def __init__(
        self,
        *,
        bayer_dir: Path | None = None,
        rgb_dir: Path | None = None,
        pairs_csv: Path | None = None,
        bayer_options: BayerLoadOptions | None = None,
        crop_size: int = 0,
        random_crop: bool = True,
        crops_per_frame: int = 1,
        noise_std: float = 0.0,
        resize: int = 0,
        bayer_target: bool = False,
    ):
        if pairs_csv:
            self.pairs = load_pairs_csv(pairs_csv)
        elif bayer_dir and rgb_dir:
            self.pairs = discover_pairs(bayer_dir, rgb_dir)
        else:
            raise ValueError("Provide pairs_csv or both bayer_dir and rgb_dir")

        if crop_size and crop_size % 2:
            raise ValueError("crop_size must be even (RGGB)")
        if resize and resize % 2:
            raise ValueError("resize must be even")
        if crops_per_frame < 1:
            raise ValueError("crops_per_frame must be >= 1")

        self.bayer_options = bayer_options or BayerLoadOptions()
        self.crop_size = crop_size
        self.random_crop = random_crop
        self.crops_per_frame = crops_per_frame
        self.noise_std = noise_std
        self.resize = resize
        self.bayer_target = bayer_target

    def __len__(self) -> int:
        return len(self.pairs) * self.crops_per_frame

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        pair_idx = index // self.crops_per_frame
        pair: BayerRgbPair = self.pairs[pair_idx]

        # Prefer: choose crop from known geometry, then partial-load .raw + RGB crop.
        # Fallback: full decode when resize is requested or crop_size is disabled.
        use_windowed = self.crop_size > 0 and self.resize == 0
        if use_windowed:
            h, w = _frame_size_for_pair(pair, self.bayer_options)
            # Verify RGB size lazily via crop path (raises if mismatch when slicing)
            top, left = choose_crop(h, w, self.crop_size, random_crop=self.random_crop)
            bayer = _load_bayer_crop(
                pair.bayer,
                self.bayer_options,
                top=top,
                left=left,
                size=self.crop_size,
            )
            rgb = _load_rgb_crop(pair.rgb, top=top, left=left, size=self.crop_size)
            if rgb.shape[-2:] != (self.crop_size, self.crop_size):
                raise ValueError(
                    f"RGB crop size mismatch for {pair.rgb.name}: "
                    f"got {tuple(rgb.shape[-2:])}, expected {self.crop_size}"
                )
        else:
            loaded = load_bayer(pair.bayer, options=self.bayer_options, pack=False)
            bayer = loaded.mosaic
            rgb = _load_rgb_full(pair.rgb)

            bh, bw = bayer.shape
            _, rh, rw = rgb.shape
            if (bh, bw) != (rh, rw):
                raise ValueError(
                    f"Size mismatch for {pair.bayer.name}: bayer {bh}x{bw} vs rgb {rh}x{rw}"
                )

            if self.resize:
                bayer = TF.resize(
                    bayer.unsqueeze(0),
                    [self.resize, self.resize],
                    antialias=True,
                ).squeeze(0)
                rgb = TF.resize(rgb, [self.resize, self.resize], antialias=True)

            bayer, rgb = _apply_spatial_crop(
                bayer, rgb, self.crop_size, self.random_crop
            )

        clean_bayer = bayer
        if self.noise_std > 0:
            bayer = (bayer + torch.randn_like(bayer) * self.noise_std).clamp(0.0, 1.0)

        packed = pack_mosaic(bayer)
        if getattr(self, "bayer_target", False):
            return packed, pack_mosaic(clean_bayer)
        return packed, rgb
