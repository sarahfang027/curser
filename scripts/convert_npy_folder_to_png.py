#!/usr/bin/env python3
"""Write grayscale mosaic PNGs for every .npy in a folder (via bayer_denoiser.preview).

  python scripts/convert_npy_folder_to_png.py "runs/static-seq-plumbing/crop compare"
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder", type=Path, help="directory containing .npy mosaics")
    p.add_argument("--pattern", default="*.npy", help="glob under folder (default: all .npy)")
    p.add_argument("--false-color", action="store_true", help="pass --mosaic-false-color to preview")
    args = p.parse_args()

    folder = args.folder
    if not folder.is_dir():
        raise SystemExit(f"not a directory: {folder}")

    files = sorted(folder.glob(args.pattern))
    if not files:
        raise SystemExit(f"no files matching {args.pattern} in {folder}")

    for npy in files:
        png = npy.with_suffix(".png")
        cmd = [
            sys.executable,
            "-m",
            "bayer_denoiser.preview",
            str(npy),
            "--write-mosaic",
            str(png),
        ]
        if args.false_color:
            cmd.append("--mosaic-false-color")
        print(" ".join(cmd))
        subprocess.run(cmd, check=True)
        print(f"wrote {png}")


if __name__ == "__main__":
    main()
