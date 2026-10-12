#!/usr/bin/env python3
"""Plot avg training loss and PSNR from history.json (train_static_noisy_sequence).

Y-axis limits are padded tightly around each series (not matplotlib auto margins) so
small epoch-to-epoch changes remain visible; tick labels use extra precision on loss.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter


def _metric(row: dict, primary: str, alias: str) -> float:
    if primary in row:
        return float(row[primary])
    if alias in row:
        return float(row[alias])
    raise KeyError(f"missing '{primary}' or '{alias}' in history row: {list(row.keys())}")


def load_series(history_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = json.loads(history_path.read_text())
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{history_path}: expected non-empty JSON array of epoch records")
    epochs = []
    losses = []
    psnrs = []
    for row in raw:
        if not isinstance(row, dict):
            raise ValueError(f"{history_path}: each entry must be an object, got {type(row)}")
        epochs.append(int(row["epoch"]))
        losses.append(_metric(row, "loss", "avg_loss"))
        psnrs.append(_metric(row, "psnr", "avg_psnr"))
    return (
        np.asarray(epochs, dtype=np.float64),
        np.asarray(losses, dtype=np.float64),
        np.asarray(psnrs, dtype=np.float64),
    )


def _padded_ylim(
    values: np.ndarray,
    *,
    frac: float = 0.05,
    min_pad: float = 0.0,
) -> tuple[float, float]:
    """Min/max of ``values`` expanded by ``frac`` of the span (at least ``min_pad``)."""
    lo = float(np.min(values))
    hi = float(np.max(values))
    if lo == hi:
        pad = max(min_pad, abs(lo) * frac if lo != 0 else min_pad or 1.0)
        return lo - pad, hi + pad
    span = hi - lo
    pad = max(min_pad, span * frac)
    return lo - pad, hi + pad


def plot_training_curves(
    epochs: np.ndarray,
    losses: np.ndarray,
    psnrs: np.ndarray,
    *,
    title: str | None = None,
) -> plt.Figure:
    fig, (ax_loss, ax_psnr) = plt.subplots(2, 1, sharex=True, figsize=(8, 5), layout="tight")
    if title:
        fig.suptitle(title)

    ax_loss.plot(epochs, losses, color="#c44e52", linewidth=1.2)
    ax_loss.set_ylabel("avg loss (L1)")
    ax_loss.set_ylim(_padded_ylim(losses, frac=0.05, min_pad=max(1e-6, float(np.max(losses)) * 1e-4)))
    ax_loss.yaxis.set_major_formatter(FormatStrFormatter("%.5f"))
    ax_loss.grid(True, alpha=0.3)

    ax_psnr.plot(epochs, psnrs, color="#4c72b0", linewidth=1.2)
    ax_psnr.set_ylabel("avg PSNR (dB)")
    ax_psnr.set_ylim(_padded_ylim(psnrs, frac=0.05, min_pad=0.1))
    ax_psnr.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax_psnr.set_xlabel("epoch")
    ax_psnr.grid(True, alpha=0.3)

    return fig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plot avg loss and avg PSNR vs epoch from train_static_noisy_sequence history.json"
    )
    parser.add_argument(
        "--history",
        type=Path,
        default=Path("history.json"),
        help="Path to history.json (default: ./history.json)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="PNG output path (default: <history parent>/training_curves.png)",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional plot title",
    )
    args = parser.parse_args(argv)

    history_path = args.history.expanduser().resolve()
    if not history_path.is_file():
        print(f"history not found: {history_path}", file=sys.stderr)
        return 1

    out_path = args.output
    if out_path is None:
        out_path = history_path.parent / "training_curves.png"
    else:
        out_path = out_path.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    epochs, losses, psnrs = load_series(history_path)
    fig = plot_training_curves(epochs, losses, psnrs, title=args.title)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path} ({len(epochs)} epochs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
