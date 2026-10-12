#!/usr/bin/env python3
"""Plot avg training loss and PSNR from history.json (train_static_noisy_sequence).

Loss subplot defaults to **Δ L1 from epoch 1** so slow plateaus stay visible; use
``--loss-absolute`` for raw L1. Y limits use tight padding (2% of span) and tick
precision follows the plotted range.
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
    frac: float = 0.02,
    min_pad: float = 0.0,
) -> tuple[float, float]:
    """Min/max of ``values`` expanded by ``frac`` of the span (at least ``min_pad``)."""
    lo = float(np.min(values))
    hi = float(np.max(values))
    span = hi - lo
    negligible = span <= max(min_pad, abs(hi) * 1e-15, 1e-18)
    if negligible:
        mean = float(np.mean(values))
        ref = max(abs(mean), abs(lo), abs(hi), min_pad, 1e-12)
        pad = max(min_pad, ref * frac if ref > 0 else min_pad or 1e-8)
        return mean - pad, mean + pad
    pad = max(min_pad, span * frac)
    return lo - pad, hi + pad


def _decimal_places_for_span(span: float) -> int:
    """Choose y-axis tick decimals from the visible value range."""
    if not np.isfinite(span) or span <= 0:
        return 6
    if span < 1e-6:
        return 8
    if span < 1e-4:
        return 7
    if span < 1e-3:
        return 6
    if span < 1e-2:
        return 5
    if span < 1e-1:
        return 4
    return 3


def _loss_series(losses: np.ndarray, *, loss_absolute: bool) -> tuple[np.ndarray, str]:
    if loss_absolute:
        return losses, "avg loss (L1)"
    baseline = float(losses[0])
    return losses - baseline, "L1 loss Δ (from epoch 1)"


def plot_training_curves(
    epochs: np.ndarray,
    losses: np.ndarray,
    psnrs: np.ndarray,
    *,
    title: str | None = None,
    loss_absolute: bool = False,
) -> plt.Figure:
    fig, (ax_loss, ax_psnr) = plt.subplots(2, 1, sharex=True, figsize=(8, 5), layout="tight")
    if title:
        fig.suptitle(title)

    loss_y, loss_ylabel = _loss_series(losses, loss_absolute=loss_absolute)
    ax_loss.plot(epochs, loss_y, color="#c44e52", linewidth=1.2)
    ax_loss.set_ylabel(loss_ylabel)

    loss_lo = float(np.min(loss_y))
    loss_hi = float(np.max(loss_y))
    loss_span = loss_hi - loss_lo
    eps = max(1e-8, 0.02 * loss_span) if loss_span > 0 else max(1e-8, abs(float(loss_y[0])) * 1e-6)
    if loss_absolute and loss_span > 0:
        ax_loss.set_ylim(loss_lo - eps, loss_hi + eps)
    else:
        ax_loss.set_ylim(_padded_ylim(loss_y, frac=0.02, min_pad=eps if loss_span <= 0 else 0.0))

    y0, y1 = ax_loss.get_ylim()
    loss_decimals = _decimal_places_for_span(y1 - y0)
    ax_loss.yaxis.set_major_formatter(FormatStrFormatter(f"%.{loss_decimals}f"))
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
        description="Plot avg loss and avg PSNR vs epoch from train_static_noisy_sequence history.json",
        epilog=(
            "Loss panel: by default plots Δ L1 from epoch 1 (un-flattens slow plateaus). "
            "Pass --loss-absolute to plot raw L1 values with tight y-limits."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
    parser.add_argument(
        "--loss-absolute",
        action="store_true",
        help="Plot raw L1 loss instead of default Δ from epoch 1",
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
    fig = plot_training_curves(
        epochs,
        losses,
        psnrs,
        title=args.title,
        loss_absolute=args.loss_absolute,
    )
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path} ({len(epochs)} epochs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
