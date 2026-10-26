"""A copy of bias.py's comparison plot with a prior histogram on the right.

Usage:
    python bias_with_prior_histogram.py prior_cache.json
    python bias_with_prior_histogram.py prior_cache.json --output-dir figures

The original bias.py is not imported: it writes flickr_bias.pdf on import.
This script preserves its left-hand data/style and writes separate new files.
All prior entries contribute -mean_nll; the fixed x limits only crop the view.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_prior_log_likelihood import (
    Distribution, X_LABEL, Y_LABEL, load_mean_log_likelihoods, prepare_distribution,
)


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_STEM = "flickr_bias_with_prior_histogram"
FIGURE_SIZE_INCHES = (16, 6)
WIDTH_RATIOS = (1.4, 1)
METHODS = ("RGP", "CGP", "Modified\nCGP", "FreeRet")
CONDITIONS = ("t2i", "i2t")
# Same rows/columns as bias.py, including FreeRet [t2i, i2t] = [80.1, 68.4].
RECALL_SCORES = np.array([[75.0, 83.2, 83.2, 80.1], [79.2, 69.4, 78.1, 68.4]])
COLORS = ("#f57c6e", "#f2b56f")
HYBRID_BASELINES = ((70.6, "blue", "i2t hybrid"), (62.9, "green", "t2i hybrid"))
PLOT_STYLE = {
    "font.family": "DejaVu Sans", "font.size": 12,
    "pdf.fonttype": 42, "ps.fonttype": 42,
}


def _plot_comparison(ax):
    """Copy the original left panel without importing its side-effectful module."""
    n_groups = len(CONDITIONS)
    bar_width = 0.05
    index = np.arange(len(METHODS)) * (n_groups * bar_width + bar_width)
    for i, condition in enumerate(CONDITIONS):
        offsets = index + (i + 0.5) * bar_width
        bars = ax.bar(offsets, RECALL_SCORES[i], bar_width, label=condition,
                      color=COLORS[i], hatch="", edgecolor="black")
        for bar in bars:
            height = bar.get_height()
            if not np.isfinite(height):
                continue
            ax.annotate(str(height), xy=(bar.get_x() + bar.get_width() / 2, height - 0.8),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                        fontsize=20)
    ax.set_ylabel("r@1", fontsize=25)
    ax.set_xticks(index + n_groups * bar_width / 2)
    ax.set_xticklabels(METHODS, fontsize=25)
    plot_left, plot_right = index[0], index[-1] + n_groups * bar_width
    plot_margin = (plot_right - plot_left) * 0.05
    ax.set_xlim(plot_left - plot_margin, plot_right + plot_margin)
    ax.tick_params(axis="y", labelsize=20)
    for value, color, label in HYBRID_BASELINES:
        ax.axhline(y=value, color=color, linestyle="--", label=label)
        ax.annotate(str(value), xy=(0, value), xycoords=ax.get_yaxis_transform(),
                    xytext=(-8, 0), textcoords="offset points", color="black", va="bottom",
                    ha="right", fontsize=18, annotation_clip=False, clip_on=False)
    ax.legend(["i2t hybrid", "t2i hybrid", "t2i", "i2t"], ncol=1, fontsize=25)


def _plot_histogram(ax, distribution: Distribution):
    ax.hist(distribution.values, bins=distribution.bin_edges, density=True,
            color=COLORS[0], edgecolor="white", linewidth=0.35, alpha=1.0)
    ax.set_xlim(distribution.xlim)
    ax.set_ylim(distribution.ylim)
    ax.set_xlabel(X_LABEL, fontsize=22)
    ax.set_ylabel(Y_LABEL, fontsize=22)
    ax.tick_params(axis="both", which="major", labelsize=25)
    # Keep the large x/y tick labels separate at the lower-left corner.
    ax.tick_params(axis="x", pad=14)
    ax.grid(axis="y", color="#d8d8d8", linestyle="--", linewidth=0.5)
    ax.set_axisbelow(True)


def plot_bias_with_prior(distribution: Distribution):
    with matplotlib.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=FIGURE_SIZE_INCHES,
                                 gridspec_kw={"width_ratios": WIDTH_RATIOS})
        try:
            _plot_comparison(axes[0])
            _plot_histogram(axes[1], distribution)
            fig.tight_layout()
        except Exception:
            plt.close(fig)
            raise
    return fig, axes


def save_figure(fig, output_dir: Path = SCRIPT_DIR) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"{OUTPUT_STEM}.pdf"
    png_path = output_dir / f"{OUTPUT_STEM}.png"
    with matplotlib.rc_context(PLOT_STYLE):
        fig.savefig(pdf_path, format="pdf", bbox_inches="tight", pad_inches=0.05)
        fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight", pad_inches=0.05)
    return pdf_path, png_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache_json", type=Path, help="Prior-cache JSON containing entries[].mean_nll.")
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR,
                        help="Output directory (default: alongside this script).")
    args = parser.parse_args(argv)
    try:
        distribution = prepare_distribution(load_mean_log_likelihoods(args.cache_json))
    except (OSError, ValueError, np.linalg.LinAlgError) as exc:
        parser.error(str(exc))
    fig, _ = plot_bias_with_prior(distribution)
    try:
        paths = save_figure(fig, args.output_dir)
    finally:
        plt.close(fig)
    print(f"Prior observations: {distribution.values.size}")
    print(f"Mean log-likelihood: {distribution.values.mean():.8f}")
    print(f"Histogram bins (Freedman-Diaconis): {distribution.bin_edges.size - 1}")
    for path in paths:
        print(f"Saved: {path.resolve()}")


if __name__ == "__main__":
    main()
