"""Plot separate histogram/KDE figures from a rerank prior-cache JSON.

Usage:
    python plot_prior_log_likelihood.py path/to/prior_cache.json
    python plot_prior_log_likelihood.py path/to/prior_cache.json --output-dir figures

Each cache entry contributes one observation: mean log-likelihood = -mean_nll.
No exponentiation, deduplication, trimming, model loading or cache mutation is
performed. Requires NumPy, Matplotlib and SciPy; uses the noninteractive backend.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde


SCRIPT_DIR = Path(__file__).resolve().parent
FIGURE_SIZE_INCHES = (3.6, 2.4)
X_LABEL = "Average Log-likelihood"
X_LIMITS = (-10.0, -2.0)
Y_LABEL = "Density"
OUTPUT_STEMS = {
    "histogram": "prior_log_likelihood_histogram",
    "kde": "prior_log_likelihood_kde",
}
PLOT_STYLE = {
    "font.family": "DejaVu Sans", "font.size": 8.5,
    "axes.labelsize": 10.5, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "pdf.fonttype": 42, "ps.fonttype": 42,
}


@dataclass(frozen=True)
class Distribution:
    values: np.ndarray
    bin_edges: np.ndarray
    histogram_density: np.ndarray
    grid: np.ndarray
    kde_density: np.ndarray
    bandwidth: float
    xlim: tuple[float, float]
    ylim: tuple[float, float]


def load_mean_log_likelihoods(cache_json: Path) -> np.ndarray:
    """Read every entry, rejecting invalid scores rather than silently skipping."""
    with Path(cache_json).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or not isinstance(payload.get("entries"), list):
        raise ValueError("Prior cache must contain an 'entries' list.")
    if not payload["entries"]:
        raise ValueError("Prior cache entries must not be empty.")
    values = []
    for index, entry in enumerate(payload["entries"]):
        score = entry.get("mean_nll") if isinstance(entry, dict) else None
        if (isinstance(score, bool) or not isinstance(score, (int, float))
                or not math.isfinite(score) or score < 0):
            raise ValueError(f"Entry {index + 1}: mean_nll must be a finite, nonnegative number.")
        values.append(-float(score))
    return np.asarray(values, dtype=np.float64)


def prepare_distribution(values: np.ndarray) -> Distribution:
    """Compute both densities and their common axes from the complete sample."""
    values = np.asarray(values, dtype=np.float64).copy()
    if values.ndim != 1 or values.size < 2:
        raise ValueError("KDE requires at least two observations in a one-dimensional sample.")
    if not np.isfinite(values).all():
        raise ValueError("All log-likelihood observations must be finite.")
    if not np.isfinite(values.std()) or values.std() == 0:
        raise ValueError("KDE requires a finite, nonzero sample variance.")
    bin_edges = np.histogram_bin_edges(values, bins="fd")
    histogram_density, _ = np.histogram(values, bins=bin_edges, density=True)
    kde = gaussian_kde(values, bw_method="scott")
    # Limit only the displayed domain; both estimators use every observation.
    xlim = X_LIMITS
    grid = np.linspace(*xlim, 1024)
    kde_density = kde(grid)
    ylim = (0.0, float(max(histogram_density.max(), kde_density.max()) * 1.12))
    return Distribution(values, bin_edges, histogram_density, grid, kde_density,
                        float(np.sqrt(kde.covariance[0, 0])), xlim, ylim)


def plot_distribution(distribution: Distribution, kind: str):
    if kind not in OUTPUT_STEMS:
        raise ValueError("Plot kind must be 'histogram' or 'kde'.")
    with matplotlib.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=FIGURE_SIZE_INCHES)
        try:
            if kind == "histogram":
                ax.hist(distribution.values, bins=distribution.bin_edges, density=True,
                        color="#71b7ed", edgecolor="white", linewidth=0.35, alpha=0.85)
            else:
                ax.plot(distribution.grid, distribution.kde_density, color="#2a6fbb", linewidth=1.7)
                ax.fill_between(distribution.grid, distribution.kde_density,
                                color="#71b7ed", alpha=0.18)
            ax.set_xlim(distribution.xlim)
            ax.set_ylim(distribution.ylim)
            ax.set_xlabel(X_LABEL)
            ax.set_ylabel(Y_LABEL)
            ax.grid(axis="y", color="#d8d8d8", linestyle="--", linewidth=0.5)
            ax.set_axisbelow(True)
            fig.tight_layout(pad=0.7)
        except Exception:
            plt.close(fig)
            raise
    return fig, ax


def save_figures(distribution: Distribution, output_dir: Path = SCRIPT_DIR) -> tuple[Path, ...]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for kind, stem in OUTPUT_STEMS.items():
        fig, _ = plot_distribution(distribution, kind)
        try:
            with matplotlib.rc_context(PLOT_STYLE):
                for extension in ("pdf", "png"):
                    path = output_dir / f"{stem}.{extension}"
                    kwargs = {"dpi": 300} if extension == "png" else {}
                    fig.savefig(path, format=extension, bbox_inches="tight", pad_inches=0.08, **kwargs)
                    paths.append(path)
        finally:
            plt.close(fig)
    return tuple(paths)


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
    paths = save_figures(distribution, args.output_dir)
    print(f"Observations: {distribution.values.size}")
    print(f"Mean log-likelihood: {distribution.values.mean():.8f}")
    print(f"Standard deviation (population): {distribution.values.std():.8f}")
    print(f"Range: [{distribution.values.min():.8f}, {distribution.values.max():.8f}]")
    print(f"Histogram bins (Freedman-Diaconis): {distribution.bin_edges.size - 1}")
    print(f"KDE bandwidth (Scott): {distribution.bandwidth:.8f}")
    for path in paths:
        print(f"Saved: {path.resolve()}")


if __name__ == "__main__":
    main()
