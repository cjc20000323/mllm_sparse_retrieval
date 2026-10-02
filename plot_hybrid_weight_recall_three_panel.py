"""Plot dense/sparse hybrid-weight recall curves in a three-panel figure.

The x-axis is the dense-representation weight alpha.  The sparse-representation
weight is 1 - alpha, matching the fusion sweep in ``src/search.py``.

Flickr and RSTPReid values below are copied from the existing alpha-ablation
script.  WebQA values were not present in this repository, so those arrays are
left as NaN placeholders and should be replaced with the measured results.
"""

import numpy as np
import matplotlib.pyplot as plt


ALPHA = np.arange(0.1, 1.0, 0.1)
RECALL_LABEL = "Recall@1"
OUTPUT_PATH = "hybrid_weight_recall_three_panel.pdf"

# The pastel palette follows plot_alpha.py, plot_prompt.py, and plot_length.py.
COLORS = {
    "llava_8b": "#f57c6e",
    "llava_mistral": "#71b7ed",
    "llava_8b_i2t": "#f2b56f",
    "llava_mistral_i2t": "#84c3b7",
}


FLICKR_RESULTS = {
    "LLaVA-8B · image retrieval": (
        ALPHA,
        np.array([53.2, 55.8, 58.0, 59.8, 61.1, 62.0, 62.2, 61.8, 60.3]),
        COLORS["llava_8b"],
        "-",
        "o",
    ),
    "LLaVA-8B · text retrieval": (
        ALPHA,
        np.array([58.0, 60.8, 64.3, 67.0, 69.9, 72.1, 72.3, 73.6, 73.1]),
        COLORS["llava_8b_i2t"],
        "--",
        "s",
    ),
    "LLaVA-Mistral-7B · image retrieval": (
        ALPHA,
        np.array([59.7, 62.0, 63.5, 65.0, 65.5, 65.7, 65.0, 64.2, 62.9]),
        COLORS["llava_mistral"],
        "-",
        "o",
    ),
    "LLaVA-Mistral-7B · text retrieval": (
        ALPHA,
        np.array([70.9, 73.1, 74.2, 75.0, 74.8, 75.4, 75.2, 74.5, 73.6]),
        COLORS["llava_mistral_i2t"],
        "--",
        "s",
    ),
}


RSTPREID_RESULTS = {
    # The existing alpha-ablation array contains eight measured points (0.1-0.8).
    # The ninth point is kept as NaN until the alpha=0.9 result is available.
    "LLaVA-8B": (
        ALPHA,
        np.array([7.0, 7.5, 7.9, 7.6, 7.7, 7.5, 7.8, 7.7, np.nan]),
        COLORS["llava_8b"],
        "-",
        "o",
    ),
    "LLaVA-Mistral-7B": (
        ALPHA,
        np.array([9.4, 9.9, 10.7, 10.6, 11.2, 11.5, 11.1, 10.8, 10.6]),
        COLORS["llava_mistral"],
        "-",
        "o",
    ),
}


WEBQA_RESULTS = {
    # Replace these placeholders with the measured WebQA recall values.
    "LLaVA-8B": (
        ALPHA,
        np.full(ALPHA.shape, np.nan, dtype=float),
        COLORS["llava_8b"],
        "-",
        "o",
    ),
    "LLaVA-Mistral-7B": (
        ALPHA,
        np.full(ALPHA.shape, np.nan, dtype=float),
        COLORS["llava_mistral"],
        "-",
        "o",
    ),
}


def plot_panel(ax, title, series):
    """Draw one recall-vs-alpha panel using the existing project style."""
    has_data = False
    for label, (x_values, y_values, color, linestyle, marker) in series.items():
        y_values = np.asarray(y_values, dtype=float)
        has_data = has_data or np.isfinite(y_values).any()
        ax.plot(
            x_values,
            y_values,
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=2.5,
            marker=marker,
            markersize=6,
        )

    ax.set_title(title, fontsize=16, pad=10)
    ax.set_xlabel("Dense weight α (sparse = 1 − α)", fontsize=13)
    ax.set_ylabel(RECALL_LABEL, fontsize=13)
    ax.set_xticks(ALPHA)
    ax.set_xlim(0.05, 0.95)
    ax.tick_params(axis="both", which="major", labelsize=11)
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best", fontsize=9, frameon=True)

    if not has_data:
        ax.text(
            0.5,
            0.5,
            "Replace NaN placeholders\nwith WebQA results",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=11,
            color="#666666",
        )


def main():
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharex=True)

    plot_panel(axes[0], "Flickr · image-text retrieval", FLICKR_RESULTS)
    plot_panel(axes[1], "RSTPReid · person text retrieval", RSTPREID_RESULTS)
    plot_panel(axes[2], "WebQA · mixed-modal retrieval", WEBQA_RESULTS)

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, format="pdf", bbox_inches="tight", pad_inches=0.05)
    print(f"Saved figure to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
