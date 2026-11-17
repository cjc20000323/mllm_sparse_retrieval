"""Plot dense/sparse hybrid-weight recall curves as separate dataset figures.

The x-axis is the dense-representation weight alpha.  The sparse-representation
weight is 1 - alpha, matching the fusion sweep in ``src/search.py``.

The populated series use LLaVA-Mistral-7B on Flickr, WebQA, RSTPReid, and LLaVA.
Legend colors identify datasets and retrieval directions without model names.
Source recall fractions are converted to percentages for plotting.
Each of Recall@1, Recall@5, Recall@10, and their per-weight average is exported
as three independent single-panel PDFs: Flickr, RSTPReid / LLaVA, and WebQA.
This produces 12 files in the current working directory, named
``hybrid_weight_recall_{panel}_{metric}.pdf``. The RSTPReid / LLaVA comparison
retains the ``person_retrieval`` filename key. Existing three-panel PDFs and
standalone LLaVA PDFs are neither overwritten nor deleted. No CLI changes.
Flickr Recall@10 retains its compact, large-type layout. Flickr and RSTPReid / LLaVA
Average Recall use rstpreid_length.pdf's landscape page size,
large typography, grid, and stroke sizes, with internal legends and no title.
Each series dictionary also receives an ``average`` entry
containing the element-wise mean of Recall@1, Recall@5, and Recall@10.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


ALPHA = np.arange(0.1, 1.0, 0.1)
EMPTY = np.array([], dtype=float)
RECALL_LEVELS = (1, 5, 10)
OUTPUT_PATH_TEMPLATE = "hybrid_weight_recall_{panel}_{metric}.pdf"
PANEL_FIGSIZE = (7, 6)
COMPACT_FLICKR_FIGSIZE = (3.4, 3.8)
COMPACT_FLICKR_METRICS = (10,)
PAPER_AVERAGE_PANELS = ("flickr", "person_retrieval")
# Measured from rstpreid_length.pdf; fixed bounds prevent data-dependent crops.
PAPER_AVERAGE_PAGE_SIZE_POINTS = (526.990625, 354.5648879717)
PAPER_AVERAGE_AXES_BOUNDS_POINTS = (76.990625, 67.834375, 446.4, 277.2)
PAPER_AVERAGE_FIGSIZE = tuple(value / 72 for value in PAPER_AVERAGE_PAGE_SIZE_POINTS)
# The pastel palette follows plot_alpha.py, plot_prompt.py, and plot_length.py.
COLORS = {
    "flickr_t2i": "#f57c6e",
    "flickr_i2t": "#71b7ed",
    "rstpreid": "#f57c6e",
    "llava": "#71b7ed",
    "webqa": "#f2b56f",
}

# Colors identify datasets/directions; retain the existing line and marker styles.
LINESTYLE = "--"


FLICKR_SERIES = {
    "Flickr T2I": (
        {
            # stage2_alpha_sweep (3).csv: flickr/test, target hybrid/all, T2I.
            # Alpha 0.1..0.9; 5000 queries; recall fractions converted to percentages.
            1: np.array([49.14, 51.28, 53.32, 55.86, 58.08, 60.48, 61.88, 62.24, 61.1]),
            5: np.array([75.08, 77.38, 79.48, 81.22, 82.48, 83.98, 85.0, 84.58, 83.84]),
            10: np.array([83.42, 85.0, 86.4, 87.42, 89.06, 90.02, 90.46, 89.98, 89.16]),
        },
        COLORS["flickr_t2i"],
        LINESTYLE,
        "o",
    ),
    "Flickr I2T": (
        {
            # stage2_alpha_sweep (2).csv: flickr/test, target hybrid/all, I2T.
            # Alpha 0.1..0.9; 1000 queries; recall fractions converted to percentages.
            1: np.array([55.6, 57.1, 59.8, 62.7, 64.5, 67.1, 69.6, 70.8, 71.4]),
            5: np.array([79.8, 80.7, 82.4, 85.0, 86.6, 88.6, 89.8, 90.5, 90.2]),
            10: np.array([86.6, 87.4, 89.3, 90.3, 92.3, 93.6, 95.0, 95.4, 95.5]),
        },
        COLORS["flickr_i2t"],
        LINESTYLE,
        "s",
    ),
}


RSTPREID_SERIES = {
    "RSTPReid": (
        {
            # stage2_alpha_sweep.csv: RSTPReid/test, target hybrid/all.
            # Alpha 0.1..0.9; recall fractions converted to percentages.
            1: np.array([8.4, 8.55, 8.6, 8.85, 9.15, 9.3, 9.65, 9.05, 9.2]),
            5: np.array([20.5, 20.8, 21.25, 22.35, 22.55, 24.35, 25.55, 27.35, 27.55]),
            10: np.array([30.95, 31.35, 32.45, 33.35, 35.5, 36.3, 37.8, 37.7, 38.15]),
        },
        COLORS["rstpreid"],
        LINESTYLE,
        "o",
    ),
}


WEBQA_SERIES = {
    "WebQA": (
        {
            # Current 0_1_0_9.xlsx ... 0_9_0_1.xlsx, hybrid results in row 4.
            1: np.array([
                38.73407643312102,
                41.52070063694268,
                42.23726114649682,
                43.27229299363057,
                43.15286624203822,
                42.078025477707,
                39.53025477707006,
                36.78343949044586,
                32.96178343949044,
            ]),
            5: np.array([
                60.11146496815286,
                62.57961783439491,
                64.53025477707006,
                65.76433121019108,
                65.92356687898089,
                64.68949044585988,
                62.14171974522293,
                58.0015923566879,
                54.89649681528662,
            ]),
            10: np.array([
                68.47133757961783,
                70.54140127388535,
                72.65127388535031,
                74.1640127388535,
                73.80573248407644,
                72.29299363057325,
                70.14331210191083,
                66.64012738853503,
                62.85828025477708,
            ]),
        },
        COLORS["webqa"],
        LINESTYLE,
        "o",
    ),
}


LLAVA_SERIES = {
    "LLaVA": (
        {
            # stage2_alpha_sweep (1).csv: llava/test, target hybrid/all.
            # Alpha 0.1..0.9; recall fractions converted to percentages.
            1: np.array([
                36.015625,
                39.35546875,
                43.57421875,
                48.92578125,
                54.4921875,
                59.70703125,
                63.41796875,
                64.5703125,
                62.79296875,
            ]),
            5: np.array([
                63.4765625,
                67.3046875,
                71.93359375,
                76.73828125,
                80.9375,
                84.35546875,
                86.640625,
                87.5,
                86.640625,
            ]),
            10: np.array([
                74.47265625,
                77.96875,
                81.50390625,
                85.52734375,
                88.65234375,
                90.72265625,
                91.9921875,
                92.71484375,
                92.44140625,
            ]),
        },
        COLORS["llava"],
        LINESTYLE,
        "s",
    ),
}


AVERAGE_RECALL_KEY = "average"


def add_average_recall(series_definitions):
    """Add the per-weight mean of Recall@1, Recall@5, and Recall@10."""
    for label, (recall_values, _color, _linestyle, _marker) in series_definitions.items():
        values = [
            np.asarray(recall_values.get(recall, EMPTY), dtype=float)
            for recall in RECALL_LEVELS
        ]
        if any(value.size == 0 for value in values):
            recall_values[AVERAGE_RECALL_KEY] = EMPTY
            continue
        if len({value.size for value in values}) != 1:
            raise ValueError(f"{label}: recall arrays must have the same length")
        recall_values[AVERAGE_RECALL_KEY] = np.mean(np.vstack(values), axis=0)


add_average_recall(FLICKR_SERIES)
add_average_recall(RSTPREID_SERIES)
add_average_recall(WEBQA_SERIES)
add_average_recall(LLAVA_SERIES)


def build_results(series_definitions, recall):
    """Build one panel's series for the requested recall level."""
    results = {}
    for label, (recall_values, color, linestyle, marker) in series_definitions.items():
        y_values = recall_values.get(recall, EMPTY)
        x_values = ALPHA if y_values.size else EMPTY
        results[label] = (x_values, y_values, color, linestyle, marker)
    return results


def build_person_retrieval_results(recall):
    """Combine RSTPReid and LLaVA using their dataset labels."""
    results = {}
    for series_definitions in (RSTPREID_SERIES, LLAVA_SERIES):
        results.update(build_results(series_definitions, recall))
    return results


def plot_panel(ax, title, series, recall_label):
    """Draw one recall-vs-alpha panel using the existing project style."""
    has_data = False
    for label, (x_values, y_values, color, linestyle, marker) in series.items():
        x_values = np.asarray(x_values, dtype=float)
        y_values = np.asarray(y_values, dtype=float)
        if x_values.size == 0 or y_values.size == 0:
            continue
        if x_values.size != y_values.size:
            raise ValueError(f"{label}: x/y arrays must have the same length")
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
    ax.set_ylabel(recall_label, fontsize=13)
    ax.set_xticks(ALPHA)
    ax.set_xlim(0.05, 0.95)
    ax.tick_params(axis="both", which="major", labelsize=11)
    ax.grid(True, alpha=0.35)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="best", fontsize=9, frameon=True)

    if not has_data:
        ax.text(
            0.5,
            0.5,
            "No data supplied",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=11,
            color="#666666",
        )


def build_panels(metric):
    """Build the three dataset panels for the requested recall metric."""
    return (
        ("flickr", "Flickr · image-text retrieval", build_results(FLICKR_SERIES, metric)),
        ("person_retrieval", "RSTPReid / LLaVA · hybrid retrieval",
         build_person_retrieval_results(metric)),
        ("webqa", "WebQA · mixed-modal retrieval", build_results(WEBQA_SERIES, metric)),
    )


def _apply_compact_flickr_style(fig, ax):
    """Keep ~7.7pt text after each figure is scaled to 4.1cm paper width."""
    ax.set_title(ax.get_title().replace(" retrieval", "\nretrieval"), fontsize=18, pad=7)
    ax.set_xlabel("Dense weight α\n(sparse = 1 − α)", fontsize=18, labelpad=5)
    ax.set_ylabel(ax.get_ylabel(), fontsize=18, labelpad=4)
    ax.set_xticks(ALPHA[::2])
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.tick_params(axis="both", labelsize=16, pad=3)
    handles, labels = ax.get_legend_handles_labels()
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    if handles:
        legend = ax.legend(
            handles, labels,
            loc="lower center", bbox_to_anchor=(0.5, 0.01), bbox_transform=fig.transFigure,
            ncol=2, fontsize=16, frameon=True, handlelength=1.15, handletextpad=0.3,
            columnspacing=0.7, borderpad=0.3, labelspacing=0.3,
        )
        # Reserve the footer explicitly; include this artist again when exporting.
        legend.set_in_layout(False)
    fig.tight_layout(rect=(0, 0.2, 1, 1), pad=0.55)


def _apply_paper_average_style(fig, ax, panel):
    """Match the length figure's landscape layout without changing curve data."""
    width, height = PAPER_AVERAGE_PAGE_SIZE_POINTS
    left, bottom, panel_width, panel_height = PAPER_AVERAGE_AXES_BOUNDS_POINTS
    if panel == "person_retrieval":
        # Reserve space for the three-digit tick and the vertical axis label.
        left += 20
        panel_width -= 20
        panel_height -= 8
    fig.subplots_adjust(left=left / width, right=(left + panel_width) / width,
                        bottom=bottom / height, top=(bottom + panel_height) / height)
    ax.set_title("")
    ax.set_xlabel("Dense weight α", fontsize=25, labelpad=4)
    ax.set_ylabel(ax.get_ylabel(), fontsize=25, labelpad=4)
    ax.set_xticks(ALPHA[::2])
    ax.set_yticks([70, 80, 90] if panel == "flickr" else [0, 20, 40, 60, 80, 100])
    ax.set_ylim((64, 92) if panel == "flickr" else (0, 100))
    ax.tick_params(axis="both", labelsize=30, pad=3.5, length=3.5, width=0.8)
    ax.grid(True, color="#b0b0b0", alpha=1, linewidth=0.8, linestyle="-")
    for line in ax.lines:
        line.set_linewidth(3)
        line.set_markersize(8)
    handles, labels = ax.get_legend_handles_labels()
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    if not handles:
        return
    ax.legend(
        handles, labels, loc="lower right" if panel == "flickr" else "center right",
        bbox_to_anchor=None if panel == "flickr" else (1, 0.43),
        ncol=2, fontsize=25, handlelength=1.4, handletextpad=0.4,
        borderpad=0.4, columnspacing=0.4 if panel == "flickr" else 0.8, labelspacing=0.3,
        borderaxespad=0.25, frameon=True,
    )


def create_panel_figure(panel, title, series, metric_label, *, compact=False, paper_layout=False):
    """Build a standalone panel without hiding its curves behind the legend."""
    if paper_layout and panel not in PAPER_AVERAGE_PANELS:
        raise ValueError(f"Paper layout requires one of {PAPER_AVERAGE_PANELS}, got {panel!r}")
    figsize = PAPER_AVERAGE_FIGSIZE if paper_layout else COMPACT_FLICKR_FIGSIZE if compact else PANEL_FIGSIZE
    fig, ax = plt.subplots(figsize=figsize)
    try:
        plot_panel(ax, title, series, metric_label)
        if panel == "person_retrieval" and not paper_layout and ax.get_legend() is not None:
            # Keep dataset labels below the standalone plotting area.
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
                      ncol=2, fontsize=9, frameon=True)
        if paper_layout:
            _apply_paper_average_style(fig, ax, panel)
        elif compact:
            _apply_compact_flickr_style(fig, ax)
        else:
            fig.tight_layout()
    except BaseException:
        plt.close(fig)
        raise
    return fig, ax


def save_single_panels(metric, metric_label, *, panels=None):
    """Save independent dataset figures, optionally selecting panels by name."""
    metric_suffix = "average" if metric == AVERAGE_RECALL_KEY else f"recall_{metric}"
    selected_panels = set(panels) if panels is not None else None
    for panel, title, series in build_panels(metric):
        if selected_panels is not None and panel not in selected_panels:
            continue
        output_path = OUTPUT_PATH_TEMPLATE.format(panel=panel, metric=metric_suffix)
        paper_layout = metric == AVERAGE_RECALL_KEY and panel in PAPER_AVERAGE_PANELS
        compact = panel == "flickr" and metric in COMPACT_FLICKR_METRICS
        fig, ax = create_panel_figure(panel, title, series, metric_label,
                                    compact=compact, paper_layout=paper_layout)
        try:
            extra = {"bbox_extra_artists": [ax.get_legend()]} if compact and ax.get_legend() is not None else {}
            export_style = {"pdf.fonttype": 42} if compact or paper_layout else {}
            if paper_layout:
                export_style["savefig.bbox"] = None
            with plt.rc_context(export_style):
                fig.savefig(output_path, format="pdf",
                            bbox_inches=None if paper_layout else "tight",
                            pad_inches=0 if paper_layout else 0.05, **extra)
        finally:
            plt.close(fig)
        print(f"Saved figure to {output_path}")


def main():
    for recall in RECALL_LEVELS:
        save_single_panels(recall, f"Recall@{recall}")

    save_single_panels(AVERAGE_RECALL_KEY, "Average Recall")


if __name__ == "__main__":
    main()
