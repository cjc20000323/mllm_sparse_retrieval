"""Plot dense/sparse hybrid-weight recall curves as separate dataset figures.

The x-axis is the dense-representation weight alpha.  The sparse-representation
weight is 1 - alpha, matching the fusion sweep in ``src/search.py``.

The populated series are the supplied Llama3-8B and Llama3-Mistral-7B Flickr
and WebQA results plus the RSTPReid / CUHK-PEDES person text-retrieval results.
The source workbooks store recall as a fraction; the plot uses percentages.
Each of Recall@1, Recall@5, Recall@10, and their per-weight average is exported
as three independent single-panel PDFs: Flickr, person retrieval, and WebQA.
This produces 12 files in the current working directory, named
``hybrid_weight_recall_{panel}_{metric}.pdf``. Existing three-panel PDFs are
neither overwritten nor deleted. Run this script as before; no CLI changes.
Flickr Recall@10 retains its compact, large-type layout. Flickr and person
Average Recall match rstpreid_length.pdf's landscape page and axes bounds,
large typography, grid, and stroke sizes, with internal legends and no title.
All other series are intentionally empty because they are not part of the
supplied result. Each series dictionary also receives an ``average`` entry
containing the element-wise mean of Recall@1, Recall@5, and Recall@10.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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
COMPACT_FLICKR_LEGEND = {
    "LLaVA-8B · image retrieval": "8B T2I",
    "LLaVA-8B · text retrieval": "8B I2T",
    "LLaVA-Mistral-7B · image retrieval": "Mistral T2I",
    "LLaVA-Mistral-7B · text retrieval": "Mistral I2T",
}

# The pastel palette follows plot_alpha.py, plot_prompt.py, and plot_length.py.
COLORS = {
    "llava_8b": "#f57c6e",
    "llava_mistral": "#71b7ed",
    "llava_8b_i2t": "#f2b56f",
    "llava_mistral_i2t": "#84c3b7",
}

# Line style identifies the model consistently across datasets and retrieval
# directions; colors and markers distinguish the retrieval direction.
LINESTYLES = {
    "llava_8b": "-",
    "llava_mistral": "--",
}


FLICKR_SERIES = {
    "LLaVA-8B · image retrieval": (
        {
            # Current 0_1_0_9.xlsx ... 0_9_0_1.xlsx, hybrid results in row 4.
            1: np.array([53.2, 55.82, 58.02, 59.84, 61.16, 62.08, 62.2, 61.86, 60.3]),
            5: np.array([77.98, 80.0, 81.92, 83.12, 83.48, 83.5, 83.38, 82.62, 81.88]),
            10: np.array([84.16, 85.9, 87.18, 87.98, 88.64, 88.72, 88.3, 87.98, 87.48]),
        },
        COLORS["llava_8b"],
        LINESTYLES["llava_8b"],
        "o",
    ),
    "LLaVA-8B · text retrieval": (
        {
            # Previously supplied image-to-text workbooks, hybrid results in row 4.
            1: np.array([58.0, 60.8, 64.3, 67.0, 69.9, 72.1, 72.3, 73.6, 73.1]),
            5: np.array([83.9, 86.0, 87.9, 89.2, 90.2, 91.7, 91.2, 91.6, 91.0]),
            10: np.array([91.0, 92.5, 94.4, 94.8, 95.5, 96.0, 95.7, 95.3, 95.5]),
        },
        COLORS["llava_8b_i2t"],
        LINESTYLES["llava_8b"],
        "s",
    ),
    "LLaVA-Mistral-7B · image retrieval": (
        {
            # Current 0_1_0_9.xlsx ... 0_9_0_1.xlsx, hybrid results in row 4.
            1: np.array([59.72, 62.06, 63.58, 65.0, 65.54, 65.72, 65.08, 64.24, 62.94]),
            5: np.array([84.54, 85.82, 86.74, 87.3, 87.54, 87.4, 86.5, 85.94, 84.6]),
            10: np.array([90.64, 91.72, 92.36, 92.94, 92.98, 92.62, 92.02, 91.28, 90.2]),
        },
        COLORS["llava_mistral"],
        LINESTYLES["llava_mistral"],
        "o",
    ),
    "LLaVA-Mistral-7B · text retrieval": (
        {
            # Current 0_1_0_9.xlsx ... 0_9_0_1.xlsx, hybrid results in row 4.
            1: np.array([70.9, 73.1, 74.2, 75.0, 74.8, 75.4, 75.2, 74.5, 73.6]),
            5: np.array([91.2, 92.4, 93.0, 93.9, 93.9, 93.9, 93.8, 92.7, 91.6]),
            10: np.array([95.0, 96.0, 96.2, 97.0, 97.7, 97.3, 97.0, 96.7, 95.8]),
        },
        COLORS["llava_mistral_i2t"],
        LINESTYLES["llava_mistral"],
        "s",
    ),
}


RSTPREID_SERIES = {
    "LLaVA-8B": (
        {
            # Current 0_1_0_9.xlsx ... 0_9_0_1.xlsx, hybrid results in row 4.
            1: np.array([7.0, 7.55, 7.9, 7.6, 7.6, 7.7, 7.5, 7.8, 7.7]),
            5: np.array([19.55, 19.5, 20.35, 20.45, 21.4, 21.1, 21.35, 21.05, 21.3]),
            10: np.array([30.15, 30.65, 31.95, 31.8, 32.0, 32.8, 33.25, 32.25, 32.8]),
        },
        COLORS["llava_8b"],
        LINESTYLES["llava_8b"],
        "o",
    ),
    "LLaVA-Mistral-7B": (
        {
            # Current 0_1_0_9.xlsx ... 0_9_0_1.xlsx, hybrid results in row 4.
            1: np.array([9.4, 9.9, 10.7, 10.6, 11.25, 11.5, 11.15, 10.8, 10.6]),
            5: np.array([28.85, 29.25, 29.6, 30.05, 30.0, 29.75, 29.75, 29.15, 27.8]),
            10: np.array([39.95, 40.35, 40.5, 41.25, 41.25, 41.15, 40.4, 39.15, 38.2]),
        },
        COLORS["llava_mistral"],
        LINESTYLES["llava_mistral"],
        "o",
    ),
}


WEBQA_SERIES = {
    "LLaVA-8B": (
        {},
        COLORS["llava_8b"],
        LINESTYLES["llava_8b"],
        "o",
    ),
    "LLaVA-Mistral-7B": (
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
        COLORS["llava_mistral"],
        LINESTYLES["llava_mistral"],
        "o",
    ),
}


CUHK_PEDES_SERIES = {
    "LLaVA-8B": (
        {
            # Current 0_1_0_9.xlsx ... 0_9_0_1.xlsx, hybrid results in row 4.
            1: np.array([
                4.8408057179987,
                5.003248862897985,
                5.214424951267056,
                5.766731643924626,
                6.059129304743339,
                6.059129304743339,
                6.010396361273554,
                5.782975958414555,
                5.474333983105913,
            ]),
            5: np.array([
                12.08576998050682,
                12.52436647173489,
                13.12540610786225,
                13.53151397011046,
                13.64522417153996,
                13.77517868745939,
                13.59649122807017,
                13.6127355425601,
                13.1578947368421,
            ]),
            10: np.array([
                17.34892787524366,
                18.01494476933073,
                18.7946718648473,
                19.03833658219623,
                19.33073424301494,
                19.33073424301494,
                18.85964912280702,
                18.42105263157895,
                17.94996751137102,
            ]),
        },
        COLORS["llava_8b"],
        LINESTYLES["llava_8b"],
        "s",
    ),
    "LLaVA-Mistral-7B": (
        {
            # Current 0_1_0_9.xlsx ... 0_9_0_1.xlsx, hybrid results in row 4.
            1: np.array([
                7.92722547108512,
                8.203378817413905,
                8.80441845354126,
                9.048083170890188,
                9.178037686809616,
                8.91812865497076,
                8.560753736192332,
                8.105912930474334,
                7.358674463937621,
            ]),
            5: np.array([
                17.80376868096166,
                18.66471734892788,
                19.50942170240416,
                19.86679662118259,
                19.83430799220273,
                19.85055230669265,
                19.33073424301494,
                18.40480831708902,
                17.08901884340481,
            ]),
            10: np.array([
                25.3411306042885,
                26.44574398960364,
                27.16049382716049,
                27.38791423001949,
                27.32293697205977,
                26.88434048083171,
                26.13710201429499,
                24.75633528265107,
                23.68421052631579,
            ]),
        },
        COLORS["llava_mistral"],
        LINESTYLES["llava_mistral"],
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
add_average_recall(CUHK_PEDES_SERIES)


def build_results(series_definitions, recall):
    """Build one panel's series for the requested recall level."""
    results = {}
    for label, (recall_values, color, linestyle, marker) in series_definitions.items():
        y_values = recall_values.get(recall, EMPTY)
        x_values = ALPHA if y_values.size else EMPTY
        results[label] = (x_values, y_values, color, linestyle, marker)
    return results


def build_person_retrieval_results(recall):
    """Combine RSTPReid and CUHK-PEDES without colliding model labels."""
    results = {}
    for dataset, series_definitions in (
        ("RSTPReid", RSTPREID_SERIES),
        ("CUHK-PEDES", CUHK_PEDES_SERIES),
    ):
        for label, plot_values in build_results(series_definitions, recall).items():
            results[f"{dataset} · {label}"] = plot_values
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
    """Keep the original three panel groupings and their series unchanged."""
    return (
        ("flickr", "Flickr · image-text retrieval", build_results(FLICKR_SERIES, metric)),
        ("person_retrieval", "RSTPReid / CUHK-PEDES · person text retrieval",
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
            handles, [COMPACT_FLICKR_LEGEND.get(label, label) for label in labels],
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
    fig.subplots_adjust(left=left / width, right=(left + panel_width) / width,
                        bottom=bottom / height, top=(bottom + panel_height) / height)
    ax.set_title("")
    ax.set_xlabel("Dense weight α", fontsize=25, labelpad=4)
    ax.set_ylabel(ax.get_ylabel(), fontsize=25, labelpad=4)
    ax.set_xticks(ALPHA[::2])
    ax.set_yticks([70, 80, 90] if panel == "flickr" else [0, 10, 20, 30])
    ax.set_ylim((64, 92) if panel == "flickr" else (0, 32))
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
    if panel == "flickr":
        labels = [COMPACT_FLICKR_LEGEND.get(label, label).replace("8B", "LLaMA", 1) for label in labels]
    else:
        # Dataset determines marker shape; model determines color and line style.
        # Separate the keys while preserving all four curve identities.
        handles = [
            Line2D([], [], color="#555555", linestyle="none", marker=marker, markersize=8)
            for marker in ("o", "s")
        ] + [
            Line2D([], [], color=COLORS[model], linestyle=LINESTYLES[model], linewidth=3)
            for model in ("llava_8b", "llava_mistral")
        ]
        labels = ["RSTPReid", "CUHK-PEDES", "LLaMA", "Mistral"]
    ax.legend(
        handles, labels, loc="lower right" if panel == "flickr" else "lower center",
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
            # Four long dataset/model labels cover the lower curve in a
            # standalone panel. Keep the same styles, below the plotting area.
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
