"""Plot CUHK-PEDES Recall@1/5/10/20 against the number of adaptive perspectives.

Example (run from the project root):
    python plot_adaptive_views_topk_cuhk_pedes.py

The supplied attachments are transcribed into the lists below. No input files
are read at runtime. Recall values remain fractions and are plotted as percent.
A 300-DPI PNG and a vector PDF are saved next to this script, independently of
the working directory. Use --output-dir only to override that location.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import textwrap


METRICS = ("R@1", "R@5", "R@10", "R@20")

# Transcribed from the supplied view_ranking.csv, in increasing rank order.
# Top-K includes the first K perspectives in this list.
VIEW_COUNTS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
VIEW_ORDER = [
    ("C", "people, such as number of people, appearance and pose"),
    ("A", "scene, such as location, indoor or outdoor setting and terrain"),
    ("D", "clothing, such as category, color, material and style"),
    ("B", "natural environment, such as weather, lighting, season and time"),
    ("J", "high-level semantics, such as events, activities and functions"),
    ("H", "visual attributes, such as color, shape, size and material"),
    ("I", "spatial layout, such as foreground and background, distance and direction"),
    ("F", "actions, such as behavior, participants, tools and targets"),
    ("G", "relations, such as spatial relations, interactions and ownership"),
    ("E", "objects, such as category, number, appearance and state"),
]

# Recall fractions transcribed from the CUHK-PEDES topk_summary.csv attachment.
# Each list follows VIEW_COUNTS; the plotting code converts these to percent.
RECALL_SERIES = {
    ("stage1", "sparse_concat"): {
        "R@1": [
            0.012833008447043534, 0.024366471734892786, 0.05051981806367771, 0.04905782975958414,
            0.04337231968810916, 0.05880441845354126, 0.0594541910331384, 0.06140350877192982,
            0.05799220272904483, 0.06481481481481481,
        ],
        "R@5": [
            0.04126055880441845, 0.0594541910331384, 0.1267056530214425, 0.12508122157244964,
            0.11809616634178037, 0.147498375568551, 0.145224171539961, 0.13905133203378817,
            0.1343404808317089, 0.14035087719298245,
        ],
        "R@10": [
            0.06286549707602339, 0.0903183885640026, 0.18144899285250163, 0.17966211825860948,
            0.17153996101364521, 0.21085120207927224, 0.19866796621182586, 0.18875893437296945,
            0.1861598440545809, 0.1949317738791423,
        ],
        "R@20": [
            0.09454191033138401, 0.13190383365821962, 0.25828460038986356, 0.25422352176738144,
            0.24285250162443145, 0.28931124106562706, 0.27469135802469136, 0.2597465886939571,
            0.260233918128655, 0.2654320987654321,
        ],
    },
    ("stage2", "dense"): {
        "R@1": [
            0.04905782975958414, 0.056855100714749834, 0.0682261208576998, 0.06838856400259909,
            0.06757634827810266, 0.06985055230669265, 0.06952566601689408, 0.0693632228719948,
            0.06887589343729694, 0.0693632228719948,
        ],
        "R@5": [
            0.11533463287849252, 0.13287849252761533, 0.16455490578297596, 0.16666666666666666,
            0.16211825860948667, 0.16764132553606237, 0.1682910981156595, 0.16439246263807666,
            0.16341780376868095, 0.1652046783625731,
        ],
        "R@10": [
            0.15756985055230668, 0.18469785575048733, 0.22612085769980506, 0.22953216374269006,
            0.2223846653671215, 0.23131903833658218, 0.22628330084470435, 0.2244964262508122,
            0.2223846653671215, 0.22709551656920077,
        ],
        "R@20": [
            0.2092267706302794, 0.24528914879792071, 0.3055555555555556, 0.3094541910331384,
            0.30246913580246915, 0.3138401559454191, 0.3062053281351527, 0.3010071474983756,
            0.29889538661468484, 0.301819363222872,
        ],
    },
    ("stage2", "sparse"): {
        "R@1": [
            0.012833008447043534, 0.024366471734892786, 0.05051981806367771, 0.04905782975958414,
            0.04337231968810916, 0.05880441845354126, 0.0594541910331384, 0.06140350877192982,
            0.05799220272904483, 0.06481481481481481,
        ],
        "R@5": [
            0.04126055880441845, 0.0594541910331384, 0.1267056530214425, 0.12508122157244964,
            0.11809616634178037, 0.147498375568551, 0.145224171539961, 0.13905133203378817,
            0.1343404808317089, 0.14035087719298245,
        ],
        "R@10": [
            0.06286549707602339, 0.0903183885640026, 0.18144899285250163, 0.17966211825860948,
            0.17153996101364521, 0.21085120207927224, 0.19866796621182586, 0.18875893437296945,
            0.1861598440545809, 0.1949317738791423,
        ],
        "R@20": [
            0.09454191033138401, 0.13190383365821962, 0.25828460038986356, 0.25422352176738144,
            0.24285250162443145, 0.28931124106562706, 0.27469135802469136, 0.2597465886939571,
            0.260233918128655, 0.2654320987654321,
        ],
    },
    ("stage2", "hybrid"): {
        "R@1": [
            0.02761533463287849, 0.03508771929824561, 0.06757634827810266, 0.06448992852501624,
            0.06481481481481481, 0.07586094866796621, 0.07504873294346978, 0.07716049382716049,
            0.07228719948018193, 0.07667316439246263,
        ],
        "R@5": [
            0.0705003248862898, 0.09015594541910331, 0.16000649772579598, 0.15789473684210525,
            0.15188434048083171, 0.1762508122157245, 0.17007797270955166, 0.16455490578297596,
            0.1596816114359974, 0.16764132553606237,
        ],
        "R@10": [
            0.10103963612735542, 0.13304093567251463, 0.22335932423651722, 0.2212475633528265,
            0.21052631578947367, 0.24626380766731643, 0.23749187784275502, 0.2235217673814165,
            0.21881091617933723, 0.22807017543859648,
        ],
        "R@20": [
            0.14912280701754385, 0.18875893437296945, 0.2997076023391813, 0.30539311241065625,
            0.28443794671864847, 0.32651072124756336, 0.3140025990903184, 0.3023066926575699,
            0.2993827160493827, 0.3125406107862248,
        ],
    },
}
EXPERIMENT = {
    "model_family": "llava_mistral",
    "model": "/root/gpufree-share/checkpoints/llava-hf-llava-v1.6-mistral-7b-hf",
    "dataset": "CUHK-PEDES",
    "split": "test",
    "task": "tbpr",
    "query_type": "text",
    "execution_mode": "two_stage",
    "hybrid_alpha": "0.5",
    "category": "all",
    "query_count": "6156",
}

SERIES_STYLES = {
    ("stage1", "sparse_concat"): ("Stage 1 - Sparse", "#0072B2", "-", "o", 8, "white", 2.6),
    ("stage2", "dense"): ("Stage 2 - Dense", "#009E73", "-", "s", 5, "#009E73", 1.8),
    ("stage2", "sparse"): ("Stage 2 - Sparse", "#E69F00", "--", "x", 5, "#E69F00", 1.8),
    ("stage2", "hybrid"): ("Stage 2 - Hybrid", "#D55E00", "-", "D", 5, "#D55E00", 1.8),
    ("stage3", "ranker"): ("Stage 3 - Ranker", "#CC79A7", "-.", "^", 6, "white", 1.8),
}


@dataclass(frozen=True)
class Perspective:
    rank: int
    label: str
    aspect: str

    @property
    def name(self):
        return self.aspect.split(", such as", 1)[0]


@dataclass
class PlotData:
    perspectives: tuple[Perspective, ...]
    metadata: dict[str, str]
    curves: dict[tuple[str, str], dict[int, tuple[float, ...]]]
    max_views: int


def build_plot_data():
    """Validate and arrange the in-code lists; no external data is loaded."""
    if not VIEW_COUNTS or VIEW_COUNTS != list(range(1, len(VIEW_COUNTS) + 1)):
        raise ValueError("VIEW_COUNTS must be consecutive integers starting at 1")
    if len(VIEW_ORDER) != len(VIEW_COUNTS):
        raise ValueError("VIEW_ORDER must have one perspective for each VIEW_COUNTS entry")
    perspectives = tuple(Perspective(rank, label, aspect) for rank, (label, aspect) in zip(VIEW_COUNTS, VIEW_ORDER))
    labels = [item.label for item in perspectives]
    if len(set(labels)) != len(labels) or any(not item.label or not item.aspect for item in perspectives):
        raise ValueError("VIEW_ORDER labels must be unique; labels and aspects must not be empty")
    if not RECALL_SERIES:
        raise ValueError("RECALL_SERIES must contain at least one series")
    curves = {}
    for key, metrics in RECALL_SERIES.items():
        for metric in METRICS:
            values = metrics.get(metric)
            if values is None or len(values) != len(VIEW_COUNTS):
                raise ValueError(f"{key}, {metric}: expected {len(VIEW_COUNTS)} Recall values")
            if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
                raise ValueError(f"{key}, {metric}: Recall values must be fractions in [0, 1]")
        curves[key] = {count: tuple(metrics[metric][index] for metric in METRICS)
                       for index, count in enumerate(VIEW_COUNTS)}
    order = {key: index for index, key in enumerate(SERIES_STYLES)}
    curves = dict(sorted(curves.items(), key=lambda pair: (order.get(pair[0], len(order)), pair[0])))
    return PlotData(perspectives, dict(EXPERIMENT), curves, max(VIEW_COUNTS))


def series_style(key, index=0):
    fallback = (f"{key[0]} - {key[1]}", f"C{index % 10}", ":", "v", 5, "white", 1.8)
    return SERIES_STYLES.get(key, fallback)


def overlap_notes(data):
    groups = []
    for key, points in data.curves.items():
        for group in groups:
            other = data.curves[group[0]]
            if points.keys() == other.keys() and all(
                    math.isclose(first, second, rel_tol=0, abs_tol=1e-12)
                    for count in points for first, second in zip(points[count], other[count])):
                group.append(key)
                break
        else:
            groups.append([key])
    return ["Overlapping curves: " + " = ".join(series_style(key)[0] for key in group)
            for group in groups if len(group) > 1]


def build_figure(data):
    """Return a Matplotlib Figure; coordinates use Recall percentages."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, StrMethodFormatter

    counts = list(range(1, data.max_views + 1))
    perspectives = data.perspectives[:data.max_views]
    footer_rows = math.ceil(len(perspectives) / 2)
    notes = overlap_notes(data)
    if any(set(points) != set(counts) for points in data.curves.values()):
        notes.append("Missing observations are shown as gaps; no interpolation is applied.")
    height = 8.8 + 0.25 * footer_rows + 0.18 * max(0, len(notes) - 1)
    bottom = (1.35 + 0.25 * footer_rows) / height
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42,
                         "ps.fonttype": 42, "axes.titlesize": 13, "axes.labelsize": 11}):
        fig, axes = plt.subplots(2, 2, figsize=(13.8, height), sharex=True)
        fig.subplots_adjust(left=0.075, right=0.975, top=0.835 - 0.017 * max(0, len(notes) - 1),
                            bottom=bottom, wspace=0.19, hspace=0.36)
        meta = data.metadata
        directions = {"tbpr": "Text to image", "t2it": "Text to image + text", "it2t": "Image + text to text"}
        direction = directions.get(meta["task"], "Image to text" if meta["query_type"] == "image" else "Text to image")
        fig.suptitle(f"{meta['dataset']} | {meta['model_family']} | {direction}", y=0.985, fontsize=16, weight="semibold")
        fig.text(0.5, 0.947, f"{meta['split']} / {meta['category']} | {meta['execution_mode'].replace('_', ' ')}"
                 f" | alpha = {meta['hybrid_alpha']} | {int(meta['query_count']):,} queries",
                 ha="center", fontsize=10, color="#555555")
        for metric_index, (metric, ax) in enumerate(zip(METRICS, axes.flat)):
            available = []
            for index, (key, points) in enumerate(data.curves.items()):
                label, color, linestyle, marker, size, face, width = series_style(key, index)
                y = [points[count][metric_index] * 100 if count in points else math.nan for count in counts]
                available.extend(value for value in y if math.isfinite(value))
                ax.plot(counts, y, label=label, color=color, linestyle=linestyle, marker=marker,
                        markersize=size, markerfacecolor=face, markeredgewidth=1.3, linewidth=width)
            padding = max(0.5, (max(available) - min(available)) * 0.12)
            ax.set_ylim(max(0, min(available) - padding), min(100, max(available) + padding))
            ax.set_xlim(0.7, data.max_views + 0.3)
            ax.set_xticks(counts, [f"{count}\n+{perspectives[count - 1].label}" for count in counts])
            ax.tick_params(axis="x", labelbottom=True, labelsize=9)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
            ax.yaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
            ax.set(title=metric, xlabel="Number of perspectives (K)", ylabel="Recall (%)")
            ax.grid(axis="y", color="#DDDDDD", linewidth=0.7, alpha=0.8)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.929),
                   ncol=min(5, len(handles)), frameon=False, fontsize=10, columnspacing=1.8)
        for index, note in enumerate(notes):
            fig.text(0.5, 0.875 - index * 0.017, note, ha="center", fontsize=9, color="#555555")
        heading = bottom - 0.082
        fig.text(0.075, heading, "Perspective order: Top-K uses the first K entries; +letter marks the newly added view.",
                 fontsize=10, weight="semibold")
        for index, perspective in enumerate(perspectives):
            column, row = divmod(index, footer_rows)
            fig.text(0.075 + column * 0.47, heading - 0.035 - row * (0.25 / height),
                     textwrap.fill(f"{perspective.rank:>2}. {perspective.label}  {perspective.name}", width=65),
                     fontsize=9.5, va="top", linespacing=1.2)
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent,
                        help="Output directory (default: the directory containing this script)")
    args = parser.parse_args(argv)
    try:
        data = build_plot_data()
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig = build_figure(data)
        try:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            paths = [args.output_dir / f"adaptive_topk_recall_cuhk_pedes.{extension}" for extension in ("png", "pdf")]
            for path in paths:
                fig.savefig(path, dpi=300, facecolor="white")
        finally:
            plt.close(fig)
    except (OSError, ValueError, ImportError) as exc:
        parser.error(str(exc))
    print(f"Plotted {len(data.curves)} stage/method series, K=1..{data.max_views}; Recall is shown in percent.")
    print("Perspective order: " + " -> ".join(item.label for item in data.perspectives[:data.max_views]))
    for note in overlap_notes(data):
        print(note)
    for path in paths:
        print(f"Saved: {path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
