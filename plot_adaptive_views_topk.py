"""Plot Recall@1/5/10/20 against the number of adaptive perspectives.

Example (run from the project root):
    python plot_adaptive_views_topk.py

The supplied attachments are transcribed into the lists below. No input files
are read at runtime. Recall values remain fractions and are plotted as percent.
All ranked perspectives are retained; only reported Top-K results are plotted.
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

# Transcribed from the supplied view_ranking (1).csv, in increasing rank order.
# Top-K includes the first K perspectives in this list.
VIEW_COUNTS = [1, 2, 3, 4, 5, 6, 7, 8, 9]
# Only K=1..9 metrics were supplied; retain all ten candidates from the ranking.
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

# Recall fractions transcribed from the supplied CUHK-PEDES topk_summary (1).csv.
# Each list follows VIEW_COUNTS; the plotting code converts these to percent.
RECALL_SERIES = {
    ("stage1", "sparse_concat"): {
        "R@1": [
            0.024691358024691357, 0.036712150747238464, 0.06692657569850552, 0.06254061078622482,
            0.06887589343729694, 0.08934372969460688, 0.08170890188434048, 0.07943469785575048,
            0.07602339181286549,
        ],
        "R@5": [
            0.06286549707602339, 0.08609486679662118, 0.15350877192982457, 0.15350877192982457,
            0.16780376868096167, 0.203866146848603, 0.19135802469135801, 0.1817738791423002,
            0.17966211825860948,
        ],
        "R@10": [
            0.09405458089668615, 0.12735542560103963, 0.21929824561403508, 0.21864847303443793,
            0.2334307992202729, 0.2751786874593892, 0.25925925925925924, 0.2519493177387914,
            0.24788823911630928,
        ],
        "R@20": [
            0.1366146848602989, 0.1825860948667966, 0.2998700454840806, 0.2982456140350877,
            0.31351526965562054, 0.36289798570500326, 0.3448667966211826, 0.3347953216374269,
            0.32423651721897334,
        ],
    },
    ("stage2", "dense"): {
        "R@1": [
            0.11809616634178037, 0.12833008447043534, 0.14230019493177387, 0.1442495126705653,
            0.14912280701754385, 0.14782326185834957, 0.14912280701754385, 0.1484730344379467,
            0.149772579597141,
        ],
        "R@5": [
            0.22498375568551007, 0.2498375568551007, 0.2883365821962313, 0.2894736842105263,
            0.29272254710851203, 0.3014944769330734, 0.297270955165692, 0.2964587394411956,
            0.2936972059779077,
        ],
        "R@10": [
            0.27810266406757633, 0.306367771280052, 0.3619233268356075, 0.362735542560104,
            0.3651721897335932, 0.3775178687459389, 0.3729694606887589, 0.37166991552956463,
            0.36955815464587394,
        ],
        "R@20": [
            0.3341455490578298, 0.3732943469785575, 0.44866796621182586, 0.449317738791423,
            0.453866146848603, 0.4766081871345029, 0.46751137102014295, 0.46182586094866795,
            0.46068875893437294,
        ],
    },
    ("stage2", "sparse"): {
        "R@1": [
            0.024691358024691357, 0.036712150747238464, 0.06692657569850552, 0.06254061078622482,
            0.06887589343729694, 0.08934372969460688, 0.08170890188434048, 0.07943469785575048,
            0.07602339181286549,
        ],
        "R@5": [
            0.06286549707602339, 0.08609486679662118, 0.15350877192982457, 0.15350877192982457,
            0.16780376868096167, 0.203866146848603, 0.19135802469135801, 0.1817738791423002,
            0.17966211825860948,
        ],
        "R@10": [
            0.09405458089668615, 0.12735542560103963, 0.21929824561403508, 0.21864847303443793,
            0.2334307992202729, 0.2751786874593892, 0.25925925925925924, 0.2519493177387914,
            0.24788823911630928,
        ],
        "R@20": [
            0.1366146848602989, 0.1825860948667966, 0.2998700454840806, 0.2982456140350877,
            0.31351526965562054, 0.36289798570500326, 0.3448667966211826, 0.3347953216374269,
            0.32423651721897334,
        ],
    },
    ("stage2", "hybrid"): {
        "R@1": [
            0.04922027290448343, 0.05847953216374269, 0.09876543209876543, 0.09616634178037686,
            0.10201429499675113, 0.12037037037037036, 0.11793372319688109, 0.1133853151397011,
            0.11143599740090968,
        ],
        "R@5": [
            0.1189083820662768, 0.14067576348278102, 0.20873944119558155, 0.20727745289148797,
            0.21881091617933723, 0.25259909031838856, 0.2444769330734243, 0.2355425601039636,
            0.23521767381416503,
        ],
        "R@10": [
            0.16682910981156596, 0.19655620532813514, 0.2826510721247563, 0.28460038986354774,
            0.29678362573099415, 0.336582196231319, 0.3198505523066927, 0.31562703053931124,
            0.31513970110461337,
        ],
        "R@20": [
            0.23440545808966862, 0.27176738141650425, 0.37638076673164395, 0.38369070825211177,
            0.38628979857050033, 0.43193632228719947, 0.4171539961013645, 0.4083820662768031,
            0.40334632878492527,
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
    if len(VIEW_ORDER) < len(VIEW_COUNTS):
        raise ValueError("VIEW_ORDER must cover every VIEW_COUNTS entry")
    perspectives = tuple(Perspective(rank, label, aspect)
                         for rank, (label, aspect) in enumerate(VIEW_ORDER, 1))
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


def plot_notes(data):
    notes = overlap_notes(data)
    if data.max_views < len(data.perspectives):
        notes.append(f"Results available for K=1..{data.max_views}; higher-K results were not supplied.")
    return notes


def build_figure(data):
    """Return a Matplotlib Figure; coordinates use Recall percentages."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, StrMethodFormatter

    counts = list(range(1, data.max_views + 1))
    perspectives = data.perspectives
    footer_rows = math.ceil(len(perspectives) / 2)
    notes = plot_notes(data)
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
        query_details = f"{int(meta['query_count']):,} queries"
        dataset_label = {"webqa": "WebQA", "llava": "LLaVA", "flickr": "Flickr"}.get(
            meta["dataset"], meta["dataset"])
        fig.suptitle(f"{dataset_label} | {meta['model_family']} | {direction}", y=0.985, fontsize=16, weight="semibold")
        fig.text(0.5, 0.947, f"{meta['split']} / {meta['category']} | {meta['execution_mode'].replace('_', ' ')}"
                 f" | alpha = {meta['hybrid_alpha']} | {query_details}",
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
            unreported = perspective.rank > data.max_views
            suffix = f" (no Top-{perspective.rank} result)" if unreported else ""
            fig.text(0.075 + column * 0.47, heading - 0.035 - row * (0.25 / height),
                     textwrap.fill(f"{perspective.rank:>2}. {perspective.label}  {perspective.name}{suffix}", width=65),
                     fontsize=9.5, va="top", linespacing=1.2, color="#777777" if unreported else "black")
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
            paths = [args.output_dir / f"adaptive_topk_recall.{extension}" for extension in ("png", "pdf")]
            for path in paths:
                fig.savefig(path, dpi=300, facecolor="white")
        finally:
            plt.close(fig)
    except (OSError, ValueError, ImportError) as exc:
        parser.error(str(exc))
    print(f"Plotted {len(data.curves)} stage/method series, K=1..{data.max_views}; Recall is shown in percent.")
    print("Perspective order: " + " -> ".join(item.label for item in data.perspectives))
    for note in plot_notes(data):
        print(note)
    for path in paths:
        print(f"Saved: {path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
