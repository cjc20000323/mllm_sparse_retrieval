"""Training-data / average-recall motivation scatter plot (no model loading).

运行方式（与其他绘图脚本一样，文件位于项目根目录）：
    python plot_motivation.py --demo       # 仅预览：带醒目的示意数据标记
    python plot_motivation.py              # 导出截图中 11 个大模型配置的真实结果

填写 MODEL_RESULTS，一行对应一个固定模型配置：
    name           图上的显示名称；同一模型多个版本请复制一行并注明版本。
    display_method 可选简写，仅用于图上标签；不改变原始 method / 配置名称。
    train_pairs_m  保留历史字段名以兼容调用；现表示额外参数训练样本量（百万），
                   含纯文本/多模态样本，按训练阶段累计，不累计训练轮数。
                   20 万样本填 0.2；没有额外参数训练填 0，不知道则填 None。
    avg_score_pct  用户截图 Average 列的平均召回率，直接填百分数。
                   原始平均值为 0.8583 时填 85.83，不自动转换或重新平均。
    params_b       语言主干的标称参数量（十亿），统一采用截图中的 7B/8B。
    backbone       四种已知主干分别用圆形、正方形、三角形、菱形表示。
                   缺省或未知主干使用圆形，不添加对应的 Backbone 图例。
    supervision    "zero_shot"：没有使用所选评测任务相关的监督信息；
                   "benchmark_supervised"：使用了，包括验证反馈优化提示词。
    is_ours        自有方法填 True，气泡及名称会标红。
    label_offset   名称相对气泡的偏移，单位为 points，例如 (12, 0)。
    label_connectionstyle 可选引线路由；默认直线，拥挤位置可使用 angle 折线。
    fill_alpha     可选填充不透明度，范围 [0, 1]；未设置或 None 时沿用分组默认值。
                   仅调整气泡填充，不改变分组、图例、边框或文字。

重要口径：
* 所有模型必须使用相同的任务集合、划分、检索方向、召回率 K 值和平均方式；
  本程序不挑选指标或计算平均值。Y_LABEL 用于坐标轴，BENCHMARK_NOTE 保存口径元数据，
  不在成图底部绘制说明文字。
* 0 只表示不额外更新检索模型参数，不表示没有预训练或没有用过验证数据。
  DSPy 使用相关验证反馈时应填 benchmark_supervised，不能标为完全零样本。
* params_b 不是整个多模态系统的参数总量，不累计视觉编码器、离线教师模型或
  检索/重排适配器；训练样本累计量也不表示去重后的独立样本量。
* 成绩只来自用户提供的截图，不混入网页中的其他评测成绩；训练量来源见下文。
  Re-M 的零额外参数训练、无评测任务相关监督由用户确认，不从项目代码推断。
* 未核实的数据仍须保留 None；不完整记录不会被画成 0 分。

默认按单栏论文图排版：标签仅显示模型名，不显示参数量、成绩或 rerank 标记。
标记统一为 80 points²，不用大小编码参数量；完整主干名显示在下方图外图例中。
成绩仍用于纵坐标。插入约 8.5cm 宽时，标签约 8pt；不移动真实坐标。
仅依赖 matplotlib，使用 Agg 后端导出矢量 PDF 和 300 DPI PNG，不调用 API/GPU。
"""

from __future__ import annotations

import argparse
import math
import warnings
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
from matplotlib.ticker import AutoMinorLocator


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_STEM = "motivation_training_data"
X_LABEL = "Training data (million pairs)"
Y_LABEL = "Average Recall"
# 保留口径和训练量说明供溯源，不在成图底部显示。
BENCHMARK_NOTE = "Average recall: screenshot's Average column across 8 datasets (R@1/5/10)."
TRAINING_NOTE = (
    "Training examples are summed over stages, not deduplicated; base-model pretraining is excluded.\n"
    "Finetuned E5-V: ~0.273M text examples. LamRA-Rank: ~1.4M + 1.1M.\n"
    "MMRet S1: ~26.2M (MegaPairs only). VLM2Vec-LLaVA-Next: ~1.03M estimated (100K/subset cap).\n"
    "Marker shape: backbone; uniform marker area. Re-M: no task-related supervision (user-confirmed)."
)
DEMO_NOTE = "Illustrative average recall and training volumes; not experimental results."
SCORE_SOURCE = "User-provided result-table screenshot, Average column; no external benchmark substitution."
TRAINING_SOURCES = {
    "e5v": "https://arxiv.org/html/2407.12580",
    "gme": "https://arxiv.org/html/2412.16855",
    "lamra": "https://arxiv.org/html/2412.01720",
    # Table 1 explicitly reports LamRA's retrieval + reranking data as 1.4 + 1.1M.
    "freeret": "https://arxiv.org/html/2509.24621v2",
    "lamra_qwen25": "https://github.com/Code-kunkun/LamRA/tree/qwen2.5vl",
    "mmret": "https://arxiv.org/html/2412.14475",
    "mmret_s1": "https://huggingface.co/BAAI/BGE-VL-MLLM-S1",
    "vlm2vec": "https://arxiv.org/html/2410.05160",
    "vlm2vec_llava_next": "https://huggingface.co/TIGER-Lab/VLM2Vec-LLaVa-Next",
}

# VLM2Vec Table 1: rounded training-set sizes (K), in table order, for 20 sets.
# The LLaVA-Next model card specifies a 100K/subset cap (not the 50K cap used
# by some other variants). The resulting 1.032M is an estimate, not a measured
# count for the released checkpoint; show it as ~1.03M, not exact precision.
VLM2VEC_TRAIN_SUBSET_SIZES_K = (
    100, 49, 8, 8, 20,  # classification
    9, 17, 40, 24, 28, 70,  # VQA
    123, 26, 100, 100, 100, 113, 16, 17,  # retrieval
    100,  # grounding
)
VLM2VEC_MAX_SUBSET_K = 100
VLM2VEC_ESTIMATED_TRAIN_M = round(
    sum(min(size, VLM2VEC_MAX_SUBSET_K) for size in VLM2VEC_TRAIN_SUBSET_SIZES_K) / 1000, 2,
)

COLORS = ["#71b7ed", "#f2b56f", "#84c3b7", "#fae69e", "#88d8db", "#f2a7da", "#b8aeeb"]
OURS_COLOR = "#f57c6e"
FIGURE_SIZE_INCHES = (3.6, 2.5)
MAIN_PANEL_HEIGHT_INCHES = 2.1
LEGEND_FOOTER_INCHES = 0.4
LABEL_FONT_SIZE = 8.5
LEGEND_FONT_SIZE = 8.5
DEMO_BANNER = "ILLUSTRATIVE DATA\nNOT EXPERIMENTAL RESULTS"
MARKER_AREA_POINTS2 = 80.0  # scatter 的 s 为面积（points²），不是直径。
BUBBLE_EDGE_WIDTH = 0.65
BACKBONE_MARKERS = {
    "LLaVA-Mistral-7B": "o",
    "LLaVA-LLaMA-8B": "s",
    "Qwen2-VL-7B": "^",
    "Qwen2.5-VL-7B": "D",
}
SYMLOG_LINTHRESH = 0.1  # 零附近使用线性区间；真实的 0 不替换为 epsilon。
SUPERVISION = {
    "zero_shot": (0.95, "Without Training"),
    "benchmark_supervised": (0.15, "With Training"),
}
PLOT_STYLE = {
    "font.family": "DejaVu Sans", "font.size": 8.5,
    "axes.labelsize": 10.5, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "pdf.fonttype": 42, "ps.fonttype": 42,
}


# ===================== 截图中的 11 行真实结果 =====================
# name 唯一标识配置；标签显示 method，参数量、backbone 与成绩保留供溯源。
# 训练量是文献报告的约数，不是 image-text pair 数、训练步数或 epoch 数。
# LamRA-Rank 的 2.5M = 检索阶段约 1.4M + 重排阶段约 1.1M，同源数据可复用。
# GME/LamRA 使用包含 COCO 等相关任务的训练数据，按“有相关监督”标注；
# Finetuned E5-V 仅用无关 NLI 文本数据，不能因“有参数训练”就标成有基准监督。
# MMRet 使用用户确认的 S1：仅 MegaPairs，不累计 S2 的 MMEB 微调。
# VLM2Vec 使用 LLaVA-Next checkpoint；MMEB 含 COCO/WebQA，属于有相关监督。
MODEL_RESULTS = [
    dict(name="E5-V (LLaVA-LLaMA-8B)", method="E5-V", backbone="LLaVA-LLaMA-8B",
         train_pairs_m=0, avg_score_pct=43.6, params_b=8, training_stages_m=(),
         supervision="zero_shot", color=COLORS[0], label_offset=(14, 10),
         training_source=TRAINING_SOURCES["e5v"]),
    dict(name="E5-V (LLaVA-Mistral-7B)", method="E5-V", backbone="LLaVA-Mistral-7B",
         train_pairs_m=0, avg_score_pct=42.1, params_b=7, training_stages_m=(),
         supervision="zero_shot", color=COLORS[0], label_offset=(14, -10),
         training_source=TRAINING_SOURCES["e5v"]),
    dict(name="FreeRet (LLaVA-Mistral-7B)", method="FreeRet (+ rerank)", backbone="LLaVA-Mistral-7B",
         train_pairs_m=0, avg_score_pct=60.1, params_b=7, training_stages_m=(),
         supervision="zero_shot", color=COLORS[2], label_offset=(11, 0),
         training_source=TRAINING_SOURCES["freeret"]),
    dict(name="Finetuned E5-V (LLaVA-LLaMA-8B)", method="Finetuned E5-V", backbone="LLaVA-LLaMA-8B",
         train_pairs_m=0.273, avg_score_pct=65.4, params_b=8, training_stages_m=(0.273,),
         supervision="zero_shot", color=COLORS[1], label_offset=(-1, 26),
         fill_alpha=0.15,
         label_connectionstyle="angle,angleA=0,angleB=90,rad=2",
         training_source=TRAINING_SOURCES["e5v"]),
    dict(name="MMRet S1 (LLaVA-Mistral-7B)", method="MMRet S1", backbone="LLaVA-Mistral-7B",
         display_method="MMRet",
         train_pairs_m=26.2, avg_score_pct=69.4, params_b=7, training_stages_m=(26.2,),
         supervision="zero_shot", color="#a8dca2", label_offset=(-42, 0),
         fill_alpha=0.15,
         checkpoint="BAAI/BGE-VL-MLLM-S1", training_examples_reported=26_235_105,
         training_source=TRAINING_SOURCES["mmret"],
         checkpoint_source=TRAINING_SOURCES["mmret_s1"],
         training_note="S1: MegaPairs only; no S2/MMEB stage; rounded from 26,235,105 pairs."),
    dict(name="VLM2Vec (LLaVA-Mistral-7B)", method="VLM2Vec", backbone="LLaVA-Mistral-7B",
         train_pairs_m=VLM2VEC_ESTIMATED_TRAIN_M, avg_score_pct=60.0, params_b=7,
         training_stages_m=(VLM2VEC_ESTIMATED_TRAIN_M,),
         supervision="benchmark_supervised", color=COLORS[4], label_offset=(19, -18),
         checkpoint="TIGER-Lab/VLM2Vec-LLaVa-Next", training_estimated=True,
         training_source=TRAINING_SOURCES["vlm2vec"],
         checkpoint_source=TRAINING_SOURCES["vlm2vec_llava_next"],
         training_note="Estimated from rounded Table 1 sizes for 20 MMEB training subsets, "
                       "capped at 100K each: ~1.032M, plotted as ~1.03M; includes COCO/WebQA."),
    dict(name="GME (Qwen2-VL-7B)", method="GME", backbone="Qwen2-VL-7B",
         train_pairs_m=8, avg_score_pct=70.9, params_b=7, training_stages_m=(8,),
         supervision="benchmark_supervised", color="#b58c81", label_offset=(-12, 10),
         fill_alpha=0.08,
         training_source=TRAINING_SOURCES["gme"]),
    dict(name="LamRA-Ret (Qwen2.5-VL-7B)", method="LamRA-Ret", backbone="Qwen2.5-VL-7B",
         train_pairs_m=1.4, avg_score_pct=55.3, params_b=7, training_stages_m=(1.4,),
         supervision="benchmark_supervised", color=COLORS[6], label_offset=(-14, -17),
         training_source=TRAINING_SOURCES["freeret"]),
    dict(name="LamRA-Rank20 (Qwen2.5-VL-7B)", method=r"LamRA-Rank$_{20}$ (+ rerank)", backbone="Qwen2.5-VL-7B",
         display_method="LamRA-Rank",
         train_pairs_m=2.5, avg_score_pct=60.4, params_b=7, training_stages_m=(1.4, 1.1),
         supervision="benchmark_supervised", color=COLORS[6], label_offset=(0, 14),
         training_source=TRAINING_SOURCES["freeret"]),
    dict(name="Re-M (LLaVA-LLaMA-8B)", method="Re-M", backbone="LLaVA-LLaMA-8B",
         train_pairs_m=0, avg_score_pct=59.0, params_b=8, training_stages_m=(),
         supervision="zero_shot", is_ours=True, color=OURS_COLOR, label_offset=(14, -17),
         training_source="User-confirmed: prompt optimization only; no task-related supervision."),
    dict(name="Re-M (LLaVA-Mistral-7B)", method="Re-M", backbone="LLaVA-Mistral-7B",
         train_pairs_m=0, avg_score_pct=64.8, params_b=7, training_stages_m=(),
         supervision="zero_shot", is_ours=True, color=OURS_COLOR, label_offset=(14, 15),
         training_source="User-confirmed: prompt optimization only; no task-related supervision."),
]


# 完全虚构的数据，只用于测试绘图布局，不对应上面任何真实模型的成绩。
DEMO_RESULTS = [
    dict(name="Demo-A", train_pairs_m=0, avg_score_pct=40, params_b=7,
         supervision="zero_shot", color=COLORS[0], label_offset=(0, -4)),
    dict(name="Demo-B", train_pairs_m=0.3, avg_score_pct=43, params_b=7,
         supervision="zero_shot", color=COLORS[1], label_offset=(-4, 2)),
    dict(name="Demo-C", train_pairs_m=0.7, avg_score_pct=68, params_b=7,
         supervision="benchmark_supervised", color=COLORS[3], label_offset=(-10, 21)),
    dict(name="Demo-D", train_pairs_m=1.2, avg_score_pct=64, params_b=7,
         supervision="benchmark_supervised", color=COLORS[5], label_offset=(8, -8)),
    dict(name="Demo-E", train_pairs_m=1.6, avg_score_pct=52, params_b=7,
         supervision="zero_shot", color=COLORS[6], label_offset=(-12, 6)),
    dict(name="Demo-F (7B)", train_pairs_m=8, avg_score_pct=57, params_b=7,
         supervision="zero_shot", color="#b58c81", label_offset=(0, -4)),
    dict(name="Demo-F (2B)", train_pairs_m=8, avg_score_pct=49, params_b=2,
         supervision="zero_shot", color="#b58c81", label_offset=(-12, 0)),
    dict(name="Demo-G", train_pairs_m=24, avg_score_pct=65, params_b=7,
         supervision="benchmark_supervised", color="#a8dca2", label_offset=(0, -4)),
    dict(name="Ours (demo)", train_pairs_m=0, avg_score_pct=72, params_b=7,
         supervision="benchmark_supervised", is_ours=True, label_offset=(14, 0)),
    dict(name="Ours, small (demo)", train_pairs_m=0, avg_score_pct=60, params_b=2,
         supervision="benchmark_supervised", is_ours=True, label_offset=(14, 0)),
]


def _number(value: Any, field: str, name: str) -> float:
    try:
        number = float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"{name}: {field} must be a finite number.") from exc
    if isinstance(value, bool) or not math.isfinite(number):
        raise ValueError(f"{name}: {field} must be a finite number.")
    return number


def prepare_points(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Validate provided values; unknown values stay unknown, never become zero."""
    points = []
    skipped = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"Record {index + 1}: expected a model dictionary.")
        name = row.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"Record {index + 1}: name must be a non-empty string.")
        point = dict(row)
        for field in ("train_pairs_m", "avg_score_pct", "params_b"):
            if row.get(field) is not None:
                point[field] = _number(row[field], field, name)
        if point.get("train_pairs_m") is not None and point["train_pairs_m"] < 0:
            raise ValueError(f"{name}: train_pairs_m must be >= 0 (million training examples).")
        if point.get("params_b") is not None and point["params_b"] <= 0:
            raise ValueError(f"{name}: params_b must be > 0 (billions).")
        if point.get("avg_score_pct") is not None and not 0 <= point["avg_score_pct"] <= 100:
            raise ValueError(f"{name}: avg_score_pct must be in [0, 100], not a fraction converted automatically.")
        if row.get("supervision") is not None and (
            not isinstance(row["supervision"], str) or row["supervision"] not in SUPERVISION
        ):
            raise ValueError(f"{name}: supervision must be one of {tuple(SUPERVISION)}.")
        missing = [field for field in ("train_pairs_m", "avg_score_pct", "params_b", "supervision")
                   if point.get(field) is None]
        if missing:
            skipped.append(f"{name} ({', '.join(missing)})")
            continue
        ours = row.get("is_ours", False)
        if not isinstance(ours, bool):
            raise ValueError(f"{name}: is_ours must be True or False.")
        color = OURS_COLOR if ours else row.get("color", COLORS[index % len(COLORS)])
        if not mcolors.is_color_like(color):
            raise ValueError(f"{name}: invalid color {color!r}.")
        fill_alpha = row.get("fill_alpha")
        if fill_alpha is None:
            fill_alpha = SUPERVISION[point["supervision"]][0]
        else:
            fill_alpha = _number(fill_alpha, "fill_alpha", name)
            if not 0 <= fill_alpha <= 1:
                raise ValueError(f"{name}: fill_alpha must be in [0, 1].")
        offset = row.get("label_offset", (12, 0) if ours else (0, -4))
        if not isinstance(offset, (tuple, list)) or len(offset) != 2:
            raise ValueError(f"{name}: label_offset must contain two numbers in points.")
        point.update(is_ours=ours, color=color, fill_alpha=fill_alpha,
                     label_offset=tuple(_number(value, "label_offset", name) for value in offset))
        points.append(point)
    if not points:
        raise ValueError(
            "No complete model records. Fill MODEL_RESULTS: train_pairs_m, avg_score_pct, "
            "params_b and supervision. Unknown values should remain None. "
            "Use --demo for an explicitly illustrative preview."
        )
    if skipped:
        warnings.warn("Skipped incomplete model records: " + "; ".join(skipped), UserWarning, stacklevel=2)
    return points


def bubble_area(params_b: float) -> float:
    """Keep the legacy call signature; marker area is independent of parameter counts."""
    return MARKER_AREA_POINTS2


def marker_radius(marker: str) -> float:
    """Conservative radius in points, including square/triangle/diamond corners."""
    style = MarkerStyle(marker)
    path = style.get_path().transformed(style.get_transform())
    return math.sqrt(MARKER_AREA_POINTS2) * max(math.hypot(x, y) for x, y in path.vertices)


def plot_motivation(rows: Iterable[Mapping[str, Any]], *, demo: bool = False):
    points = prepare_points(rows)  # Fail before allocating/exporting an empty figure.
    with matplotlib.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=FIGURE_SIZE_INCHES)
        # Reserve a separate footer below the compact main panel.
        height = FIGURE_SIZE_INCHES[1]
        fig.subplots_adjust(
            left=0.15, right=0.98,
            bottom=(0.18 * MAIN_PANEL_HEIGHT_INCHES + LEGEND_FOOTER_INCHES) / height,
            top=1 - (0.17 if demo else 0.05) * MAIN_PANEL_HEIGHT_INCHES / height,
        )
        ax.set_xscale("symlog", base=10, linthresh=SYMLOG_LINTHRESH, linscale=0.65)
        max_x = max(point["train_pairs_m"] for point in points)
        right_edge = max(max_x, SYMLOG_LINTHRESH) * 1.9
        ax.set_xlim(-SYMLOG_LINTHRESH * 0.65, right_edge)
        tick_candidates = (0, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100)
        ticks = [tick for tick in tick_candidates if tick <= right_edge]
        if max_x > tick_candidates[-1]:
            ticks.extend(10.0 ** exponent for exponent in range(3, math.ceil(math.log10(right_edge)) + 1)
                         if 10.0 ** exponent <= right_edge)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{tick:g}" for tick in ticks])
        scores = [point["avg_score_pct"] for point in points]
        margin = max(5.0, (max(scores) - min(scores)) * 0.17)
        ax.set_ylim(min(scores) - margin, max(scores) + margin)
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.set_axisbelow(True)
        ax.grid(which="major", color="#c7c7c7", linestyle="--", linewidth=0.65)
        ax.grid(which="minor", axis="y", color="#e7e7e7", linewidth=0.5)
        ax.axvline(0, color="#8b8b8b", linestyle="--", linewidth=1.3, zorder=1)
        for point in points:
            opacity = point["fill_alpha"]
            marker = BACKBONE_MARKERS.get(point.get("backbone"), "o")
            ax.scatter(
                point["train_pairs_m"], point["avg_score_pct"],
                s=bubble_area(point["params_b"]), marker=marker,
                facecolors=[mcolors.to_rgba(point["color"], opacity)],
                edgecolors="#a73832" if point["is_ours"] else "#555555",
                linewidths=BUBBLE_EDGE_WIDTH, zorder=3 if point["is_ours"] else 2,
            )
            dx, dy = point["label_offset"]
            # Offset labels only: never jitter measured scores or zero data.
            label = point["name"]
            if point.get("backbone"):
                method = point.get("display_method", point.get("method", point["name"]))
                label = method.removesuffix(' (+ rerank)')
            ax.annotate(
                label, (point["train_pairs_m"], point["avg_score_pct"]),
                xytext=(dx, dy), textcoords="offset points",
                ha="left" if dx > 0 else "right" if dx < 0 else "center", va="center",
                fontsize=LABEL_FONT_SIZE, color="#c52828" if point["is_ours"] else "black",
                fontweight="semibold" if point["is_ours"] else "normal",
                arrowprops=(dict(arrowstyle="-", color="#8a8a8a", linewidth=0.5,
                                 connectionstyle=point.get("label_connectionstyle", "arc3"),
                                 shrinkA=2, shrinkB=marker_radius(marker) + 1)
                            if point.get("backbone") else None),
                annotation_clip=False, zorder=4,
            )
        ax.set_xlabel(X_LABEL, labelpad=3)
        ax.set_ylabel(Y_LABEL)
        present_backbones = {point.get("backbone") for point in points}
        backbone_handles = [
            Line2D([], [], linestyle="none", marker=marker, markersize=6.5,
                   markerfacecolor="#bbbbbb", markeredgecolor="#555555",
                   markeredgewidth=BUBBLE_EDGE_WIDTH, label=backbone)
            for backbone, marker in BACKBONE_MARKERS.items() if backbone in present_backbones
        ]
        # Matplotlib fills columns first: LLaVA, then Qwen.
        if backbone_handles:
            fig.legend(
                handles=backbone_handles,
                loc="lower center", bbox_to_anchor=(0.5, 0.015), ncol=2,
                frameon=False, fontsize=LEGEND_FONT_SIZE,
                borderaxespad=0, borderpad=0, handlelength=1, handletextpad=0.4,
                columnspacing=0.25, labelspacing=0.3,
            )
        if demo:
            fig.text(0.545, 0.995, DEMO_BANNER,
                     ha="center", va="top", color="#b22222", fontsize=LABEL_FONT_SIZE, fontweight="bold")
        return fig, ax


def save_figure(fig: Any, output_dir: Path = SCRIPT_DIR, *, demo: bool = False) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = OUTPUT_STEM + ("_demo" if demo else "")
    pdf_path, png_path = output_dir / f"{stem}.pdf", output_dir / f"{stem}.png"
    with matplotlib.rc_context(PLOT_STYLE):
        fig.savefig(pdf_path, format="pdf", bbox_inches="tight", pad_inches=0.08)
        fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight", pad_inches=0.08)
    return pdf_path, png_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Plot training-data efficiency using manually supplied average recall percentages.")
    parser.add_argument("--demo", action="store_true", help="Use explicitly synthetic data; output filenames end in _demo.")
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR, help="Output directory (default: alongside this script).")
    args = parser.parse_args(argv)
    try:
        fig, _ = plot_motivation(DEMO_RESULTS if args.demo else MODEL_RESULTS, demo=args.demo)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        paths = save_figure(fig, args.output_dir, demo=args.demo)
    finally:
        plt.close(fig)
    if args.demo:
        print("DEMO ONLY: these synthetic values are not experimental results.")
    for path in paths:
        print(f"Saved: {path.resolve()}")


if __name__ == "__main__":
    main()
