import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = ["#f57c6e", "#f2b56f"]
X_LABELS = [
    "SSR(T)\nPTT(I)",
    "PTT(T)\nPTT(I)",
    "MPP(SSR(T)\nPTT(I))",
    "MPP(PTT(T)\nPTT(I))",
]


def plot_compare(data, dense_score, output_path):
    df = pd.DataFrame(data)

    n_groups = len(df)
    n_datasets = len(df.columns) - 1
    bar_width = 0.055
    index = np.arange(n_datasets) * (n_groups * bar_width + bar_width)

    fig, ax = plt.subplots(figsize=(10.6, 3.4), constrained_layout=True)
    ax.axhline(y=dense_score, color="b", linestyle="--", linewidth=1.6, label="Dense")
    ax.annotate(
        f"{dense_score:.1f}",
        xy=(0.01, dense_score),
        xycoords=("axes fraction", "data"),
        xytext=(3, 3),
        textcoords="offset points",
        va="bottom",
        ha="left",
        fontsize=15,
    )

    for i, condition in enumerate(df["Condition"]):
        offsets = index + (i + 0.5) * bar_width
        bars = ax.bar(
            offsets,
            df.iloc[i, 1:],
            bar_width,
            label=condition,
            color=COLORS[i],
            edgecolor="black",
            linewidth=0.8,
        )
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=15,
                clip_on=False,
            )

    max_score = max(df.iloc[:, 1:].to_numpy().max(), dense_score)
    ax.set_ylim(0, max_score + 20)
    ax.set_ylabel("r@1", fontsize=21, labelpad=4)
    ax.set_xticks(index + n_groups * bar_width / 2)
    ax.set_xticklabels(X_LABELS, fontsize=15)
    ax.tick_params(axis="y", labelsize=16)
    ax.tick_params(axis="x", pad=2)
    ax.margins(x=0.05)
    ax.legend(
        loc="upper center",
        ncol=3,
        fontsize=16,
        bbox_to_anchor=(0.5, 0.99),
        frameon=False,
        handlelength=1.7,
        handletextpad=0.5,
        columnspacing=1.0,
    )

    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def main():
    plot_compare(
        {
            "Condition": ["Sparse", "Hybrid"],
            "SSE(T) PTT(I) r@1": [25.0, 68.4],
            "PTT(T) PTT(I) r@1": [54.0, 66.0],
            "MPP(SSE(T) PTT(I)) r@1": [31.2, 69.4],
            "MPP(PTT(T) PTT(I)) r@1": [62.7, 70.4],
        },
        dense_score=66.2,
        output_path="compare_flickr.pdf",
    )

    plot_compare(
        {
            "Condition": ["Sparse", "Hybrid"],
            "SSE(T) PTT(I) r@1": [15.3, 73.7],
            "PTT(T) PTT(I) r@1": [59.4, 71.9],
            "MPP(SSE(T) PTT(I)) r@1": [28.2, 73.6],
            "MPP(PTT(T) PTT(I)) r@1": [68.4, 75.4],
        },
        dense_score=72.3,
        output_path="compare_i2t_flickr.pdf",
    )


if __name__ == "__main__":
    main()
