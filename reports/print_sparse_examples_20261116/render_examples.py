"""Render the captured server evidence; no model inference or server writes."""
import json
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


ROOT = Path(__file__).resolve().parent
EVIDENCE = json.loads((ROOT / "evidence.json").read_text(encoding="utf-8"))
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "savefig.facecolor": "white"})
VIEW_NAMES = {"J": "semantics", "A": "scene", "C": "people", "B": "environment",
              "D": "clothing", "I": "spatial layout"}


def normalized(token):
    return token.lstrip(" \u2581\u0120\u0121").lower()


def find_matches(record):
    """Require positive text weights and absence across all baseline image variants."""
    original = {normalized(t): w for t, w in record["original"]["vector"].items() if w > 0}
    summary = {normalized(t): w for t, w in record["summary"]["vector"].items() if w > 0}
    text_top10 = {normalized(t["token"]): t for t in record["text_log"]["top_tokens"]}
    baseline = {normalized(t["token"]) for t in record["single_image"]["top_tokens"]}
    matches = []
    for row in record["adaptive_image"]:
        assert len(row["top_tokens"]) == 10
        for rank, token in enumerate(row["top_tokens"], 1):
            term = normalized(token["token"])
            if term not in original or term not in summary or term not in text_top10 or term in baseline:
                continue
            matches.append(dict(word=term, view=row["view"], rank=rank, token=token["token"],
                token_id=token["token_id"], image_weight=token["weight"],
                retained_text_token=text_top10[term], original_text_sparse_weight=original[term],
                summary_text_sparse_weight=summary[term],
                image_source=dict(file=row["_file"], line=row["_line"])))
    assert matches
    return matches


def render(record, matches):
    caption = record["caption"]
    highlights = {match["word"] for match in matches}
    fig = plt.figure(figsize=(20, 6.4), dpi=150, facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    columns = [0.012, 0.255, 0.322, 0.384, 0.514, 0.992]
    header_top, header_bottom, row_height = 0.955, 0.88, 0.077
    bottom = header_bottom - 8 * row_height
    for index, heading in enumerate(("Image & Text", "Method", "Modality", "Perspective", "Top-10 Tokens")):
        fig.text((columns[index] + columns[index + 1]) / 2, 0.916, heading,
                 ha="center", va="center", fontsize=14, weight="bold")
    for y in (header_top, header_bottom, bottom):
        ax.plot([columns[0], columns[-1]], [y, y], color="#252525", lw=1.0)
    for x in columns[1:-1]:
        ax.plot([x, x], [bottom, header_top], color="#777777", lw=0.6)
    for index in range(1, 8):
        y = header_bottom - index * row_height
        left = columns[1] if index <= 2 else columns[3]
        ax.plot([left, columns[-1]], [y, y], color="#777777", lw=0.55)

    image_axes = fig.add_axes([0.025, 0.445, 0.216, 0.405])
    with Image.open(ROOT / record["local_image"]) as image:
        image_axes.imshow(image)
    image_axes.axis("off")
    fig.text(0.133, 0.398, textwrap.fill(caption["caption"], width=38),
             ha="center", va="top", fontsize=12.4, linespacing=1.38)

    rows = [("SSR*", "text", "coarse", record["text_log"]["top_tokens"]),
            ("PTT", "image", "coarse", record["single_image"]["top_tokens"])]
    rows += [("MPP", "image", row["view"] + ": " + VIEW_NAMES[row["view"]], row["top_tokens"])
             for row in record["adaptive_image"]]
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for index, (method, modality, perspective, tokens) in enumerate(rows):
        y = header_bottom - (index + 0.5) * row_height
        if index < 2:
            fig.text((columns[1] + columns[2]) / 2, y, method, ha="center", va="center", fontsize=13)
            fig.text((columns[2] + columns[3]) / 2, y, modality, ha="center", va="center", fontsize=13)
        fig.text((columns[3] + columns[4]) / 2, y, perspective, ha="center", va="center", fontsize=12)
        x = columns[4] + 0.007
        for position, token in enumerate(tokens):
            selected = normalized(token["token"]) in highlights
            artist = fig.text(x, y, token["token"], va="center", fontsize=12.2,
                              weight="bold" if selected else "normal", color="#123c60" if selected else "#111111",
                              bbox=dict(facecolor="#dcecf9", edgecolor="none", pad=1.5) if selected else None)
            x += artist.get_window_extent(renderer).width / fig.bbox.width
            if position != len(tokens) - 1:
                comma = fig.text(x, y, ", ", va="center", fontsize=12.2)
                x += comma.get_window_extent(renderer).width / fig.bbox.width
        assert x < columns[-1], (caption["imgid"], perspective, "Token row overflows")
    middle = header_bottom - 5 * row_height
    fig.text((columns[1] + columns[2]) / 2, middle, "MPP", ha="center", va="center", fontsize=13)
    fig.text((columns[2] + columns[3]) / 2, middle, "image", ha="center", va="center", fontsize=13)

    fig.text(0.016, 0.194, "Flickr test | LLaVA-Mistral-7B, base | "
             f"image_id={caption['imgid']} | sentid={caption['sentid']} | {caption['filename']}", fontsize=10.4)
    fig.text(0.016, 0.143, "Blue: present in retained text Top-10, absent from the single-summary image Top-10, "
             "and recovered by an adaptive image view.", fontsize=10.4, color="#123c60")
    fig.text(0.016, 0.096, "SSR*: original-text tokens retained with sparse_manual=false. PTT: single summary. "
             "MPP: adaptive views. Raw token spelling is preserved; matching ignores case and word-prefix markers.", fontsize=9.3)
    fig.text(0.016, 0.052, "Image baseline: saved rank-0 checkpoint. Positive text weights were also checked in both "
             "saved text sparse representations. Source paths, lines, IDs and weights: evidence.json / matches.json.", fontsize=9.3)
    path = ROOT / f"example_{caption['imgid']}.png"
    fig.savefig(path, dpi=160, bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)
    return path


if __name__ == "__main__":
    result = []
    for record in EVIDENCE["records"]:
        matches = find_matches(record)
        path = render(record, matches)
        result.append(dict(**record["caption"], figure=path.name, matches=matches,
            text_source=dict(file=record["text_log"]["_file"], line=record["text_log"]["_line"]),
            baseline_source=dict(file=record["single_image"]["_file"], line=record["single_image"]["_line"])))
        print(path.name, ":", ", ".join(sorted({match["word"] for match in matches})))
    (ROOT / "matches.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
