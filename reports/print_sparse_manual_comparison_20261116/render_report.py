"""Validate and render four captured cases without inference or server access."""
import hashlib
import json
import math
from pathlib import Path
import re
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


ROOT = Path(__file__).resolve().parent
EXPECTED = [("3245", "16227", 4), ("7996", "39982", 3),
            ("12949", "64748", 3), ("19643", "98219", 3)]
BLUE, GOLD = "#dcecf9", "#fff0c2"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "savefig.facecolor": "white"})


def canonical(token):
    # Match the actual sparse_lower_or_upper=lower, is_filtered=false representation.
    return token.lower()


def term_entries(log, positive=True):
    result = {}
    for rank, token in enumerate(log["payload"]["top_tokens"], 1):
        if positive and token["weight"] <= 0:
            continue
        result.setdefault(canonical(token["token"]), []).append(dict(rank=rank, **token))
    return result


def validate(evidence):
    assert evidence["criteria"]["comparison"] == "lowercase_only_prefix_preserved"
    assert [(r["caption"]["imgid"], r["caption"]["sentid"], len(r["matches"]))
            for r in evidence["records"]] == EXPECTED
    manifest = evidence["adaptive_manifest"]["payload"]["configuration"]
    assert manifest["model"]["family"] == "llava_mistral"
    assert manifest["encoding"]["sparse_lower_or_upper"] == "lower"
    assert manifest["encoding"]["is_filtered"] is False
    for source in evidence["sources"].values():
        assert source["bytes"] > 0 and len(source["sha256"]) == 64
    for record in evidence["records"]:
        assert hashlib.sha256((ROOT / record["image"]["local_file"]).read_bytes()).hexdigest() == record["image"]["sha256"]
        caption = record["caption"]
        logs = record["logs"]
        for role in ("text_false", "text_true"):
            assert logs[role]["payload"]["sample_id"] == caption["sentid"]
            assert logs[role]["payload"]["sample_text"] == caption["caption"]
        assert logs["text_false"]["payload"]["token_source"] == "retained_text"
        image_logs = [logs["image_true"]] + logs["adaptive_image"]
        assert all(log["payload"]["sample_id"] == caption["imgid"] for log in image_logs)
        assert [log["payload"]["view"] for log in logs["adaptive_image"]] == [v["label"] for v in manifest["selected"]]
        all_logs = [logs["text_false"], logs["text_true"]] + image_logs
        for log in all_logs:
            row = log["payload"]
            assert row["top_k"] == len(row["top_tokens"]) <= 10
            assert row["weight_type"] == "log1p_relu"
            assert log["source"]["file"] in evidence["sources"] and log["source"]["line"] >= 1
            weights = [token["weight"] for token in row["top_tokens"]]
            assert all(math.isfinite(weight) and weight >= 0 for weight in weights)
            assert weights == sorted(weights, reverse=True)
            assert len({t["token_id"] for t in row["top_tokens"]}) == len(row["top_tokens"])
        false, true = term_entries(logs["text_false"]), term_entries(logs["text_true"])
        baseline = term_entries(logs["image_true"], positive=False)
        multi_terms = {term for log in logs["adaptive_image"] for term in term_entries(log)}
        expected_terms = (set(false) & set(true) & multi_terms) - set(baseline)
        expected_terms = {term for term in expected_terms if any(char.isalnum() for char in term)}
        assert expected_terms == set(record["matches"])
        for term, entries in record["matches"].items():
            assert false[term] and true[term] and term not in baseline
            for entry in entries:
                log = next(row for row in logs["adaptive_image"] if row["payload"]["view"] == entry["view"])
                rank = entry["token"]["rank"]
                assert entry["token"] == dict(rank=rank, **log["payload"]["top_tokens"][rank - 1])
                assert canonical(entry["token"]["token"]) == term and entry["token"]["weight"] > 0
                assert entry["source"] == log["source"]
                assert entry["text_false"] == false[term] and entry["text_true"] == true[term]
            for role, row in record["sparse_rows"].items():
                assert row["source"]["file"] in evidence["sources"]
                actual = row["payload"]["vector"].get(term, 0)
                assert actual == record["stored_weights"][term][role]
                if role != "image_true":
                    assert actual > 0
        added = {term for term, weights in record["stored_weights"].items() if weights["image_true"] == 0}
        assert added == set(record["added_terms"])


def perspective(aspect):
    return aspect.split(", such as", 1)[0]


def token_row(fig, y, tokens, matching, added, left, right, renderer):
    def width(fontsize):
        artists = [fig.text(0, 0, token["token"] + (", " if i + 1 < len(tokens) else ""),
                            fontsize=fontsize, weight="bold" if canonical(token["token"]) in matching else "normal")
                   for i, token in enumerate(tokens)]
        total = sum(a.get_window_extent(renderer).width for a in artists) / fig.bbox.width
        for artist in artists:
            artist.remove()
        return total
    fontsize = 12.6
    available = right - left
    measured = width(fontsize)
    if measured > available:
        fontsize *= available / measured * 0.985
    assert fontsize >= 10.5, "Token row needs more width"
    x = left
    for i, token in enumerate(tokens):
        term = canonical(token["token"])
        selected = term in matching
        new = term in added
        artist = fig.text(x, y, token["token"], va="center", fontsize=fontsize,
            weight="bold" if selected else "normal",
            color=("#6b460b" if new else "#123c60") if selected else "#111111",
            bbox=dict(facecolor=GOLD if new else BLUE, edgecolor="none", pad=1.6) if selected else None)
        x += artist.get_window_extent(renderer).width / fig.bbox.width
        if i + 1 < len(tokens):
            comma = fig.text(x, y, ", ", va="center", fontsize=fontsize)
            x += comma.get_window_extent(renderer).width / fig.bbox.width
    assert x <= right + 0.001, "Token row overflows"


def render(record, index):
    caption, logs = record["caption"], record["logs"]
    matching, added = set(record["matches"]), set(record["added_terms"])
    rows = [("SSR*", "text", "coarse", logs["text_false"]["payload"]["top_tokens"]),
            ("PTT", "text", "coarse", logs["text_true"]["payload"]["top_tokens"]),
            ("PTT", "image", "coarse", logs["image_true"]["payload"]["top_tokens"])]
    rows += [("MPP", "image", log["payload"]["view"] + ": " + perspective(log["payload"]["aspect"]),
              log["payload"]["top_tokens"]) for log in logs["adaptive_image"]]
    fig = plt.figure(figsize=(21, 7.4), dpi=160, facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    columns = [0.012, 0.255, 0.319, 0.381, 0.537, 0.992]
    top, header, bottom = 0.965, 0.897, 0.265
    height = (header - bottom) / len(rows)
    for i, heading in enumerate(("Image & Text", "Method", "Modality", "Perspective", "Top-10 Tokens")):
        fig.text((columns[i] + columns[i + 1]) / 2, (top + header) / 2, heading,
                 ha="center", va="center", fontsize=14, weight="bold")
    for y in (top, header, bottom):
        ax.plot([columns[0], columns[-1]], [y, y], color="#252525", lw=1.0)
    for x in columns[1:-1]:
        ax.plot([x, x], [bottom, top], color="#888888", lw=0.6)
    for i in range(1, len(rows)):
        left = columns[1] if i in (1, 3) else columns[2] if i == 2 else columns[3]
        ax.plot([left, columns[-1]], [header - i * height] * 2, color="#888888", lw=0.55)
    image_ax = fig.add_axes([0.025, 0.455, 0.216, 0.415])
    with Image.open(ROOT / record["image"]["local_file"]) as image:
        image_ax.imshow(image)
    image_ax.axis("off")
    fig.text(0.133, 0.417, textwrap.fill(caption["caption"], width=39),
             ha="center", va="top", fontsize=12.5, linespacing=1.35)
    fig.text(0.133, 0.285, f"{len(matching)} distinct matching tokens", ha="center", va="center",
             fontsize=11.5, weight="bold", color="#123c60")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for i, (method, modality, aspect, tokens) in enumerate(rows):
        y = header - (i + 0.5) * height
        if i < 3:
            if i == 0:
                fig.text((columns[1] + columns[2]) / 2, y, method, ha="center", va="center", fontsize=13)
            fig.text((columns[2] + columns[3]) / 2, y, modality, ha="center", va="center", fontsize=13)
        fig.text((columns[3] + columns[4]) / 2, y, aspect, ha="center", va="center", fontsize=11.8)
        token_row(fig, y, tokens, matching, added, columns[4] + 0.007, columns[-1] - 0.006, renderer)
    fig.text((columns[1] + columns[2]) / 2, header - 2 * height, "PTT", ha="center", va="center", fontsize=13)
    multi_y = header - 6 * height
    fig.text((columns[1] + columns[2]) / 2, multi_y, "MPP", ha="center", va="center", fontsize=13)
    fig.text((columns[2] + columns[3]) / 2, multi_y, "image", ha="center", va="center", fontsize=13)
    fig.text(0.016, 0.225, f"Case {index} | Flickr test | LLaVA-Mistral-7B, base | "
             f"image_id={caption['imgid']} | sentid={caption['sentid']} | {caption['filename']}", fontsize=10.3)
    fig.text(0.016, 0.185, "Blue / gold: positive in BOTH text Top-10 lists, absent from the single-summary image Top-10, "
             "recovered by MPP. Gold also adds a dimension absent from the final single-image vector.", fontsize=9.7)
    if added:
        entries = []
        for term in sorted(added):
            values = record["stored_weights"][term]
            entries.append(term + " = " + "/".join(str(values[k]) for k in ("text_false", "text_true", "image_true", "adaptive_image")))
        footer = "Saved integer weights (text false / text true / single image / multi image): " + ";  ".join(entries)
    else:
        footer = "Top-10 coverage gains only: all four matching terms already have positive weights in the final single-image Top-128 vector."
    fig.text(0.016, 0.145, footer, fontsize=10.0, color="#6b460b" if added else "#123c60")
    fig.text(0.016, 0.105, "SSR*: text sparse_manual=false; PTT: text/image sparse_manual=true; MPP: selected adaptive image views. "
             "Comparison lowercases tokens and PRESERVES the word-prefix marker; raw spellings are displayed.", fontsize=9.3)
    words = set(re.findall("[a-z]+", caption["caption"].lower()))
    pieces = sorted(term for term in matching if term.lstrip("\u2581") not in words)
    piece_note = " Word pieces remain unexpanded: " + ", ".join(pieces) + "." if pieces else ""
    fig.text(0.016, 0.065, "Image vector settings: single-view Top-128; MPP Top-30 per selected view. "
             "Shown ranks come directly from raw Top-10 logs." + piece_note, fontsize=9.3)
    fig.text(0.016, 0.025, "Evidence: evidence.json and matches.json include source paths/lines, token IDs, model weights, "
             "saved sparse weights and source SHA-256 checksums. These cases establish term coverage; retrieval gains require metric evaluation.", fontsize=9.0)
    path = ROOT / f"example_{caption['imgid']}.png"
    fig.savefig(path, dpi=160, bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)
    return path.name


def summarize(record, figure):
    caption = record["caption"]
    words = set(re.findall("[a-z]+", caption["caption"].lower()))
    terms = []
    for term, occurrences in record["matches"].items():
        terms.append(dict(canonical_term=term,
            term_kind="complete_word" if term.lstrip("\u2581") in words else "word_piece",
            text_false=occurrences[0]["text_false"], text_true=occurrences[0]["text_true"],
            adaptive_occurrences=[dict(view=entry["view"], aspect=entry["aspect"],
                                      **entry["token"], source=entry["source"]) for entry in occurrences],
            stored_weights=record["stored_weights"][term], added_to_final_sparse=term in record["added_terms"]))
    return dict(**caption, figure=figure, distinct_match_count=len(terms),
        added_dimension_count=len(record["added_terms"]), terms=terms,
        log_sources={role:([row["source"] for row in rows] if isinstance(rows, list) else rows["source"])
                     for role, rows in record["logs"].items()},
        sparse_sources={role:row["source"] for role,row in record["sparse_rows"].items()})


if __name__ == "__main__":
    evidence = json.loads((ROOT / "evidence.json").read_text(encoding="utf-8"))
    validate(evidence)
    results = []
    for index, record in enumerate(evidence["records"], 1):
        filename = render(record, index)
        results.append(summarize(record, filename))
        print(filename, "matches=", len(record["matches"]), "added_dimensions=", len(record["added_terms"]))
    summary = dict(validation="passed", captured_at_utc=evidence["captured_at_utc"],
        comparison="lowercase only; preserve tokenizer prefix; exclude zero-weight matches and pure whitespace/punctuation",
        population=evidence["population"], cases=results)
    (ROOT / "matches.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Validated all four cases, raw ranks, positive text weights, final sparse weights, image hashes and provenance.")
