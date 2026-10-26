from __future__ import annotations

import contextlib
import copy
import importlib.util
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from matplotlib.markers import MarkerStyle


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("plot_motivation", PROJECT_ROOT / "plot_motivation.py")
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Cannot import plot_motivation.py")
plot = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = plot
SPEC.loader.exec_module(plot)


def point(**overrides):
    row = dict(name="Test model", train_pairs_m=0.5, avg_score_pct=70.0,
               params_b=7.0, supervision="zero_shot", is_ours=False)
    row.update(overrides)
    return row


class MotivationPlotTests(unittest.TestCase):
    def tearDown(self):
        plot.plt.close("all")

    def test_real_results_match_all_eleven_screenshot_rows(self):
        expected = {
            "E5-V (LLaVA-LLaMA-8B)": (43.6, 0, 8, "zero_shot"),
            "E5-V (LLaVA-Mistral-7B)": (42.1, 0, 7, "zero_shot"),
            "FreeRet (LLaVA-Mistral-7B)": (60.1, 0, 7, "zero_shot"),
            "Finetuned E5-V (LLaVA-LLaMA-8B)": (65.4, 0.273, 8, "zero_shot"),
            "MMRet S1 (LLaVA-Mistral-7B)": (69.4, 26.2, 7, "zero_shot"),
            "VLM2Vec (LLaVA-Mistral-7B)": (60.0, 1.03, 7, "benchmark_supervised"),
            "GME (Qwen2-VL-7B)": (70.9, 8, 7, "benchmark_supervised"),
            "LamRA-Ret (Qwen2.5-VL-7B)": (55.3, 1.4, 7, "benchmark_supervised"),
            "LamRA-Rank20 (Qwen2.5-VL-7B)": (60.4, 2.5, 7, "benchmark_supervised"),
            "Re-M (LLaVA-LLaMA-8B)": (59.0, 0, 8, "zero_shot"),
            "Re-M (LLaVA-Mistral-7B)": (64.8, 0, 7, "zero_shot"),
        }
        self.assertEqual(len(plot.MODEL_RESULTS), 11)
        self.assertEqual({row["name"] for row in plot.MODEL_RESULTS}, set(expected))
        self.assertEqual(len(plot.prepare_points(plot.MODEL_RESULTS)), 11)
        for row in plot.MODEL_RESULTS:
            self.assertEqual(tuple(row[field] for field in (
                "avg_score_pct", "train_pairs_m", "params_b", "supervision"
            )), expected[row["name"]])
            self.assertAlmostEqual(sum(row["training_stages_m"]), row["train_pairs_m"])
            self.assertTrue(row["training_source"])
            self.assertEqual(row.get("is_ours", False), row["name"].startswith("Re-M"))
        rank = next(row for row in plot.MODEL_RESULTS if row["name"].startswith("LamRA-Rank"))
        self.assertEqual(rank["training_stages_m"], (1.4, 1.1))
        self.assertIn("Average column", plot.SCORE_SOURCE)
        self.assertIn("not deduplicated", plot.TRAINING_NOTE)
        self.assertIn("text examples", plot.TRAINING_NOTE)
        self.assertEqual(set(plot.TRAINING_SOURCES), {
            "e5v", "gme", "lamra", "freeret", "lamra_qwen25",
            "mmret", "mmret_s1", "vlm2vec", "vlm2vec_llava_next",
        })

    def test_new_checkpoints_keep_s1_scope_and_vlm2vec_estimate_provenance(self):
        mmret = next(row for row in plot.MODEL_RESULTS if row["name"].startswith("MMRet"))
        self.assertEqual(mmret["checkpoint"], "BAAI/BGE-VL-MLLM-S1")
        self.assertEqual(mmret["training_examples_reported"], 26_235_105)
        self.assertEqual(mmret["training_stages_m"], (26.2,))
        self.assertIn("no S2/MMEB", mmret["training_note"])
        self.assertEqual(mmret["checkpoint_source"], plot.TRAINING_SOURCES["mmret_s1"])

        vlm = next(row for row in plot.MODEL_RESULTS if row["name"].startswith("VLM2Vec"))
        self.assertEqual(vlm["checkpoint"], "TIGER-Lab/VLM2Vec-LLaVa-Next")
        self.assertTrue(vlm["training_estimated"])
        self.assertEqual(len(plot.VLM2VEC_TRAIN_SUBSET_SIZES_K), 20)
        self.assertEqual(plot.VLM2VEC_MAX_SUBSET_K, 100)
        capped_k = sum(min(size, 100) for size in plot.VLM2VEC_TRAIN_SUBSET_SIZES_K)
        self.assertEqual(capped_k, 1032)
        self.assertEqual(vlm["train_pairs_m"], round(capped_k / 1000, 2))
        self.assertIn("COCO/WebQA", vlm["training_note"])
        self.assertEqual(vlm["checkpoint_source"], plot.TRAINING_SOURCES["vlm2vec_llava_next"])
        self.assertIn("~1.03M estimated", plot.TRAINING_NOTE)

    def test_zero_is_a_valid_value_and_percentages_are_not_rescaled(self):
        rows = [point(train_pairs_m=0, avg_score_pct=0), point(avg_score_pct=0.8583)]
        result = plot.prepare_points(rows)
        self.assertEqual(result[0]["train_pairs_m"], 0)
        self.assertEqual(result[0]["avg_score_pct"], 0)
        self.assertEqual(result[1]["avg_score_pct"], 0.8583)

    def test_incomplete_rows_are_skipped_not_converted_to_zero(self):
        rows = [point(name="Unknown score", avg_score_pct=None), point(name="Known score")]
        original = copy.deepcopy(rows)
        with self.assertWarnsRegex(UserWarning, "Unknown score.*avg_score_pct"):
            result = plot.prepare_points(rows)
        self.assertEqual([row["name"] for row in result], ["Known score"])
        self.assertEqual(rows, original)

    def test_all_unknown_data_reports_required_fields_without_a_figure(self):
        unknown = [point(train_pairs_m=None, avg_score_pct=None, params_b=None, supervision=None)]
        before = plot.plt.get_fignums()
        with self.assertRaisesRegex(ValueError, "No complete model records.*MODEL_RESULTS"):
            plot.plot_motivation(unknown)
        self.assertEqual(plot.plt.get_fignums(), before)
        with tempfile.TemporaryDirectory() as temp_dir, contextlib.redirect_stderr(io.StringIO()), \
                patch.object(plot, "MODEL_RESULTS", unknown):
            output = Path(temp_dir) / "not-created"
            with self.assertRaises(SystemExit) as raised:
                plot.main(["--output-dir", str(output)])
            self.assertEqual(raised.exception.code, 2)
            self.assertFalse(output.exists())

    def test_invalid_numeric_values_are_rejected(self):
        for field, values in (
            ("train_pairs_m", [-1, float("inf"), float("nan"), True, "bad"]),
            ("avg_score_pct", [-0.1, 100.1, float("nan"), True]),
            ("params_b", [0, -2, float("inf"), False]),
        ):
            for value in values:
                with self.subTest(field=field, value=value):
                    with self.assertRaisesRegex(ValueError, field):
                        plot.prepare_points([point(**{field: value})])

    def test_invalid_metadata_has_actionable_errors(self):
        for field, value in (("name", ""), ("supervision", "unknown"), ("supervision", []),
                             ("is_ours", "yes"), ("label_offset", (1,)),
                             ("label_offset", (0, float("nan"))), ("color", "not-a-color")):
            with self.subTest(field=field, value=value):
                with self.assertRaisesRegex(ValueError, field):
                    plot.prepare_points([point(**{field: value})])
        with self.assertRaisesRegex(ValueError, "model dictionary"):
            plot.prepare_points([None])

    def test_symlog_axis_contains_real_zero_and_marker_areas_are_uniform(self):
        rows = [point(name="Small", train_pairs_m=0, params_b=2, avg_score_pct=50),
                point(name="Large", train_pairs_m=10, params_b=7, avg_score_pct=75)]
        fig, ax = plot.plot_motivation(rows)
        self.assertEqual(ax.get_xscale(), "symlog")
        self.assertIn(0, ax.get_xticks())
        self.assertEqual(float(ax.collections[0].get_offsets()[0, 0]), 0.0)
        self.assertEqual([collection.get_sizes()[0] for collection in ax.collections], [80, 80])
        self.assertEqual(plot.bubble_area(7), 80)
        self.assertEqual(plot.bubble_area(2), 80)
        self.assertEqual(len(ax.collections), len(rows))
        for collection, row in zip(ax.collections, rows):
            self.assertEqual(collection.get_sizes()[0], plot.bubble_area(row["params_b"]))
        self.assertFalse(fig.legends)

    def test_backbone_shapes_match_the_four_requested_markers_without_mutating_data(self):
        expected = {
            "LLaVA-Mistral-7B": "o", "LLaVA-LLaMA-8B": "s",
            "Qwen2-VL-7B": "^", "Qwen2.5-VL-7B": "D",
        }
        original = copy.deepcopy(plot.MODEL_RESULTS)
        fig, ax = plot.plot_motivation(plot.MODEL_RESULTS)
        self.assertEqual(plot.BACKBONE_MARKERS, expected)
        for collection, annotation, row in zip(ax.collections, ax.texts, plot.MODEL_RESULTS):
            style = MarkerStyle(expected[row["backbone"]])
            path = style.get_path().transformed(style.get_transform())
            np.testing.assert_array_equal(collection.get_paths()[0].vertices, path.vertices)
            np.testing.assert_array_equal(collection.get_paths()[0].codes, path.codes)
            self.assertEqual(collection.get_sizes()[0], 80)
            radius = 80 ** 0.5 * max(np.hypot(x, y) for x, y in path.vertices)
            self.assertAlmostEqual(annotation.arrow_patch.shrinkB, radius + 1)
        self.assertEqual(plot.MODEL_RESULTS, original)
        legend = fig.legends[0]
        self.assertEqual([item.get_text() for item in legend.get_texts()],
                         list(expected))
        handles = getattr(legend, "legend_handles", None)
        if handles is None:
            handles = legend.legendHandles
        self.assertEqual([handle.get_marker() for handle in handles[:4]], list(expected.values()))
        self.assertTrue(all(handle.get_markerfacecolor() == "#bbbbbb" for handle in handles[:4]))

    def test_custom_and_demo_data_do_not_get_invented_backbone_legends(self):
        rows = [point(name="Missing"), point(name="Unknown", backbone="Other backbone"),
                point(name="Known", backbone="Qwen2-VL-7B")]
        fig, ax = plot.plot_motivation(rows)
        for collection, marker in zip(ax.collections, ("o", "o", "^")):
            style = MarkerStyle(marker)
            path = style.get_path().transformed(style.get_transform())
            np.testing.assert_array_equal(collection.get_paths()[0].vertices, path.vertices)
        self.assertEqual(len(fig.legends), 1)
        self.assertEqual([item.get_text() for item in fig.legends[0].get_texts()],
                         ["Qwen2-VL-7B"])
        demo_fig, _ = plot.plot_motivation(plot.DEMO_RESULTS, demo=True)
        self.assertFalse(demo_fig.legends)

    def test_plot_uses_supplied_average_scores_and_marks_ours(self):
        rows = [point(name="Baseline", avg_score_pct=61.7),
                point(name="Ours", train_pairs_m=0, avg_score_pct=83.2,
                      supervision="benchmark_supervised", is_ours=True, label_offset=(12, 2))]
        _, ax = plot.plot_motivation(rows)
        self.assertAlmostEqual(ax.collections[0].get_offsets()[0, 1], 61.7)
        self.assertAlmostEqual(ax.collections[1].get_offsets()[0, 1], 83.2)
        self.assertAlmostEqual(ax.collections[0].get_facecolors()[0, 3], 0.95)
        self.assertAlmostEqual(ax.collections[1].get_facecolors()[0, 3], 0.15)
        label = next(text for text in ax.texts if text.get_text() == "Ours")
        self.assertEqual(label.get_color(), "#c52828")
        self.assertEqual(label.get_position(), (12, 2))

    def test_fill_alpha_inherits_defaults_and_accepts_zero_and_one_without_mutation(self):
        rows = [point(), point(fill_alpha=None),
                point(supervision="benchmark_supervised", fill_alpha=None),
                point(fill_alpha=0), point(fill_alpha=1),
                point(supervision="benchmark_supervised", fill_alpha=0.6)]
        original = copy.deepcopy(rows)
        prepared = plot.prepare_points(rows)
        expected = [0.95, 0.95, 0.15, 0, 1, 0.6]
        self.assertEqual([row["fill_alpha"] for row in prepared], expected)
        _, ax = plot.plot_motivation(rows)
        for collection, alpha in zip(ax.collections, expected):
            self.assertAlmostEqual(collection.get_facecolors()[0, 3], alpha)
            self.assertAlmostEqual(collection.get_edgecolors()[0, 3], 1.0)
        self.assertEqual(rows, original)

    def test_invalid_fill_alpha_is_rejected_before_allocating_a_figure(self):
        for value in (-0.1, 1.1, float("nan"), float("inf"), -float("inf"), True, False, "bad", []):
            with self.subTest(value=value):
                before = plot.plt.get_fignums()
                with self.assertRaisesRegex(ValueError, "fill_alpha"):
                    plot.plot_motivation([point(fill_alpha=value)])
                self.assertEqual(plot.plt.get_fignums(), before)

    def test_only_requested_models_are_faded_and_backbone_legend_is_unchanged(self):
        expected_opacity = {"zero_shot": 0.95, "benchmark_supervised": 0.15}
        expected_overrides = {
            "Finetuned E5-V (LLaVA-LLaMA-8B)": 0.15,
            "MMRet S1 (LLaVA-Mistral-7B)": 0.15,
            "GME (Qwen2-VL-7B)": 0.08,
        }
        self.assertEqual({row["name"]: row["fill_alpha"] for row in plot.MODEL_RESULTS
                          if "fill_alpha" in row}, expected_overrides)
        for rows, demo in ((plot.MODEL_RESULTS, False), (plot.DEMO_RESULTS, True)):
            with self.subTest(demo=demo):
                fig, ax = plot.plot_motivation(rows, demo=demo)
                for collection, row in zip(ax.collections, rows):
                    alpha = expected_overrides.get(row["name"], expected_opacity[row["supervision"]])
                    color = plot.OURS_COLOR if row.get("is_ours") else row["color"]
                    self.assertEqual(tuple(collection.get_facecolors()[0]),
                                     plot.mcolors.to_rgba(color, alpha), row["name"])
                    # Fade only the fill: outlines must remain fully visible.
                    edge_color = "#a73832" if row.get("is_ours") else "#555555"
                    self.assertEqual(tuple(collection.get_edgecolors()[0]),
                                     plot.mcolors.to_rgba(edge_color), row["name"])
                expected_labels = [] if demo else list(plot.BACKBONE_MARKERS)
                self.assertEqual([text.get_text() for legend in fig.legends for text in legend.get_texts()],
                                 expected_labels)
                for legend in fig.legends:
                    handles = getattr(legend, "legend_handles", None)
                    if handles is None:  # Matplotlib < 3.7
                        handles = legend.legendHandles
                    for handle in handles:
                        self.assertEqual(handle.get_markerfacecolor(), "#bbbbbb")
                        self.assertAlmostEqual(plot.mcolors.to_rgba(handle.get_markeredgecolor())[3], 1.0)
                plot.plt.close(fig)

    def test_axes_keep_average_recall_labels_and_benchmark_metadata_without_footer(self):
        fig, ax = plot.plot_motivation([point()])
        self.assertEqual(ax.get_ylabel(), "Average Recall")
        self.assertEqual(ax.get_xlabel(), plot.X_LABEL)
        self.assertFalse(fig.texts)
        self.assertIn("Average recall", plot.BENCHMARK_NOTE)
        self.assertEqual(plot.X_LABEL, "Training data (million pairs)")
        self.assertNotIn("\n", plot.X_LABEL)
        self.assertAlmostEqual(fig.subplotpars.bottom * fig.get_figheight(), 0.378 + 0.4)

    def test_legends_are_below_the_xlabel_and_inside_the_canvas(self):
        for demo in (False, True):
            for dpi in (100, 300):
                with self.subTest(demo=demo, dpi=dpi):
                    rows = plot.DEMO_RESULTS if demo else plot.MODEL_RESULTS
                    fig, ax = plot.plot_motivation(rows, demo=demo)
                    fig.set_dpi(dpi)
                    fig.canvas.draw()
                    renderer = fig.canvas.get_renderer()
                    xlabel_box = ax.xaxis.label.get_window_extent(renderer)
                    self.assertEqual(len(fig.legends), 0 if demo else 1)
                    expected_labels = [] if demo else list(plot.BACKBONE_MARKERS)
                    self.assertEqual([text.get_text() for legend in fig.legends for text in legend.get_texts()],
                                     expected_labels)
                    boxes = [legend.get_window_extent(renderer) for legend in fig.legends]
                    for legend, box in zip(fig.legends, boxes):
                        self.assertFalse(legend.get_frame_on())
                        self.assertEqual(legend.get_title().get_text(), "")
                        self.assertGreaterEqual(box.x0, fig.bbox.x0)
                        self.assertGreaterEqual(box.y0, fig.bbox.y0)
                        self.assertLessEqual(box.x1, fig.bbox.x1)
                        self.assertLessEqual(box.y1, xlabel_box.y0 - 2 * dpi / 72)
                    if not demo:
                        legend = fig.legends[0]
                        self.assertEqual(legend._ncols, 2)
                        self.assertAlmostEqual((boxes[0].x0 + boxes[0].x1) / 2, fig.bbox.width / 2)
                        text_boxes = [text.get_window_extent(renderer) for text in legend.get_texts()]
                        # Two columns, with exactly two aligned entries each.
                        for index in (0, 2):
                            self.assertAlmostEqual(text_boxes[index].x0, text_boxes[index + 1].x0)
                            self.assertGreater(text_boxes[index].y0, text_boxes[index + 1].y0)
                            self.assertAlmostEqual(text_boxes[index].y0, text_boxes[0].y0)
                            self.assertAlmostEqual(text_boxes[index + 1].y0, text_boxes[1].y0)
                        for index, box in enumerate(text_boxes):
                            for other in text_boxes[index + 1:]:
                                self.assertFalse(box.overlaps(other))
                    # Keep panel width while reducing its physical height.
                    self.assertAlmostEqual(ax.get_position().width * fig.get_figwidth(), 3.6 * 0.83)
                    self.assertAlmostEqual(ax.get_position().height * fig.get_figheight(),
                                           2.1 * ((0.83 if demo else 0.95) - 0.18))
                    plot.plt.close(fig)

    def test_no_bottom_notes_and_tight_crop_includes_external_legends_at_both_dpis(self):
        for demo in (False, True):
            for dpi in (100, 300):
                with self.subTest(demo=demo, dpi=dpi):
                    rows = plot.DEMO_RESULTS if demo else plot.MODEL_RESULTS
                    fig, ax = plot.plot_motivation(rows, demo=demo)
                    fig.set_dpi(dpi)
                    fig.canvas.draw()
                    renderer = fig.canvas.get_renderer()
                    expected_texts = ["ILLUSTRATIVE DATA\nNOT EXPERIMENTAL RESULTS"] if demo else []
                    self.assertEqual([text.get_text() for text in fig.texts], expected_texts)
                    content_bottom = min(
                        (legend.get_window_extent(renderer).y0 for legend in fig.legends),
                        default=ax.xaxis.label.get_window_extent(renderer).y0,
                    )
                    self.assertAlmostEqual(fig.get_tightbbox(renderer).y0, content_bottom / dpi)
                    self.assertEqual([text.get_text() for text in ax.get_xticklabels()],
                                     ["0", "0.1", "0.5", "1", "2", "5", "10", "20"])
                    plot.plt.close(fig)

    def test_demo_legends_do_not_cover_bubbles_or_model_labels(self):
        from matplotlib.transforms import Bbox

        for dpi in (100, 300):
            with self.subTest(dpi=dpi):
                fig, ax = plot.plot_motivation(plot.DEMO_RESULTS, demo=True)
                fig.set_dpi(dpi)
                fig.canvas.draw()
                renderer = fig.canvas.get_renderer()
                boxes = [legend.get_window_extent(renderer) for legend in fig.legends]
                for row in plot.DEMO_RESULTS:
                    x, y = ax.transData.transform((row["train_pairs_m"], row["avg_score_pct"]))
                    radius = (plot.bubble_area(row["params_b"]) ** 0.5 + plot.BUBBLE_EDGE_WIDTH) * dpi / 144
                    bubble_box = Bbox.from_extents(x - radius, y - radius, x + radius, y + radius)
                    for box in boxes:
                        self.assertFalse(box.overlaps(bubble_box), row["name"])
                for text in ax.texts:
                    for box in boxes:
                        self.assertFalse(box.overlaps(text.get_window_extent(renderer)), text.get_text())
                plot.plt.close(fig)

    def test_no_size_legend_or_empty_scatter_handles_for_real_demo_and_custom_data(self):
        for rows, demo in ((plot.MODEL_RESULTS, False), (plot.DEMO_RESULTS, True), ([point()], False)):
            with self.subTest(demo=demo, count=len(rows)):
                fig, ax = plot.plot_motivation(rows, demo=demo)
                expected_labels = list(plot.BACKBONE_MARKERS) if rows is plot.MODEL_RESULTS else []
                self.assertEqual(len(fig.legends), int(bool(expected_labels)))
                self.assertEqual([text.get_text() for legend in fig.legends for text in legend.get_texts()],
                                 expected_labels)
                for legend in fig.legends:
                    self.assertEqual(legend.get_title().get_text(), "")
                    self.assertNotIn("LLM backbone size", legend.get_title().get_text())
                self.assertEqual(len(ax.collections), len(rows))
                for collection, row in zip(ax.collections, rows):
                    self.assertEqual(list(collection.get_offsets()[0]),
                                     [row["train_pairs_m"], row["avg_score_pct"]])
                    self.assertEqual(len(collection.get_offsets()), 1)
                plot.plt.close(fig)

    def test_real_plot_keeps_exact_coordinates_scores_and_compact_labels(self):
        fig, ax = plot.plot_motivation(plot.MODEL_RESULTS)
        self.assertEqual(len(ax.texts), 11)
        expected_labels = [
            "E5-V", "E5-V", "FreeRet", "Finetuned E5-V",
            "MMRet", "VLM2Vec", "GME", "LamRA-Ret",
            "LamRA-Rank", "Re-M", "Re-M",
        ]
        for collection, annotation, row, label in zip(ax.collections, ax.texts, plot.MODEL_RESULTS,
                                                     expected_labels):
            self.assertEqual(list(collection.get_offsets()[0]), [row["train_pairs_m"], row["avg_score_pct"]])
            self.assertEqual(collection.get_sizes()[0], plot.bubble_area(row["params_b"]))
            self.assertEqual(annotation.get_text(), label)
            self.assertNotIn("\n", annotation.get_text())
            self.assertNotIn("rerank", annotation.get_text())
            self.assertNotIn(f"{row['avg_score_pct']:.1f}", annotation.get_text())
            self.assertTrue(row["backbone"])
            self.assertNotIn(row["backbone"], annotation.get_text())
            self.assertEqual(annotation.get_fontsize(), 8.5)
            self.assertIsNotNone(annotation.arrow_patch)
            self.assertAlmostEqual(annotation.arrow_patch.get_linewidth(), 0.5)
            self.assertEqual(annotation.arrow_patch.shrinkA, 2)
            self.assertAlmostEqual(annotation.arrow_patch.shrinkB,
                                   plot.marker_radius(plot.BACKBONE_MARKERS[row["backbone"]]) + 1)
            if row.get("is_ours"):
                self.assertEqual(annotation.get_color(), "#c52828")
                self.assertEqual(collection.get_facecolors()[0, 3], plot.SUPERVISION["zero_shot"][0])
        # Hide the display suffix without changing the underlying method metadata.
        self.assertEqual(sum("+ rerank" in row["method"] for row in plot.MODEL_RESULTS), 2)
        self.assertEqual({row["method"]: row["display_method"] for row in plot.MODEL_RESULTS
                          if "display_method" in row}, {
            "MMRet S1": "MMRet",
            "LamRA-Rank$_{20}$ (+ rerank)": "LamRA-Rank",
        })
        self.assertEqual(len(ax.collections), 11)
        self.assertEqual(len(fig.legends), 1)
        self.assertEqual([item.get_text() for item in fig.legends[0].get_texts()],
                         list(plot.BACKBONE_MARKERS))

    def test_real_labels_keep_two_point_clearance_and_stay_inside_axes(self):
        from matplotlib.text import Text
        from matplotlib.transforms import Bbox

        for dpi in (100, 300):
            with self.subTest(dpi=dpi):
                fig, ax = plot.plot_motivation(plot.MODEL_RESULTS)
                fig.set_dpi(dpi)
                fig.canvas.draw()
                renderer = fig.canvas.get_renderer()
                axes = ax.get_window_extent(renderer)
                legends = [item.get_window_extent(renderer) for item in fig.legends]
                # Annotation extents include leader lines. Test the text box
                # separately so an intentional line ending at a bubble is OK.
                labels = [Text.get_window_extent(item, renderer) for item in ax.texts]
                bubbles = []
                for collection, row in zip(ax.collections, plot.MODEL_RESULTS):
                    x, y = ax.transData.transform((row["train_pairs_m"], row["avg_score_pct"]))
                    marker_box = collection.get_paths()[0].get_extents()
                    scale = collection.get_sizes()[0] ** 0.5 * dpi / 72
                    edge = plot.BUBBLE_EDGE_WIDTH * dpi / 144
                    bubbles.append(Bbox.from_extents(
                        x + marker_box.x0 * scale - edge, y + marker_box.y0 * scale - edge,
                        x + marker_box.x1 * scale + edge, y + marker_box.y1 * scale + edge,
                    ))
                for index, box in enumerate(labels):
                    name = plot.MODEL_RESULTS[index]["name"]
                    self.assertGreaterEqual(box.x0, axes.x0, name)
                    self.assertGreaterEqual(box.y0, axes.y0, name)
                    self.assertLessEqual(box.x1, axes.x1, name)
                    self.assertLessEqual(box.y1, axes.y1, name)
                    for other in labels[index + 1:] + legends + bubbles:
                        # Expanding each box by 1pt requires a 2pt gap between them.
                        self.assertFalse(box.padded(dpi / 72).overlaps(other.padded(dpi / 72)), name)
                for box in legends:
                    for bubble in bubbles:
                        self.assertFalse(box.padded(dpi / 72).overlaps(bubble.padded(dpi / 72)))
                plot.plt.close(fig)

    def test_single_column_print_size_keeps_labels_and_bubbles_legible(self):
        for dpi in (100, 300):
            with self.subTest(dpi=dpi):
                fig, ax = plot.plot_motivation(plot.MODEL_RESULTS)
                self.assertEqual(list(fig.get_size_inches()), [3.6, 2.5])
                self.assertAlmostEqual((1 - fig.subplotpars.top) * fig.get_figheight(), 0.105)
                self.assertEqual(plot.MARKER_AREA_POINTS2, 80)
                self.assertEqual(ax.get_xlim(), (-0.065, 26.2 * 1.9))
                self.assertEqual(ax.get_ylim(), (37.1, 75.9))
                self.assertEqual(ax.get_xlabel(), "Training data (million pairs)")
                self.assertEqual(ax.xaxis.label.get_fontsize(), 10.5)
                self.assertEqual(ax.yaxis.label.get_fontsize(), 10.5)
                fig.set_dpi(dpi)
                fig.canvas.draw()
                content_box = fig.get_tightbbox(fig.canvas.get_renderer())
                width_cm = (content_box.width + 0.16) * 2.54
                height_cm = (content_box.height + 0.16) * 2.54
                self.assertGreaterEqual(width_cm, 9.0)
                self.assertLessEqual(width_cm, 9.3)
                # Reduce export height while preserving width, fonts and markers.
                self.assertGreaterEqual(height_cm, 6.3)
                self.assertLessEqual(height_cm, 6.5)
                print_scale = 8.5 / width_cm
                texts = list(ax.texts) + ax.get_xticklabels() + ax.get_yticklabels()
                texts += [text for legend in fig.legends for text in legend.get_texts()]
                for text in texts:
                    self.assertGreaterEqual(text.get_fontsize() * print_scale, 7.8, text.get_text())
                for collection in ax.collections:
                    diameter_mm = collection.get_sizes()[0] ** 0.5 / 72 * 25.4 * print_scale
                    self.assertGreaterEqual(diameter_mm, 2.8)
                    self.assertLessEqual(diameter_mm, 3.1)
                    self.assertAlmostEqual(collection.get_linewidths()[0], 0.65)
                plot.plt.close(fig)

    def test_real_leader_lines_do_not_cross_other_labels(self):
        from matplotlib.text import Text

        for dpi in (100, 300):
            with self.subTest(dpi=dpi):
                fig, ax = plot.plot_motivation(plot.MODEL_RESULTS)
                fig.set_dpi(dpi)
                fig.canvas.draw()
                renderer = fig.canvas.get_renderer()
                labels = [Text.get_window_extent(text, renderer) for text in ax.texts]
                for index, annotation in enumerate(ax.texts):
                    arrow = annotation.arrow_patch
                    path = arrow.get_path().transformed(arrow.get_transform())
                    for other_index, box in enumerate(labels):
                        if other_index != index:
                            self.assertFalse(path.intersects_bbox(box, filled=False),
                                             f"{plot.MODEL_RESULTS[index]['name']} -> "
                                             f"{plot.MODEL_RESULTS[other_index]['name']}")
                finetuned = ax.texts[3].arrow_patch.get_connectionstyle()
                self.assertEqual((finetuned.angleA, finetuned.angleB, finetuned.rad), (0, 90, 2))
                plot.plt.close(fig)

    def test_demo_banner_and_labels_fit_single_column_canvas(self):
        from matplotlib.text import Text

        for dpi in (100, 300):
            with self.subTest(dpi=dpi):
                fig, ax = plot.plot_motivation(plot.DEMO_RESULTS, demo=True)
                fig.set_dpi(dpi)
                fig.canvas.draw()
                renderer = fig.canvas.get_renderer()
                banner = fig.texts[0]
                self.assertEqual(banner.get_text(), "ILLUSTRATIVE DATA\nNOT EXPERIMENTAL RESULTS")
                self.assertEqual(banner.get_position(), (0.545, 0.995))
                self.assertAlmostEqual((1 - fig.subplotpars.top) * fig.get_figheight(), 0.17 * 2.1)
                banner_box = banner.get_window_extent(renderer)
                label_boxes = [Text.get_window_extent(text, renderer) for text in ax.texts]
                for box in [banner_box] + label_boxes:
                    self.assertGreaterEqual(box.x0, fig.bbox.x0)
                    self.assertGreaterEqual(box.y0, fig.bbox.y0)
                    self.assertLessEqual(box.x1, fig.bbox.x1)
                    self.assertLessEqual(box.y1, fig.bbox.y1)
                for index, box in enumerate(label_boxes):
                    self.assertFalse(banner_box.overlaps(box))
                    for other in label_boxes[index + 1:]:
                        self.assertFalse(box.padded(dpi / 72).overlaps(other.padded(dpi / 72)),
                                         ax.texts[index].get_text())
                plot.plt.close(fig)

    def test_main_real_exports_screenshot_data_without_demo_watermark(self):
        with tempfile.TemporaryDirectory() as temp_dir, contextlib.redirect_stdout(io.StringIO()) as output:
            plot.main(["--output-dir", temp_dir])
            self.assertTrue((Path(temp_dir) / "motivation_training_data.pdf").exists())
            self.assertTrue((Path(temp_dir) / "motivation_training_data.png").exists())
            self.assertNotIn("DEMO ONLY", output.getvalue())
            self.assertFalse((Path(temp_dir) / "motivation_training_data_demo.pdf").exists())
            self.assertFalse(plot.plt.get_fignums())

    def test_demo_is_explicitly_labeled_and_not_named_as_real_models(self):
        self.assertFalse({row["name"] for row in plot.DEMO_RESULTS} &
                         {row["name"] for row in plot.MODEL_RESULTS})
        fig, _ = plot.plot_motivation(plot.DEMO_RESULTS, demo=True)
        self.assertTrue(any("NOT EXPERIMENTAL RESULTS" in text.get_text() for text in fig.texts))

    def test_pdf_png_export_and_demo_names_do_not_overwrite_real_outputs(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir)
            fig, _ = plot.plot_motivation([point()])
            real_paths = plot.save_figure(fig, output)
            hashes = {path: path.read_bytes() for path in real_paths}
            demo_fig, _ = plot.plot_motivation(plot.DEMO_RESULTS, demo=True)
            demo_paths = plot.save_figure(demo_fig, output, demo=True)
            self.assertFalse(set(real_paths) & set(demo_paths))
            for path, content in hashes.items():
                self.assertEqual(path.read_bytes(), content)
            self.assertTrue(real_paths[0].read_bytes().startswith(b"%PDF-"))
            self.assertTrue(demo_paths[0].read_bytes().startswith(b"%PDF-"))
            with Image.open(real_paths[1]) as image:
                self.assertEqual(image.format, "PNG")
                self.assertAlmostEqual(image.info["dpi"][0], 300, delta=0.1)
                # Match the tight content bounds plus 0.08in padding on each side,
                # rather than retaining the canvas space formerly used by notes.
                fig.set_dpi(300)
                fig.canvas.draw()
                content_box = fig.get_tightbbox(fig.canvas.get_renderer())
                self.assertAlmostEqual(image.width, (content_box.width + 0.16) * 300, delta=2)
                self.assertAlmostEqual(image.height, (content_box.height + 0.16) * 300, delta=2)
            self.assertIn(b"/FontFile2", real_paths[0].read_bytes())
            self.assertNotIn(b"/Subtype /Image", real_paths[0].read_bytes())

    def test_main_demo_exports_and_closes_figure(self):
        with tempfile.TemporaryDirectory() as temp_dir, contextlib.redirect_stdout(io.StringIO()) as output:
            plot.main(["--demo", "--output-dir", temp_dir])
            self.assertTrue((Path(temp_dir) / "motivation_training_data_demo.pdf").exists())
            self.assertTrue((Path(temp_dir) / "motivation_training_data_demo.png").exists())
            self.assertIn("DEMO ONLY", output.getvalue())
            self.assertFalse(plot.plt.get_fignums())

    def test_noninteractive_backend_and_style_do_not_leak_font_changes(self):
        original = plot.matplotlib.rcParams["font.size"]
        plot.plot_motivation([point()])
        self.assertEqual(plot.matplotlib.get_backend().lower(), "agg")
        self.assertEqual(plot.matplotlib.rcParams["font.size"], original)


if __name__ == "__main__":
    unittest.main()
