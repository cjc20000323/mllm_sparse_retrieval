"""Regression checks for splitting each hybrid-recall figure into three PDFs."""

import contextlib
import importlib.util
import io
import os
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
from matplotlib.figure import Figure
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "hybrid_plot_under_test", PROJECT_ROOT / "plot_hybrid_weight_recall_three_panel.py"
)
assert SPEC is not None and SPEC.loader is not None
plot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plot)
METRICS = ((1, "Recall@1"), (5, "Recall@5"), (10, "Recall@10"), ("average", "Average Recall"))


class HybridWeightRecallPlotTests(unittest.TestCase):
    def tearDown(self):
        plot.plt.close("all")

    def test_all_metrics_keep_three_panel_groupings_and_original_curve_styles(self):
        for metric, ylabel in METRICS:
            panels = plot.build_panels(metric)
            self.assertEqual([panel[0] for panel in panels], ["flickr", "person_retrieval", "webqa"])
            for (panel, title, series), expected_count in zip(panels, (4, 4, 1)):
                with self.subTest(metric=metric, panel=panel):
                    fig, ax = plot.create_panel_figure(panel, title, series, ylabel)
                    expected = [(label, values) for label, values in series.items() if values[1].size]
                    self.assertEqual(len(ax.lines), expected_count)
                    self.assertEqual(len(fig.axes), 1)
                    self.assertEqual(ax.get_title(), title)
                    self.assertEqual(ax.get_xlabel(), "Dense weight α (sparse = 1 − α)")
                    self.assertEqual(ax.get_ylabel(), ylabel)
                    self.assertEqual(ax.get_xlim(), (0.05, 0.95))
                    np.testing.assert_array_equal(ax.get_xticks(), plot.ALPHA)
                    for line, (label, (xs, ys, color, linestyle, marker)) in zip(ax.lines, expected):
                        np.testing.assert_array_equal(line.get_xdata(), xs)
                        np.testing.assert_array_equal(line.get_ydata(), ys)
                        np.testing.assert_array_equal(xs, plot.ALPHA)
                        self.assertEqual(line.get_label(), label)
                        self.assertEqual(line.get_color(), color)
                        self.assertEqual(line.get_linestyle(), linestyle)
                        self.assertEqual(line.get_marker(), marker)
                        self.assertEqual(line.get_linewidth(), 2.5)
                        self.assertEqual(line.get_markersize(), 6)
                    self.assertEqual([item.get_text() for item in ax.get_legend().get_texts()],
                                     [label for label, _ in expected])
                    if panel == "person_retrieval":
                        self.assertEqual(sum(name.startswith("RSTPReid · ") for name in series), 2)
                        self.assertEqual(sum(name.startswith("CUHK-PEDES · ") for name in series), 2)
                    plot.plt.close(fig)

    def test_per_weight_average_and_missing_webqa_data_are_unchanged(self):
        np.testing.assert_allclose(plot.ALPHA, np.arange(0.1, 1.0, 0.1))
        for definitions in (plot.FLICKR_SERIES, plot.RSTPREID_SERIES, plot.CUHK_PEDES_SERIES, plot.WEBQA_SERIES):
            for values, *_ in definitions.values():
                if all(key in values for key in (1, 5, 10)):
                    expected = np.mean(np.vstack([values[key] for key in (1, 5, 10)]), axis=0)
                    np.testing.assert_array_equal(values["average"], expected)
                    self.assertEqual(expected.shape, (9,))
                else:
                    self.assertEqual(values["average"].size, 0)
        missing = plot.build_results(plot.WEBQA_SERIES, "average")["LLaVA-8B"]
        self.assertEqual(missing[0].size, 0)
        self.assertEqual(missing[1].size, 0)

    def test_empty_panel_has_no_fabricated_curve(self):
        fig, ax = plot.plt.subplots()
        plot.plot_panel(ax, "Empty", {"missing": ([], [], "red", "-", "o")}, "Recall@1")
        self.assertEqual(len(ax.lines), 0)
        self.assertIsNone(ax.get_legend())
        self.assertEqual([item.get_text() for item in ax.texts], ["No data supplied"])

    def test_mismatched_data_still_raise(self):
        fig, ax = plot.plt.subplots()
        with self.assertRaisesRegex(ValueError, "x/y arrays must have the same length"):
            plot.plot_panel(ax, "Bad", {"bad": ([0.1], [1, 2], "red", "-", "o")}, "Recall@1")
        with self.assertRaisesRegex(ValueError, "recall arrays must have the same length"):
            plot.add_average_recall({"bad": ({1: np.ones(9), 5: np.ones(8), 10: np.ones(9)}, "red", "-", "o")})

    def test_main_exports_twelve_single_page_files_without_touching_old_figures(self):
        captures = []
        original_savefig = Figure.savefig

        def capture(figure, filename, *args, **kwargs):
            self.assertEqual(len(figure.axes), 1)
            paper_layout = Path(filename).name in {
                "hybrid_weight_recall_flickr_average.pdf", "hybrid_weight_recall_person_retrieval_average.pdf",
            }
            compact = Path(filename).name == "hybrid_weight_recall_flickr_recall_10.pdf"
            expected_size = plot.PAPER_AVERAGE_FIGSIZE if paper_layout else (3.4, 3.8) if compact else (7, 6)
            np.testing.assert_array_equal(figure.get_size_inches(), expected_size)
            if paper_layout:
                self.assertIsNone(kwargs.get("bbox_inches"))
                self.assertEqual(kwargs.get("pad_inches"), 0)
                self.assertNotIn("bbox_extra_artists", kwargs)
                self.assertEqual(matplotlib.rcParams["pdf.fonttype"], 42)
                self.assertEqual(figure.axes[0].get_legend().get_texts()[0].get_fontsize(), 25)
            elif compact:
                self.assertEqual(kwargs.get("bbox_extra_artists"), [figure.axes[0].get_legend()])
                self.assertEqual(matplotlib.rcParams["pdf.fonttype"], 42)
            else:
                self.assertNotIn("bbox_extra_artists", kwargs)
                self.assertEqual(figure.axes[0].get_legend().get_texts()[0].get_fontsize(), 9)
            captures.append((Path(filename).name, len(figure.axes[0].lines)))
            return original_savefig(figure, filename, *args, **kwargs)

        expected = {
            f"hybrid_weight_recall_{panel}_{metric}.pdf": curves
            for panel, curves in (("flickr", 4), ("person_retrieval", 4), ("webqa", 1))
            for metric in ("recall_1", "recall_5", "recall_10", "average")
        }
        old_names = ["hybrid_weight_recall_three_panel.pdf", "hybrid_weight_recall_three_panel_average.pdf"]
        old_names += [f"hybrid_weight_recall_three_panel_recall_{recall}.pdf" for recall in (1, 5, 10)]
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for name in old_names:
                (directory / name).write_bytes(b"existing three-panel result")
            previous_cwd = Path.cwd()
            try:
                os.chdir(directory)
                with patch.object(Figure, "savefig", new=capture), contextlib.redirect_stdout(io.StringIO()) as output:
                    plot.main()
            finally:
                os.chdir(previous_cwd)
            self.assertEqual(dict(captures), expected)
            self.assertEqual(len(captures), 12)
            self.assertEqual(set(path.name for path in directory.iterdir()), set(expected) | set(old_names))
            self.assertEqual(output.getvalue().count("Saved figure to "), 12)
            for name in expected:
                contents = (directory / name).read_bytes()
                self.assertTrue(contents.startswith(b"%PDF-"), name)
                self.assertTrue(contents.rstrip().endswith(b"%%EOF"), name)
                self.assertEqual(len(re.findall(rb"/Type\s*/Page\b", contents)), 1, name)
                if name in {"hybrid_weight_recall_flickr_recall_10.pdf", "hybrid_weight_recall_flickr_average.pdf",
                            "hybrid_weight_recall_person_retrieval_average.pdf"}:
                    self.assertIn(b"/FontFile2", contents)
                    self.assertNotIn(b"/Subtype /Image", contents)
            for name in old_names:
                self.assertEqual((directory / name).read_bytes(), b"existing three-panel result")
        self.assertFalse(plot.plt.get_fignums())

    def test_compact_flickr_keeps_all_points_curve_styles_and_full_model_metadata(self):
        for metric, ylabel in ((10, "Recall@10"), ("average", "Average Recall")):
            with self.subTest(metric=metric):
                panel, title, series = plot.build_panels(metric)[0]
                ordinary_fig, ordinary = plot.create_panel_figure(panel, title, series, ylabel)
                fig, ax = plot.create_panel_figure(panel, title, series, ylabel, compact=True)
                np.testing.assert_array_equal(fig.get_size_inches(), (3.4, 3.8))
                self.assertEqual(ax.get_title(), "Flickr · image-text\nretrieval")
                self.assertEqual(ax.get_xlabel(), "Dense weight α\n(sparse = 1 − α)")
                self.assertEqual(ax.get_ylabel(), ylabel)
                self.assertEqual(ax.title.get_fontsize(), 18)
                self.assertEqual(ax.xaxis.label.get_fontsize(), 18)
                self.assertEqual(ax.yaxis.label.get_fontsize(), 18)
                self.assertEqual(ax.get_xlim(), ordinary.get_xlim())
                self.assertEqual(ax.get_ylim(), ordinary.get_ylim())
                np.testing.assert_array_equal(ax.get_xticks(), plot.ALPHA[::2])
                self.assertEqual(len(ax.lines), 4)
                for line, original in zip(ax.lines, ordinary.lines):
                    np.testing.assert_array_equal(line.get_xdata(), plot.ALPHA)
                    np.testing.assert_array_equal(line.get_ydata(), original.get_ydata())
                    self.assertEqual(line.get_label(), original.get_label())
                    self.assertEqual(line.get_color(), original.get_color())
                    self.assertEqual(line.get_linestyle(), original.get_linestyle())
                    self.assertEqual(line.get_marker(), original.get_marker())
                    self.assertEqual(line.get_linewidth(), original.get_linewidth())
                    self.assertEqual(line.get_markersize(), original.get_markersize())
                legend = ax.get_legend()
                self.assertEqual([text.get_text() for text in legend.get_texts()],
                                 ["8B T2I", "8B I2T", "Mistral T2I", "Mistral I2T"])
                self.assertFalse(legend.get_in_layout())
                self.assertTrue(all(text.get_fontsize() == 16 for text in legend.get_texts()))
                plot.plt.close(fig)
                plot.plt.close(ordinary_fig)

    def test_compact_flickr_is_clear_at_half_column_print_size_and_both_dpis(self):
        for metric, ylabel in ((10, "Recall@10"), ("average", "Average Recall")):
            for dpi in (100, 300):
                with self.subTest(metric=metric, dpi=dpi):
                    panel, title, series = plot.build_panels(metric)[0]
                    fig, ax = plot.create_panel_figure(panel, title, series, ylabel, compact=True)
                    fig.set_dpi(dpi)
                    fig.canvas.draw()
                    renderer = fig.canvas.get_renderer()
                    legend = ax.get_legend()
                    items = [ax.title, ax.xaxis.label, ax.yaxis.label, legend]
                    ticks = [text for value, text in zip(ax.get_xticks(), ax.get_xticklabels())
                             if ax.get_xlim()[0] <= value <= ax.get_xlim()[1]]
                    ticks += [text for value, text in zip(ax.get_yticks(), ax.get_yticklabels())
                              if ax.get_ylim()[0] <= value <= ax.get_ylim()[1]]
                    items += ticks
                    boxes = [item.get_window_extent(renderer) for item in items]
                    for index, box in enumerate(boxes):
                        self.assertGreaterEqual(box.x0, fig.bbox.x0)
                        self.assertGreaterEqual(box.y0, fig.bbox.y0)
                        self.assertLessEqual(box.x1, fig.bbox.x1)
                        self.assertLessEqual(box.y1, fig.bbox.y1)
                        for other in boxes[index + 1:]:
                            self.assertFalse(box.overlaps(other))
                    legend_box = legend.get_window_extent(renderer).padded(2 * dpi / 72)
                    for line in ax.lines:
                        path = line.get_path().transformed(line.get_transform())
                        self.assertFalse(path.intersects_bbox(legend_box, filled=False))
                    content_box = fig.get_tightbbox(renderer, bbox_extra_artists=[legend])
                    width_cm = (content_box.width + 0.1) * 2.54
                    self.assertGreaterEqual(width_cm, 8.4)
                    self.assertLessEqual(width_cm, 8.7)
                    scale = 4.1 / width_cm
                    for text in [ax.title, ax.xaxis.label, ax.yaxis.label] + ticks + legend.get_texts():
                        self.assertGreaterEqual(text.get_fontsize() * scale, 7.5, text.get_text())
                    # The default tight crop would omit the out-of-layout legend.
                    self.assertLess(content_box.y0, fig.get_tightbbox(renderer).y0)
                    plot.plt.close(fig)

    def test_paper_average_preserves_data_and_fits_internal_legends_at_both_dpis(self):
        expected_legends = {
            "flickr": ["LLaMA T2I", "LLaMA I2T", "Mistral T2I", "Mistral I2T"],
            "person_retrieval": ["RSTPReid", "CUHK-PEDES", "LLaMA", "Mistral"],
        }
        for panel, title, series in plot.build_panels("average")[:2]:
            for dpi in (100, 300):
                with self.subTest(panel=panel, dpi=dpi):
                    fig, ax = plot.create_panel_figure(panel, title, series, "Average Recall", paper_layout=True)
                    np.testing.assert_allclose(fig.get_size_inches() * 72, (526.990625, 354.5648879717))
                    bounds = np.array(ax.get_position().bounds)
                    bounds[[0, 2]] *= fig.get_figwidth() * 72
                    bounds[[1, 3]] *= fig.get_figheight() * 72
                    np.testing.assert_allclose(bounds, (76.990625, 67.834375, 446.4, 277.2))
                    self.assertEqual(ax.get_title(), "")
                    self.assertEqual(ax.get_xlabel(), "Dense weight α")
                    self.assertEqual(ax.get_ylabel(), "Average Recall")
                    self.assertEqual(ax.xaxis.label.get_fontsize(), 25)
                    self.assertEqual(ax.yaxis.label.get_fontsize(), 25)
                    self.assertEqual(ax.get_xlim(), (0.05, 0.95))
                    self.assertEqual(ax.get_ylim(), (64, 92) if panel == "flickr" else (0, 32))
                    np.testing.assert_array_equal(ax.get_xticks(), plot.ALPHA[::2])
                    self.assertEqual(len(ax.lines), 4)
                    for line, (label, (xs, ys, color, linestyle, marker)) in zip(ax.lines, series.items()):
                        np.testing.assert_array_equal(line.get_xdata(), xs)
                        np.testing.assert_array_equal(line.get_ydata(), ys)
                        self.assertEqual((line.get_label(), line.get_color(), line.get_linestyle(), line.get_marker()),
                                         (label, color, linestyle, marker))
                        self.assertEqual(line.get_linewidth(), 3)
                        self.assertEqual(line.get_markersize(), 8)
                    for grid in ax.get_xgridlines() + ax.get_ygridlines():
                        self.assertEqual(grid.get_color(), "#b0b0b0")
                        self.assertEqual(grid.get_alpha(), 1)
                        self.assertEqual(grid.get_linewidth(), 0.8)
                    legend = ax.get_legend()
                    self.assertEqual([text.get_text() for text in legend.get_texts()], expected_legends[panel])
                    self.assertEqual(legend._ncols, 2)
                    if panel == "person_retrieval":
                        handles = getattr(legend, "legend_handles", None)
                        if handles is None:
                            handles = legend.legendHandles
                        self.assertEqual([handle.get_marker() for handle in handles[:2]], ["o", "s"])
                        self.assertEqual([handle.get_color() for handle in handles[2:]],
                                         [plot.COLORS["llava_8b"], plot.COLORS["llava_mistral"]])
                        self.assertEqual([handle.get_linestyle() for handle in handles[2:]], ["-", "--"])
                    fig.set_dpi(dpi)
                    fig.canvas.draw()
                    renderer = fig.canvas.get_renderer()
                    box = legend.get_window_extent(renderer)
                    axes_box = ax.get_window_extent(renderer)
                    self.assertGreaterEqual(box.x0, axes_box.x0)
                    self.assertGreaterEqual(box.y0, axes_box.y0)
                    self.assertLessEqual(box.x1, axes_box.x1)
                    self.assertLessEqual(box.y1, axes_box.y1)
                    for line in ax.lines:
                        path = line.get_path().transformed(line.get_transform())
                        # Include the 4pt marker radius plus a 2pt clearance.
                        self.assertFalse(path.intersects_bbox(box.padded(6 * dpi / 72), filled=False))
                    ticks = ax.get_xticklabels() + ax.get_yticklabels()
                    for text in ticks:
                        self.assertEqual(text.get_fontsize(), 30)
                    for text in legend.get_texts():
                        self.assertEqual(text.get_fontsize(), 25)
                    texts = [ax.xaxis.label, ax.yaxis.label] + ticks + legend.get_texts()
                    boxes = []
                    for text in texts:
                        self.assertGreaterEqual(text.get_fontsize(), 25)
                        bounds = text.get_window_extent(renderer)
                        self.assertGreaterEqual(bounds.x0, 0)
                        self.assertGreaterEqual(bounds.y0, 0)
                        self.assertLessEqual(bounds.x1, fig.bbox.width)
                        self.assertLessEqual(bounds.y1, fig.bbox.height)
                        for previous in boxes:
                            self.assertFalse(bounds.overlaps(previous), text.get_text())
                        boxes.append(bounds)
                    plot.plt.close(fig)

    def test_selected_average_exports_have_identical_page_sizes_and_leave_webqa_untouched(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            other_path = directory / "hybrid_weight_recall_webqa_average.pdf"
            other_path.write_bytes(b"existing WebQA output")
            previous_cwd = Path.cwd()
            try:
                os.chdir(directory)
                with contextlib.redirect_stdout(io.StringIO()), plot.plt.rc_context({"savefig.bbox": "tight"}):
                    plot.save_single_panels("average", "Average Recall", panels=plot.PAPER_AVERAGE_PANELS)
            finally:
                os.chdir(previous_cwd)
            self.assertEqual(other_path.read_bytes(), b"existing WebQA output")
            self.assertEqual({path.name for path in directory.iterdir()}, {
                "hybrid_weight_recall_flickr_average.pdf",
                "hybrid_weight_recall_person_retrieval_average.pdf",
                "hybrid_weight_recall_webqa_average.pdf",
            })
            expected_bounds = [0, 0, 526.990625, 354.5648879717]
            for panel in plot.PAPER_AVERAGE_PANELS:
                contents = (directory / f"hybrid_weight_recall_{panel}_average.pdf").read_bytes()
                box = re.search(rb"/MediaBox\s*\[([^\]]+)\]", contents)
                self.assertIsNotNone(box)
                np.testing.assert_allclose([float(value) for value in box.group(1).split()], expected_bounds)
                self.assertIn(b"/FontFile2", contents)
                self.assertNotIn(b"/Subtype /Image", contents)
        self.assertFalse(plot.plt.get_fignums())

    def test_save_failure_closes_the_figure(self):
        with patch.object(Figure, "savefig", side_effect=OSError("cannot write PDF")):
            with self.assertRaisesRegex(OSError, "cannot write PDF"):
                plot.save_single_panels(1, "Recall@1")
        self.assertFalse(plot.plt.get_fignums())

    def test_plot_failure_closes_the_figure(self):
        with patch.object(plot, "plot_panel", side_effect=ValueError("bad curve")):
            with self.assertRaisesRegex(ValueError, "bad curve"):
                plot.save_single_panels(1, "Recall@1")
        self.assertFalse(plot.plt.get_fignums())

    def test_legends_do_not_cross_curves(self):
        for metric, ylabel in METRICS:
            for panel, title, series in plot.build_panels(metric):
                with self.subTest(metric=metric, panel=panel):
                    fig, ax = plot.create_panel_figure(panel, title, series, ylabel)
                    fig.canvas.draw()
                    box = ax.get_legend().get_window_extent(fig.canvas.get_renderer()).padded(2)
                    for line in ax.lines:
                        path = line.get_path().transformed(line.get_transform())
                        self.assertFalse(path.intersects_bbox(box, filled=False), line.get_label())
                    plot.plt.close(fig)

    def test_single_panel_titles_axes_and_legends_fit_the_canvas(self):
        for metric, ylabel in METRICS:
            for panel, title, series in plot.build_panels(metric):
                with self.subTest(metric=metric, panel=panel):
                    fig, ax = plot.create_panel_figure(panel, title, series, ylabel)
                    fig.canvas.draw()
                    renderer = fig.canvas.get_renderer()
                    items = [ax.title, ax.xaxis.label, ax.yaxis.label, ax.get_legend()]
                    # Auto locators also expose ticks beyond the view limits;
                    # Matplotlib does not draw those, even if get_visible() is true.
                    items += [label for value, label in zip(ax.get_xticks(), ax.get_xticklabels())
                              if ax.get_xlim()[0] <= value <= ax.get_xlim()[1]]
                    items += [label for value, label in zip(ax.get_yticks(), ax.get_yticklabels())
                              if ax.get_ylim()[0] <= value <= ax.get_ylim()[1]]
                    for item in items:
                        if not item.get_visible():
                            continue
                        box = item.get_window_extent(renderer)
                        self.assertGreaterEqual(box.x0, fig.bbox.x0 - 1)
                        self.assertGreaterEqual(box.y0, fig.bbox.y0 - 1)
                        self.assertLessEqual(box.x1, fig.bbox.x1 + 1)
                        self.assertLessEqual(box.y1, fig.bbox.y1 + 1)
                    plot.plt.close(fig)


if __name__ == "__main__":
    unittest.main()
