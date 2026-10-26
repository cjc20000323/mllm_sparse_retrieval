from __future__ import annotations

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import re
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("bias_with_prior_histogram", PROJECT_ROOT / "bias_with_prior_histogram.py")
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Cannot import bias_with_prior_histogram.py")
plot = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = plot
SPEC.loader.exec_module(plot)
SAMPLE = np.array([-3.0, -4.0, -2.0, -5.0, -6.0, -4.0, -4.0, -7.0, -3.0, -11.8])


class BiasWithPriorHistogramTests(unittest.TestCase):
    def tearDown(self):
        plot.plt.close("all")

    def test_left_panel_keeps_original_bar_scores_labels_colors_and_positions(self):
        _, (left, _) = plot.plot_bias_with_prior(plot.prepare_distribution(SAMPLE))
        expected = [[75.0, 83.2, 83.2, 80.1], [79.2, 69.4, 78.1, 68.4]]
        self.assertEqual(len(left.patches), 8)
        self.assertEqual([text.get_text() for text in left.get_xticklabels()],
                         ["RGP", "CGP", "Modified\nCGP", "FreeRet"])
        self.assertEqual(left.get_ylabel(), "r@1")
        for series, scores in enumerate(expected):
            bars = left.patches[series * 4:(series + 1) * 4]
            np.testing.assert_allclose([bar.get_height() for bar in bars], scores)
            for index, bar in enumerate(bars):
                self.assertAlmostEqual(bar.get_width(), 0.05)
                self.assertAlmostEqual(bar.get_x() + bar.get_width() / 2,
                                       index * 0.15 + (series + 0.5) * 0.05)
                self.assertEqual(bar.get_facecolor(), plot.matplotlib.colors.to_rgba(plot.COLORS[series]))
        self.assertEqual([text.get_text() for text in left.texts[:8]],
                         [str(score) for series in expected for score in series])

    def test_left_panel_keeps_hybrid_baselines_and_legend(self):
        _, (left, _) = plot.plot_bias_with_prior(plot.prepare_distribution(SAMPLE))
        self.assertEqual(len(left.lines), 2)
        for line, value, color, label in zip(left.lines, (70.6, 62.9), ("blue", "green"),
                                            ("i2t hybrid", "t2i hybrid")):
            np.testing.assert_array_equal(line.get_ydata(), [value, value])
            self.assertEqual(line.get_color(), color)
            self.assertEqual(line.get_linestyle(), "--")
            self.assertEqual(line.get_label(), label)
        self.assertEqual([text.get_text() for text in left.texts[8:]], ["70.6", "62.9"])
        self.assertEqual([text.get_text() for text in left.get_legend().get_texts()],
                         ["i2t hybrid", "t2i hybrid", "t2i", "i2t"])

    def test_right_panel_is_full_sample_density_histogram_without_kde_or_old_curve(self):
        result = plot.prepare_distribution(SAMPLE)
        _, (left, right) = plot.plot_bias_with_prior(result)
        self.assertFalse(right.lines)
        self.assertFalse(right.collections)
        self.assertEqual(len(right.patches), len(result.bin_edges) - 1)
        for bar in right.patches:
            self.assertEqual(bar.get_facecolor(), left.patches[0].get_facecolor())
            self.assertEqual(bar.get_alpha(), 1.0)
        np.testing.assert_allclose([bar.get_height() for bar in right.patches], result.histogram_density)
        np.testing.assert_allclose([bar.get_x() for bar in right.patches], result.bin_edges[:-1])
        self.assertAlmostEqual(sum(bar.get_width() * bar.get_height() for bar in right.patches), 1)
        self.assertEqual(right.get_xlim(), (-10.0, -2.0))
        self.assertEqual(right.get_ylim(), result.ylim)
        self.assertEqual(right.get_xlabel(), "Average Log-likelihood")
        self.assertEqual(right.get_ylabel(), "Density")
        self.assertLess(right.patches[0].get_x(), -10)  # Outside-view observations are retained.
        self.assertIsNone(right.get_legend())

    def test_original_two_panel_size_width_ratio_and_fonts_are_preserved(self):
        original_font_size = plot.matplotlib.rcParams["font.size"]
        fig, (left, right) = plot.plot_bias_with_prior(plot.prepare_distribution(SAMPLE))
        self.assertEqual(list(fig.get_size_inches()), [16, 6])
        self.assertEqual(len(fig.axes), 2)
        self.assertEqual(list(left.get_subplotspec().get_gridspec().get_width_ratios()), [1.4, 1])
        self.assertEqual(left.yaxis.label.get_fontsize(), 25)
        self.assertEqual(left.texts[0].get_fontsize(), 20)
        self.assertEqual(right.xaxis.label.get_fontsize(), 22)
        self.assertEqual(right.yaxis.label.get_fontsize(), 22)
        self.assertEqual(plot.matplotlib.rcParams["font.size"], original_font_size)

    def test_labels_fit_canvas_and_do_not_overlap_at_preview_and_export_resolutions(self):
        for dpi in (100, 300):
            with self.subTest(dpi=dpi):
                fig, axes = plot.plot_bias_with_prior(plot.prepare_distribution(SAMPLE))
                fig.set_dpi(dpi)
                fig.canvas.draw()
                renderer = fig.canvas.get_renderer()
                all_boxes = []
                for ax in axes:
                    texts = list(ax.texts) + [ax.xaxis.label, ax.yaxis.label]
                    texts += [text for loc, text in zip(ax.get_xticks(), ax.get_xticklabels())
                              if ax.get_xlim()[0] <= loc <= ax.get_xlim()[1]]
                    texts += [text for loc, text in zip(ax.get_yticks(), ax.get_yticklabels())
                              if ax.get_ylim()[0] <= loc <= ax.get_ylim()[1]]
                    for text in texts:
                        if not text.get_visible() or not text.get_text():
                            continue
                        box = text.get_window_extent(renderer)
                        self.assertGreaterEqual(box.x0, fig.bbox.x0, text.get_text())
                        self.assertGreaterEqual(box.y0, fig.bbox.y0, text.get_text())
                        self.assertLessEqual(box.x1, fig.bbox.x1, text.get_text())
                        self.assertLessEqual(box.y1, fig.bbox.y1, text.get_text())
                        all_boxes.append((text.get_text(), box))
                for index, (name, box) in enumerate(all_boxes):
                    for other_name, other in all_boxes[index + 1:]:
                        self.assertFalse(box.overlaps(other), f"{name} overlaps {other_name}")
                plot.plt.close(fig)

    def test_export_uses_new_names_without_overwriting_existing_outputs(self):
        fig, _ = plot.plot_bias_with_prior(plot.prepare_distribution(SAMPLE))
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            existing = {}
            for name in ("flickr_bias.pdf", "prior_log_likelihood_histogram.pdf", "prior_log_likelihood_kde.pdf"):
                path = output / name
                path.write_bytes(b"existing artifact")
                existing[path] = path.read_bytes()
            pdf, png = plot.save_figure(fig, output)
            self.assertEqual(pdf.name, "flickr_bias_with_prior_histogram.pdf")
            self.assertEqual(png.name, "flickr_bias_with_prior_histogram.png")
            for path, original in existing.items():
                self.assertEqual(path.read_bytes(), original)
            content = pdf.read_bytes()
            self.assertTrue(content.startswith(b"%PDF-"))
            self.assertTrue(content.rstrip().endswith(b"%%EOF"))
            self.assertEqual(len(re.findall(rb"/Type\s*/Page\b", content)), 1)
            self.assertIn(b"/FontFile2", content)
            self.assertNotIn(b"/Subtype /Image", content)
            with Image.open(png) as image:
                self.assertEqual(image.format, "PNG")
                self.assertAlmostEqual(image.info["dpi"][0], 300, delta=0.1)

    def test_cli_reads_json_without_mutation_reports_count_and_closes_figures(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()) as stdout:
            source = Path(folder) / "prior.json"
            source.write_text(json.dumps({"entries": [{"mean_nll": float(-value)} for value in SAMPLE]}), encoding="utf-8")
            original = source.read_bytes()
            output = Path(folder) / "figures"
            plot.main([str(source), "--output-dir", str(output)])
            self.assertEqual(len(list(output.iterdir())), 2)
            self.assertEqual(source.read_bytes(), original)
            self.assertIn("Prior observations: 10", stdout.getvalue())
            self.assertFalse(plot.plt.get_fignums())

    def test_invalid_json_data_creates_no_output(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stderr(io.StringIO()):
            source = Path(folder) / "bad.json"
            source.write_text('{"entries": []}', encoding="utf-8")
            output = Path(folder) / "not_created"
            with self.assertRaises(SystemExit) as error:
                plot.main([str(source), "--output-dir", str(output)])
            self.assertEqual(error.exception.code, 2)
            self.assertFalse(output.exists())
            self.assertFalse(plot.plt.get_fignums())

    def test_plotting_and_save_errors_close_figures(self):
        result = plot.prepare_distribution(SAMPLE)
        with patch.object(plot, "_plot_histogram", side_effect=ValueError("plot failed")):
            with self.assertRaisesRegex(ValueError, "plot failed"):
                plot.plot_bias_with_prior(result)
        self.assertFalse(plot.plt.get_fignums())
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "prior.json"
            source.write_text(json.dumps({"entries": [{"mean_nll": float(-value)} for value in SAMPLE]}), encoding="utf-8")
            with patch.object(plot, "save_figure", side_effect=OSError("save failed")):
                with self.assertRaisesRegex(OSError, "save failed"):
                    plot.main([str(source), "--output-dir", str(Path(folder) / "output")])
            self.assertFalse(plot.plt.get_fignums())


if __name__ == "__main__":
    unittest.main()
