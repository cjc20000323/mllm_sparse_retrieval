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
from scipy.stats import gaussian_kde


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("plot_prior_log_likelihood", PROJECT_ROOT / "plot_prior_log_likelihood.py")
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Cannot import plot_prior_log_likelihood.py")
plot = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = plot
SPEC.loader.exec_module(plot)
SAMPLE = np.array([-3.0, -4.0, -2.0, -5.0, -6.0, -4.0, -4.0, -7.0, -3.0, -11.8])


class PriorDistributionPlotTests(unittest.TestCase):
    def tearDown(self):
        plot.plt.close("all")

    def write_cache(self, folder, payload):
        path = Path(folder) / "prior.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_load_negates_nll_and_keeps_duplicate_texts_zero_and_extreme_values(self):
        payload = {"entries": [{"key": str(i), "text_sha256": "same", "mean_nll": score}
                               for i, score in enumerate((2, 4, 2, 0, 30))]}
        with tempfile.TemporaryDirectory() as folder:
            path = self.write_cache(folder, payload)
            original = path.read_bytes()
            np.testing.assert_array_equal(plot.load_mean_log_likelihoods(path), [-2, -4, -2, 0, -30])
            self.assertEqual(path.read_bytes(), original)

    def test_invalid_cache_structure_and_empty_entries_raise(self):
        for payload in ([], {}, {"entries": {}}, {"entries": []}, {"entries": [None]}, {"entries": [{}]}):
            with self.subTest(payload=payload), tempfile.TemporaryDirectory() as folder:
                path = self.write_cache(folder, payload)
                with self.assertRaises(ValueError):
                    plot.load_mean_log_likelihoods(path)

    def test_invalid_nll_is_not_silently_skipped(self):
        for value in (None, True, False, "2.0", [], -0.1, float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as folder:
                path = self.write_cache(folder, {"entries": [{"mean_nll": 2}, {"mean_nll": value}]})
                with self.assertRaisesRegex(ValueError, "Entry 2: mean_nll"):
                    plot.load_mean_log_likelihoods(path)

    def test_degenerate_or_nonfinite_samples_fail_before_figure_allocation(self):
        for values in ([], [-2], [-2, -2], [0, 0], [float("nan"), -2], [-2, float("inf")], [[-1, -2]]):
            with self.subTest(values=values):
                before = plot.plt.get_fignums()
                with self.assertRaises(ValueError):
                    plot.prepare_distribution(values)
                self.assertEqual(plot.plt.get_fignums(), before)

    def test_histogram_matches_fd_edges_and_integrates_to_one_without_mutation(self):
        original = SAMPLE.copy()
        result = plot.prepare_distribution(SAMPLE)
        edges = np.histogram_bin_edges(SAMPLE, bins="fd")
        density, _ = np.histogram(SAMPLE, bins=edges, density=True)
        np.testing.assert_array_equal(result.bin_edges, edges)
        np.testing.assert_array_equal(result.histogram_density, density)
        self.assertAlmostEqual(float(np.sum(density * np.diff(edges))), 1)
        np.testing.assert_array_equal(SAMPLE, original)
        np.testing.assert_array_equal(result.values, SAMPLE)

    def test_kde_uses_scott_bandwidth_and_fixed_display_axes(self):
        result = plot.prepare_distribution(SAMPLE)
        expected = gaussian_kde(SAMPLE, bw_method="scott")
        self.assertEqual(result.grid.size, 1024)
        np.testing.assert_allclose(result.kde_density, expected(result.grid))
        self.assertTrue(np.isfinite(result.kde_density).all())
        self.assertTrue((result.kde_density >= 0).all())
        self.assertAlmostEqual(result.bandwidth, float(np.sqrt(expected.covariance[0, 0])))
        self.assertEqual(result.xlim, (-10.0, -2.0))
        self.assertEqual(result.grid[0], result.xlim[0])
        self.assertEqual(result.grid[-1], result.xlim[1])
        self.assertEqual(result.ylim, (0, max(result.histogram_density.max(), result.kde_density.max()) * 1.12))
        self.assertAlmostEqual(float(np.trapz(result.kde_density, result.grid)),
                               expected.integrate_box_1d(*result.xlim), delta=1e-5)

    def test_fixed_display_range_does_not_filter_or_renormalize_observations(self):
        values = np.array([-15.0, -11.0, -8.0, -6.0, -4.0, -1.0, 0.0])
        result = plot.prepare_distribution(values)
        np.testing.assert_array_equal(result.values, values)
        self.assertLess(result.bin_edges[0], result.xlim[0])
        self.assertGreater(result.bin_edges[-1], result.xlim[1])
        self.assertAlmostEqual(float(np.sum(result.histogram_density * np.diff(result.bin_edges))), 1)
        expected = gaussian_kde(values, bw_method="scott")
        np.testing.assert_allclose(result.kde_density, expected(result.grid))
        self.assertAlmostEqual(result.bandwidth, float(np.sqrt(expected.covariance[0, 0])))
        # The visible part must not be rescaled to integrate to one.
        self.assertLess(float(np.trapz(result.kde_density, result.grid)), 1)

    def test_figures_are_separate_and_keep_matching_axes_and_supplied_densities(self):
        result = plot.prepare_distribution(SAMPLE)
        hist_fig, hist_ax = plot.plot_distribution(result, "histogram")
        kde_fig, kde_ax = plot.plot_distribution(result, "kde")
        self.assertIsNot(hist_fig, kde_fig)
        self.assertFalse(hist_ax.lines)
        self.assertFalse(kde_ax.patches)
        self.assertEqual(len(hist_ax.patches), len(result.bin_edges) - 1)
        self.assertEqual(len(kde_ax.lines), 1)
        np.testing.assert_allclose([bar.get_height() for bar in hist_ax.patches], result.histogram_density)
        np.testing.assert_array_equal(kde_ax.lines[0].get_xdata(), result.grid)
        np.testing.assert_array_equal(kde_ax.lines[0].get_ydata(), result.kde_density)
        for fig, ax in ((hist_fig, hist_ax), (kde_fig, kde_ax)):
            self.assertEqual(list(fig.get_size_inches()), [3.6, 2.4])
            self.assertEqual(ax.get_xlim(), result.xlim)
            self.assertEqual(ax.get_ylim(), result.ylim)
            self.assertEqual(ax.get_xlabel(), "Average Log-likelihood")
            self.assertEqual(ax.get_ylabel(), "Density")
            self.assertEqual(ax.xaxis.label.get_fontsize(), 10.5)

    def test_axis_text_fits_canvas_at_preview_and_export_resolutions(self):
        result = plot.prepare_distribution(SAMPLE)
        for kind in plot.OUTPUT_STEMS:
            for dpi in (100, 300):
                with self.subTest(kind=kind, dpi=dpi):
                    fig, ax = plot.plot_distribution(result, kind)
                    fig.set_dpi(dpi)
                    fig.canvas.draw()
                    renderer = fig.canvas.get_renderer()
                    # Locators also return ticks outside the view; Matplotlib
                    # does not draw their labels, even when get_visible() is True.
                    texts = [ax.xaxis.label, ax.yaxis.label]
                    texts += [text for location, text in zip(ax.get_xticks(), ax.get_xticklabels())
                              if result.xlim[0] <= location <= result.xlim[1]]
                    texts += [text for location, text in zip(ax.get_yticks(), ax.get_yticklabels())
                              if result.ylim[0] <= location <= result.ylim[1]]
                    for text in texts:
                        if not text.get_visible() or not text.get_text():
                            continue
                        box = text.get_window_extent(renderer)
                        self.assertGreaterEqual(box.x0, fig.bbox.x0)
                        self.assertGreaterEqual(box.y0, fig.bbox.y0)
                        self.assertLessEqual(box.x1, fig.bbox.x1)
                        self.assertLessEqual(box.y1, fig.bbox.y1)
                    plot.plt.close(fig)

    def test_exports_four_separate_vector_pdf_and_300dpi_png_files_and_closes_figures(self):
        result = plot.prepare_distribution(SAMPLE)
        with tempfile.TemporaryDirectory() as folder:
            paths = plot.save_figures(result, Path(folder))
            self.assertEqual({path.name for path in paths}, {
                f"{stem}.{extension}" for stem in plot.OUTPUT_STEMS.values() for extension in ("pdf", "png")
            })
            for path in paths:
                if path.suffix == ".pdf":
                    content = path.read_bytes()
                    self.assertTrue(content.startswith(b"%PDF-"))
                    self.assertTrue(content.rstrip().endswith(b"%%EOF"))
                    self.assertEqual(len(re.findall(rb"/Type\s*/Page\b", content)), 1)
                    self.assertIn(b"/FontFile2", content)
                    self.assertNotIn(b"/Subtype /Image", content)
                else:
                    with Image.open(path) as image:
                        self.assertEqual(image.format, "PNG")
                        self.assertAlmostEqual(image.info["dpi"][0], 300, delta=0.1)
            self.assertFalse(plot.plt.get_fignums())

    def test_cli_accepts_output_directory_and_reports_population_statistics(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()) as stdout:
            path = self.write_cache(folder, {"entries": [{"mean_nll": float(-value)} for value in SAMPLE]})
            output = Path(folder) / "figures"
            plot.main([str(path), "--output-dir", str(output)])
            self.assertEqual(len(list(output.iterdir())), 4)
            self.assertIn("Observations: 10", stdout.getvalue())
            self.assertIn(f"Mean log-likelihood: {SAMPLE.mean():.8f}", stdout.getvalue())
            self.assertIn(f"Standard deviation (population): {SAMPLE.std():.8f}", stdout.getvalue())
            self.assertFalse(plot.plt.get_fignums())

    def test_cli_errors_do_not_create_output_files(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stderr(io.StringIO()):
            path = self.write_cache(folder, {"entries": [{"mean_nll": 2}]})
            output = Path(folder) / "not_created"
            with self.assertRaises(SystemExit) as error:
                plot.main([str(path), "--output-dir", str(output)])
            self.assertEqual(error.exception.code, 2)
            self.assertFalse(output.exists())

    def test_save_failure_closes_figure_and_unknown_plot_kind_allocates_nothing(self):
        result = plot.prepare_distribution(SAMPLE)
        with self.assertRaisesRegex(ValueError, "Plot kind"):
            plot.plot_distribution(result, "normal")
        self.assertFalse(plot.plt.get_fignums())
        with tempfile.TemporaryDirectory() as folder, patch.object(plot.plt.Figure, "savefig", side_effect=OSError("save failed")):
            with self.assertRaisesRegex(OSError, "save failed"):
                plot.save_figures(result, Path(folder))
            self.assertFalse(plot.plt.get_fignums())


if __name__ == "__main__":
    unittest.main()
