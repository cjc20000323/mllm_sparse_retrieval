"""Seed isolation, parity with choose_views, reports, and selection-only execution."""
from contextlib import redirect_stderr, redirect_stdout
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import adaptive_views as views
import adaptive_view_shuffle as shuffle
import adaptive_views_seed_experiment as experiment


IDENTITY = dict(family="qwen2_5_vl", checkpoint={"path": "fixture"})
TEXTS = [dict(id=f"t{i}", text=f"example {i}") for i in range(30)]
IMAGES = [dict(id=f"i{i}", image=f"image-{i}.jpg") for i in range(20)]
RUN_ID = "0123456789abcdef"


class Processor:
    def apply_chat_template(self, messages, **kwargs):
        return "<user>" + "".join("<image>" if part["type"] == "image" else part["text"]
            for part in messages[0]["content"]) + "<assistant>"


class Encoder:
    config = NS(model_type="qwen2_5_vl", text_config=NS(max_position_embeddings=64))

    def eval(self):
        return self


def prepared(processor, family, pool, items, modality, *, prior=False):
    prompt = views.render_choice(processor, family, pool,
        texts=[item["text"] for item in items] if modality == "text" and not prior else None,
        image_count=len(items) if modality == "image" else 0, prior=prior)
    return dict(inputs={}, tokens=10 + sum(2 + int(item["id"][1:]) % 4 for item in items), prompt=prompt)


def synthetic_score(model, processor, family, pool, items, modality, *, prior=False,
                    return_details=False, prepared=None, input_limit=None):
    # Candidate contents and actual sampled IDs affect scores, independent of their labels.
    source = "prior" if prior else modality + "/" + "/".join(item["id"] for item in items)
    logits = [int(hashlib.sha256((source + row["aspect"]).encode()).hexdigest()[:4], 16) / 16384
              for row in pool]
    normalizer = math.log(sum(math.exp(value) for value in logits))
    logp = [value - normalizer for value in logits]
    probabilities = [math.exp(value) for value in logp]
    if prepared is None:
        prepared = globals()["prepared"](processor, family, pool, items, modality, prior=prior)
    if return_details:
        return dict(probabilities=probabilities, log_probabilities=logp,
                    tokens=prepared["tokens"], prompt=prepared["prompt"])
    return probabilities, prepared["tokens"]


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


class SeedExperimentTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.base = Path(self.directory.name)
        self.runtime = NS(encoder=Encoder(), processor=Processor())

    def args(self, *extra):
        return experiment.validate_args(experiment.parser().parse_args([
            "--model-path", str(self.base / "checkpoint"), "--data-root", str(self.base),
            "--output-root", str(self.base / "results"), "--seeds", "0", "42",
            "--view-selection-rounds", "2", *extra]))

    def execute(self, args, *, scorer=synthetic_score):
        setter = Mock()
        with patch.object(views, "score_group", side_effect=scorer), \
                patch.object(views, "prepare_choice_group", side_effect=prepared), redirect_stdout(io.StringIO()):
            path = experiment.run_experiment(args, self.runtime, IDENTITY, TEXTS, IMAGES,
                                            views=views, seed_inference=setter)
        return path, setter

    def selections(self, path, seeds=(0, 42)):
        return [(seed, json.loads((path / f"seed_{seed}" / "view_selection.json").read_text(encoding="utf-8")))
                for seed in seeds]

    def test_defaults_validation_and_task_adapters(self):
        defaults = experiment.parser().parse_args([])
        self.assertEqual(defaults.seeds, [0, 1, 2, 3, 42])
        self.assertEqual((defaults.view_example_mode, defaults.view_selection_rounds,
                          defaults.view_texts_per_round, defaults.view_images_per_round), ("fixed", 20, 8, 2))
        self.assertEqual(self.args("--seeds", "42", "0", "42").seeds, [42, 0])
        for extra in (("--llava-image-mode", "base"), ("--view-selection-rounds", "0"),
                      ("--qwen-min-pixels", "100", "--qwen-max-pixels", "50"),
                      ("--view-texts-per-round", "0"), ("--inference-seed", "-1"),
                      ("--view-selection-run-id", RUN_ID)):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                self.args(*extra)
        self.args("--view-selection-modality", "image", "--view-texts-per-round", "0")
        import three_stage_retrieval_pipeline_adaptive_views as api
        for dataset, task in experiment.TASKS.items():
            args = self.args("--dataset", dataset, "--view-selection-split", "train")
            actual = experiment.pipeline_args(args, api)
            self.assertEqual((actual.task_type, actual.dataset_split), (task, "train"))
            self.assertEqual(actual.query_type, "image" if task == "it2t" else "text")
            self.assertTrue(actual.adaptive_views)

    def test_fixed_seeds_reproduce_and_keep_inference_seed_fixed(self):
        args = self.args()
        first, setter = self.execute(args)
        second, _ = self.execute(args)
        self.assertNotEqual(first, second)
        first_results = self.selections(first)
        self.assertEqual(first_results, self.selections(second))
        self.assertNotEqual(first_results[0][1]["rounds"][0]["text_ids"],
                            first_results[1][1]["rounds"][0]["text_ids"])
        self.assertNotEqual(first_results[0][1]["scores"], first_results[1][1]["scores"])
        self.assertEqual(setter.call_args_list, [unittest.mock.call(42), unittest.mock.call(42)])
        for seed, selection in first_results:
            for row in selection["rounds"]:
                self.assertEqual((len(row["text_ids"]), len(row["image_ids"])), (8, 2))
            self.assertEqual(selection["sampling"]["seed"], seed)
            # The JSON remains readable by the original artifact validator.
            self.assertEqual(views.load_selection(first / f"seed_{seed}" / "view_selection.json"), selection)

    def test_csv_probabilities_rounds_and_baseline_comparisons(self):
        path, _ = self.execute(self.args())
        selections = dict(self.selections(path))
        final = read_csv(path / "seed_view_probabilities.csv")
        self.assertEqual(len(final), 40)
        for row in final:
            selection = selections[int(row["seed"])]
            index = next(i for i, item in enumerate(selection["pool"]) if item["label"] == row["label"])
            text = sum(r["text_probabilities"][index] for r in selection["rounds"]) / 2
            image = sum(r["image_probabilities"][index] for r in selection["rounds"]) / 2
            self.assertAlmostEqual(float(row["probability"]), (text + image) / 2)
        self.assertEqual(len(read_csv(path / "round_probabilities.csv")), 2 * 2 * 2 * 20)
        stats = read_csv(path / "view_stability.csv")
        self.assertAlmostEqual(sum(float(row["top1_frequency"]) for row in stats), 1)
        self.assertAlmostEqual(sum(float(row["top5_frequency"]) for row in stats), 5)
        for row in stats:
            index = next(i for i, item in enumerate(selections[0]["pool"]) if item["label"] == row["label"])
            a, b = (selections[seed]["scores"][index] for seed in (0, 42))
            self.assertAlmostEqual(float(row["probability_mean"]), (a + b) / 2)
            self.assertAlmostEqual(float(row["probability_std"]), abs(a - b) / 2)
        comparison = read_csv(path / "seed_comparison.csv")
        self.assertEqual(comparison[0]["ranking_changed"], "False")
        self.assertEqual(float(comparison[0]["max_probability_delta"]), 0)
        deltas = [abs(a - b) for a, b in zip(selections[0]["scores"], selections[42]["scores"])]
        self.assertAlmostEqual(float(comparison[1]["max_probability_delta"]), max(deltas))
        self.assertAlmostEqual(float(comparison[1]["mean_absolute_probability_delta"]), sum(deltas) / 20)

    def test_comparison_distinguishes_top5_membership_from_order(self):
        pool = views.view_pool()[:6]
        def selection(scores):
            return dict(pool=pool, scores=scores, selected=pool[:5], rounds=[], text_mean=None, image_mean=None)
        result = experiment.reports([
            (42, selection([0.30, 0.25, 0.20, 0.12, 0.08, 0.05])),
            (7, selection([0.25, 0.30, 0.20, 0.12, 0.08, 0.05])),
            (0, selection([0.05, 0.25, 0.20, 0.12, 0.08, 0.30])),
        ])["seed_comparison"]
        self.assertEqual([row["baseline_seed"] for row in result], [42, 42, 42])
        self.assertEqual([row["top1"] for row in result], ["A", "B", "F"])
        self.assertEqual([row["top5_set_changed"] for row in result], [False, False, True])
        self.assertEqual([row["ranking_changed"] for row in result], [False, True, True])

    def test_saved_prompts_examples_and_image_references(self):
        path, _ = self.execute(self.args())
        selection = self.selections(path)[0][1]
        saved = json.loads((path / "seed_0" / "sampled_examples.json").read_text(encoding="utf-8"))
        self.assertEqual(len(saved), 4)
        for row in saved:
            original = selection["rounds"][row["round"] - 1]
            self.assertEqual(row["sample_ids"], original[row["modality"] + "_ids"])
            if row["modality"] == "text":
                for index, item in enumerate(row["examples"], 1):
                    self.assertIn(f"Example {index}: {item['text']}", row["prompt"])
            else:
                self.assertEqual(row["prompt"].count("<image>"), 2)
                self.assertTrue(all(item["image_reference"]["path"].endswith(".jpg") for item in row["examples"]))
        self.assertEqual(experiment.image_reference(b"abc")["sha256"], hashlib.sha256(b"abc").hexdigest())

    def test_pmi_hierarchical_and_one_shared_shuffle_mapping_match_original(self):
        args = self.args("--use-view-pmi", "true", "--use-hierarchical-views", "true",
                         "--shuffle-view-options", "true", "--view-selection-run-id", RUN_ID)
        path, _ = self.execute(args)
        stored = self.selections(path)
        self.assertEqual(stored[0][1][shuffle.FIELD], stored[1][1][shuffle.FIELD])
        self.assertEqual(len(stored[0][1]["pool"]), 10)
        for seed, selection in stored:
            options = experiment.selection_options(args, views)
            options.view_selection_seed = seed
            with patch.object(views, "score_group", side_effect=synthetic_score), redirect_stdout(io.StringIO()):
                original = views.choose_views(self.runtime.encoder, self.runtime.processor, IDENTITY,
                                             "flickr", "test", TEXTS, IMAGES, options)
            self.assertEqual(original, selection)
            views.load_selection(path / f"seed_{seed}" / "view_selection.json")
        rows = read_csv(path / "round_probabilities.csv")
        self.assertTrue(all(row["prior_probability"] and row["pmi_score"] for row in rows))
        mapping = {item["canonical_label"]: item["display_label"] for item in stored[0][1][shuffle.FIELD]["mapping"]}
        self.assertTrue(all(row["display_label"] == mapping[row["label"]] for row in rows))

    def test_auto_packing_budget_and_text_only_mode(self):
        path, _ = self.execute(self.args("--view-example-mode", "auto", "--view-example-max-tokens", "42",
                                        "--view-selection-modality", "text"))
        for seed, selection in self.selections(path):
            self.assertIsNone(selection["image_mean"])
            for row in selection["rounds"]:
                self.assertIsNone(row["image_probabilities"])
                self.assertLessEqual(row["text_tokens"], 41)
                self.assertGreater(row["text_examples"]["next_input_tokens"], 41)
            views.load_selection(path / f"seed_{seed}" / "view_selection.json")
        self.assertEqual({row["modality"] for row in read_csv(path / "round_probabilities.csv")}, {"text"})
        self.assertTrue(all(row["image_mean"] == "" for row in read_csv(path / "seed_view_probabilities.csv")))

    def test_single_seed_std_zero_and_fixed_counts_clip_to_available_pool(self):
        args = self.args("--seeds", "42", "--view-selection-modality", "image", "--view-images-per-round", "100")
        path, _ = self.execute(args)
        for row in read_csv(path / "view_stability.csv"):
            self.assertEqual(float(row["probability_std"]), 0)
            self.assertEqual(row["seed_count"], "1")
        selection = self.selections(path, (42,))[0][1]
        self.assertTrue(all(len(row["image_ids"]) == len(IMAGES) for row in selection["rounds"]))
        self.assertTrue(all(row["text_mean"] == "" for row in read_csv(path / "seed_view_probabilities.csv")))

    def test_failed_later_seed_preserves_completed_json_and_reports(self):
        args = self.args("--view-selection-rounds", "1", "--view-selection-modality", "text")
        def fail_second(*args, **kwargs):
            if scorer.call_count > 1:
                raise RuntimeError("fixture failure")
            return synthetic_score(*args, **kwargs)
        scorer = Mock(side_effect=fail_second)
        with self.assertRaisesRegex(RuntimeError, "fixture failure"):
            self.execute(args, scorer=scorer)
        manifest_path = next(args.output_root.rglob("experiment.json"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual((manifest["status"], manifest["completed_seeds"], manifest["failed_seed"]), ("failed", [0], 42))
        path = manifest_path.parent
        self.assertTrue((path / "seed_0" / "view_selection.json").is_file())
        self.assertEqual(len(read_csv(path / "seed_view_probabilities.csv")), 20)
        self.assertEqual({row["seed"] for row in read_csv(path / "seed_comparison.csv")}, {"0"})

    def test_real_run_loads_model_and_source_pools_once_without_encoding_or_retrieval(self):
        import adaptive_view_encoding as adaptive
        import three_stage_retrieval_pipeline_adaptive_views as pipeline
        args = self.args()
        api = NS(_parser=pipeline._parser, _load_retrieval_runtime=Mock(return_value=self.runtime),
                 _release_retrieval_runtime=Mock())
        with patch.object(adaptive, "pipeline_module", return_value=api), \
                patch.object(adaptive, "configure_processor", return_value=self.runtime.processor), \
                patch.object(adaptive, "source_pools", return_value=(TEXTS, IMAGES)) as source, \
                patch.object(adaptive, "run_encode", side_effect=AssertionError("No corpus encoding")), \
                patch.object(views, "model_identity", return_value=IDENTITY), \
                patch.object(views, "score_group", side_effect=synthetic_score), \
                patch("transformers.set_seed") as setter, redirect_stdout(io.StringIO()):
            output = experiment.run(args)
        api._load_retrieval_runtime.assert_called_once()
        api._release_retrieval_runtime.assert_called_once_with(self.runtime)
        source.assert_called_once()
        self.assertTrue(all(call.args == (42,) for call in setter.call_args_list))
        self.assertEqual(len(self.selections(output)), 2)

    def test_dry_run_validates_paths_without_model_import_or_output_writes(self):
        args = self.args()
        args.model_path.mkdir()
        (args.model_path / "config.json").write_text("{}", encoding="utf-8")
        annotation = self.base / "data" / "flickr" / "flickr_test.csv"
        annotation.parent.mkdir(parents=True)
        annotation.write_text("fixture", encoding="utf-8")
        argv = ["--model-path", str(args.model_path), "--data-root", str(self.base),
                "--output-root", str(args.output_root), "--dry-run"]
        with patch.object(experiment, "run", side_effect=AssertionError("Must not load model")), \
                redirect_stdout(io.StringIO()) as output:
            self.assertEqual(experiment.main(argv), 0)
        self.assertEqual(json.loads(output.getvalue())["baseline_seed"], 0)
        self.assertFalse(args.output_root.exists())
        (args.model_path / "config.json").unlink()
        with redirect_stderr(io.StringIO()) as error:
            self.assertEqual(experiment.main(argv), 2)
        self.assertIn("config.json", error.getvalue())

    def test_shell_syntax_and_dry_run_forward_configuration(self):
        candidates = [shutil.which("bash")]
        git = shutil.which("git")
        if git:
            candidates.append(str(Path(git).resolve().parents[1] / "bin" / "bash.exe"))
        bash = next((value for value in candidates if value and Path(value).is_file()), None)
        if not bash:
            self.skipTest("Bash is unavailable")
        args = self.args()
        args.model_path.mkdir()
        (args.model_path / "config.json").write_text("{}", encoding="utf-8")
        annotation = self.base / "data" / "flickr" / "flickr_test.csv"
        annotation.parent.mkdir(parents=True)
        annotation.write_text("fixture", encoding="utf-8")
        script = ROOT / "scripts" / "adaptive_views_seed_experiment.sh"
        syntax = subprocess.run([bash, "-n", str(script)], capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(syntax.returncode, 0, syntax.stderr)
        env = dict(os.environ, PYTHON_BIN=Path(sys.executable).as_posix(), MODEL_FAMILY="llava_mistral",
            MODEL_PATH=args.model_path.as_posix(), DATASET="flickr", DATASET_SPLIT="test", VIEW_SELECTION_SPLIT="",
            DATA_ROOT=self.base.as_posix(), OUTPUT_ROOT=args.output_root.as_posix(), LORA_MODEL_PATH="",
            VIEW_SELECTION_SEEDS="42 7 42", VIEW_SELECTION_ROUNDS="3", VIEW_TEXTS_PER_ROUND="4",
            VIEW_IMAGES_PER_ROUND="1", VIEW_EXAMPLE_MODE="fixed", VIEW_SELECTION_MODALITY="both",
            USE_HIERARCHICAL_VIEWS="true", USE_VIEW_PMI="true", SHUFFLE_VIEW_OPTIONS="true",
            VIEW_SELECTION_RUN_ID=RUN_ID, LLAVA_IMAGE_MODE="base", DRY_RUN="true")
        result = subprocess.run([bash, str(script)], capture_output=True, text=True, encoding="utf-8",
                                cwd=ROOT, env=env)
        self.assertEqual(result.returncode, 0, result.stderr)
        config = json.loads(result.stdout)["configuration"]
        self.assertEqual((config["model_family"], config["seeds"], config["view_selection_rounds"]),
                         ("llava_mistral", [42, 7], 3))
        self.assertEqual((config["view_texts_per_round"], config["view_images_per_round"]), (4, 1))
        self.assertTrue(config["shuffle_view_options"] and config["use_hierarchical_views"] and config["use_view_pmi"])
        self.assertFalse(args.output_root.exists())


if __name__ == "__main__":
    unittest.main()
