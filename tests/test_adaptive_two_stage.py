"""CPU-only checks for stopping adaptive retrieval after its second stage."""
import argparse
import ast
from contextlib import contextmanager, nullcontext
import copy
import csv
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace as NS

import pytest

from test_adaptive_rgp import (BASH, CASES, CGP_NO_DEFAULTS, ROOT, atomic_json,
                               batch, load, pipeline)


PREFIX = "two_stage_retrieval_pipeline_adaptive_views"
IRRELEVANT = dict(RANKER_DTYPE="invalid", RANKER_LLAVA_IMAGE_MODE="invalid",
                  RANKER_QWEN_MIN_PIXELS="bad", RANKER_QWEN_MAX_PIXELS="-1",
                  RERANK_NUM="-1", RERANK_BATCH_SIZE="0", RERANK_TEMPLATE="invalid",
                  RANKER_MODEL_PATH="missing/model", USE_RERANK_PRIOR_CACHE="true")


@pytest.mark.parametrize("stage2", ["hybrid", "dense"])
@pytest.mark.parametrize("group,dataset,query,task", CASES)
def test_jobs_ignore_rankers_and_reuse_candidate_artifacts(batch, stage2, group, dataset, query, task):
    settings = batch.settings(group, dataset, query)
    settings.controls["STAGE2_METHOD"] = stage2
    original, _ = batch.api.prepare_jobs(settings, backend=batch.backend, auto_modify=True)
    settings.stop_after_stage2 = True
    settings.rankers = {}  # Any access to ranker resolution is a failure.
    settings.controls.update(IRRELEVANT)
    jobs, missing = batch.api.prepare_jobs(settings, backend=batch.backend, auto_modify=True,
                                            rerank_template="invalid", force_no_modify=True)
    assert not missing
    job, = jobs
    assert job.configuration["execution_mode"] == "two_stage"
    for key in ("ranker", "rerank_template", "modify_type", "use_rerank_prior_cache"):
        assert key not in job.configuration
    assert not any(k.startswith(("RANKER_", "RERANK_")) for k in job.variables)
    for key in ("DENSE_CORPUS_DIR", "SPARSE_CORPUS_DIR", "STAGE1_SPARSE_INDEX", "VIEW_SELECTION_FILE"):
        assert job.variables[key] == original[0].variables[key]
    assert job.fingerprint != original[0].fingerprint and job.output != original[0].output
    settings.controls.update({k: "another invalid value" for k in IRRELEVANT})
    again, _ = batch.api.prepare_jobs(settings, backend=batch.backend)
    assert again[0].fingerprint == job.fingerprint and again[0].output == job.output


def test_settings_drop_inactive_options_before_validation(batch):
    namespace = dict(batch.namespace, math=__import__("math"), dataclass=dataclass, PROJECT=ROOT,
                     VALIDATION_INPUTS=(),
                     CHECKPOINTS={"llava_mistral": "model"}, GROUPS={"ir": {"datasets": ("flickr",)}},
                     configured_list=lambda env, one, many, allowed, default: tuple(env.get(one, "").split()) or tuple(default),
                     split_name=lambda value, _: value,
                     encoding_controls=lambda env: dict(batch.settings().controls, SHUFFLE_VIEW_OPTIONS="false"),
                     selection_options=lambda *a: NS(view_overlap_selection=False))
    source = (ROOT / "src/adaptive_retrieval_batch.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    body = [n for n in tree.body if (isinstance(n, ast.ImportFrom) and n.module == "__future__")
            or (isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in {"Settings", "pipeline_controls"})]
    exec(compile(ast.Module(body=body, type_ignores=[]), "settings", "exec"), namespace)
    env = dict(IRRELEVANT, MODEL_FAMILY="llava_mistral", DATASET="flickr", IR_QUERY_TYPE="image")
    settings = namespace["Settings"].from_environment("ir", env, stop_after_stage2=True)
    assert settings.stop_after_stage2 and settings.rankers == {}
    assert settings.roots["results"].name == "two_stage_adaptive_results"
    assert settings.controls["RERANK_NUM"] != "-1"
    with pytest.raises(ValueError, match="RANKER_DTYPE"):
        namespace["Settings"].from_environment("ir", env)


@pytest.mark.parametrize("stage2", ["hybrid", "dense"])
@pytest.mark.parametrize("group,dataset,query,task", CASES)
def test_native_validation_ignores_all_inactive_stage3_settings(pipeline, monkeypatch, stage2, group, dataset, query, task):
    extra = ["--stop-after-stage2", "--stage2-method", stage2,
             "--ranker-run", "missing/ranker.json", "--ranker-model-name-or-path", "missing/model",
             "--ranker-qwen-min-pixels", "-5", "--ranker-qwen-max-pixels", "-10",
             "--rerank-num", "-1", "--rerank-batch-size", "0", "--rerank-template", "invalid",
             "--rerank-type", "listwise", "--modify-type", "modify_single", "--use-rerank-prior-cache"]
    args = pipeline.arguments(dataset, query, task, extra)
    # Simulate the shared preparation that otherwise tries to open ranker_run.
    monkeypatch.setattr(sys.modules["dense_representation"], "prepare_pipeline",
                        lambda args: args.ranker_run is None or pytest.fail("read inactive ranker run"))
    pipeline.api._validate_args(args)
    assert args.ranker_run is None
    assert pipeline.arguments(extra=["--stop_after_stage2"]).stop_after_stage2


def test_two_stage_preflight_still_checks_candidates(pipeline):
    (pipeline.dense / "corpus_1.pkl").touch()
    with pytest.raises(ValueError, match="shard count mismatch"):
        pipeline.api._validate_args(pipeline.arguments(extra=["--stop-after-stage2"]))


def test_two_stage_batch_skips_policy_helpers_and_cleans_environment(batch, monkeypatch, capsys):
    settings = batch.settings("fused", "webqa", "text")
    settings.stop_after_stage2 = True
    settings.rankers = {}
    settings.environment.update(IRRELEVANT)
    for name in ("launcher_rerank_template", "launcher_force_no_modify"):
        monkeypatch.setitem(batch.namespace, name, lambda *a: pytest.fail("read reranking policy"))
    launcher = ROOT / "scripts" / f"{PREFIX}_fused.sh"
    calls = []

    def runner(command, **kwargs):
        calls.append(command)
        assert not any(k.startswith(("RANKER_", "RERANK_")) for k in kwargs["env"])
        assert "USE_RERANK_PRIOR_CACHE" not in kwargs["env"]
        return NS(returncode=0)

    assert batch.api.run(settings, launcher=launcher, backend=batch.backend, runner=runner) == 0
    assert calls == [[settings.bash, launcher.as_posix(), "--run-one"]]
    assert "Stage 3 is skipped" in capsys.readouterr().out
    metadata, = batch.roots["results"].rglob("batch_job.json")
    assert json.loads(metadata.read_text())["configuration"]["execution_mode"] == "two_stage"
    with pytest.raises(ValueError, match="matching two_stage"):
        batch.api.run(settings, launcher=ROOT / "scripts/three_stage_retrieval_pipeline_adaptive_views_rgp_ir.sh")


@pytest.fixture
def execute(pipeline, monkeypatch, tmp_path, batch):
    """Execute the real main/reporting/fusion/alpha code with tiny in-memory retrieval results."""
    source = (ROOT / "src/three_stage_retrieval_pipeline_adaptive_views.py").read_text(encoding="utf-8")
    names = {"main", "StageTimer", "TimingReporter", "RecallReporter", "AlphaSweepReporter", "_format_duration",
             "_write_metrics_csv", "evaluate_recall", "write_json_run", "write_trec_run", "fuse_restricted_runs",
             "_compute_scope", "_top_ids", "_minmax", "_validate_query_coverage", "_alpha_objective"}
    validation_namespace = pipeline.api._parser.__globals__
    load("three_stage_retrieval_pipeline_adaptive_views.py", {"_alpha_validation_args"}, validation_namespace)
    namespace = dict(validation_namespace)
    namespace.update(csv=csv, time=time, json=json, contextmanager=contextmanager, nullcontext=nullcontext,
                     ALPHA_CANDIDATES=[round(i / 10, 1) for i in range(1, 10)],
                     _model_cuda_devices=lambda *a: (), _synchronize_cuda_devices=lambda *a: None,
                     stage2_output_path=lambda p, mode: p)
    progress = NS(n=0, total=0, closed=False)

    def show(iterable=None, **kwargs):
        if iterable is not None:
            return iterable
        progress.total = kwargs["total"]
        return progress

    progress.update = lambda count: setattr(progress, "n", progress.n + count)
    progress.close = lambda: setattr(progress, "closed", True)
    namespace.update(_progress=show, tqdm=NS(write=lambda *a: None))
    tree = ast.parse(source)
    body = [n for n in tree.body if (isinstance(n, ast.ImportFrom) and n.module == "__future__")
            or (isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names)]
    exec(compile(ast.Module(body=body, type_ignores=[]), "native_two_stage", "exec"), namespace)
    context = NS(is_main=True, is_distributed=False, world_size=1, rank=0)
    candidates = {"q": {"positive": 2.0, "negative": 1.0}}
    dense = {"q": {"negative": 3.0, "positive": 1.0}}
    sparse = {"q": {"positive": 5.0, "negative": 1.0}}
    encoded = object()
    calls = []

    def first(args, context, timer):
        calls.append("stage1")
        with timer.measure("retrieval"):
            return copy.deepcopy(candidates), encoded, copy.deepcopy(candidates), [1]

    def second(args, stage1, runtime, timing, encoded_queries):
        calls.append("stage2")
        assert encoded_queries is encoded and runtime is None
        with timing.measure("retrieval"):
            if args.stage2_method == "dense":
                return dense, {}, dense
            return dense, sparse, namespace["fuse_restricted_runs"](stage1, dense, sparse, args.hybrid_alpha, args.stage1_depth)

    def forbidden(*a, **k):
        pytest.fail("Loaded a model or entered Stage 3 in two-stage mode")

    namespace.update(_initialise_distributed=lambda *a: context,
                     _shutdown_distributed=lambda ctx, synchronize: calls.append(("shutdown", synchronize)),
                     _run_sparse_concat_stage1=first, run_stage2=second, run_stage3=forbidden,
                     _load_retrieval_runtime=forbidden, _load_ranker=forbidden, read_run=forbidden,
                     _release_retrieval_runtime=lambda runtime: calls.append("release"),
                     _gather_run=lambda run, ctx: run,
                     _alpha_validation_args=lambda args: None, _alpha_main_result=lambda ctx, fn: fn(),
                     _build_recall_evaluation=lambda *a: NS(targets={"q": {"positive"}}, query_categories={},
                                                           candidate_identities=None, k_values=[1, 5, 10], dataset_query_count=1),
                     _record_stage_timing=lambda stage, timer, ctx, reporter: reporter.record(stage, timer.snapshot(0)))
    monkeypatch.setattr(sys.modules["dense_representation"], "write_manifest", lambda *a: None, raising=False)
    for name in ("stc_sparse", "lens_sparse"):
        monkeypatch.setattr(sys.modules[name], "configuration", lambda *a: None, raising=False)
    selection = dict(batch.namespace["views"].load_selection(pipeline.arguments().view_selection_file),
                     selection_modality="both", selected={"text": [], "image": []})
    monkeypatch.setattr(batch.namespace["views"], "load_selection", lambda *a, **k: selection)
    monkeypatch.setitem(sys.modules, "adaptive_views", batch.namespace["views"])

    def run(dataset="flickr", query="image", task="ir", mode="hybrid", scan=False, stop=True, stale=False):
        extra = ["--stage2-method", mode, "--overwrite", "--hybrid-alpha", "0.7"]
        if stop:
            extra += ["--stop-after-stage2", "--rerank-template", "invalid", "--rerank-num", "-1"]
        if scan:
            extra += ["--select-hybrid-alpha"]
        args = pipeline.arguments(dataset, query, task, extra)
        args.output_dir = str(tmp_path / "result")
        if stale:
            Path(args.output_dir).mkdir()
            for name in ("stage3_ranker_run.json", "stage3_ranker_run.trec"):
                (Path(args.output_dir) / name).write_text("old stage3")
        namespace["_parser"] = lambda: NS(set_defaults=lambda **k: None, parse_args=lambda: args,
                                           error=lambda error: pytest.fail(error))
        namespace["main"]()
        return Path(args.output_dir)

    return NS(run=run, namespace=namespace, calls=calls, progress=progress, tmp_path=tmp_path)


@pytest.mark.parametrize("mode,scan", [("hybrid", False), ("hybrid", True), ("dense", False)])
@pytest.mark.parametrize("group,dataset,query,task", CASES)
def test_main_outputs_two_complete_stages_without_ranker(execute, mode, scan, group, dataset, query, task):
    output = execute.run(dataset, query, task, mode, scan, stale=True)
    names = {p.name for p in output.iterdir()}
    required = {"stage1_run.json", "stage1_run.trec", "stage2_dense_run.json", "stage2_dense_run.trec",
                "recall_metrics.json", "recall_metrics.csv", "timing_metrics.json", "timing_metrics.csv", "pipeline_metadata.json"}
    if mode == "hybrid":
        required |= {f"stage2_{variant}_run.{ext}" for variant in ("sparse", "hybrid") for ext in ("json", "trec")}
    if scan:
        required |= {"stage2_alpha_sweep.csv", "alpha_selection.json"}
        sweep = list(csv.DictReader((output / "stage2_alpha_sweep.csv").open(encoding="utf-8-sig")))
        assert {float(row["alpha"]) for row in sweep} == {round(i / 10, 1) for i in range(1, 10)}
    assert names == required
    metadata = json.loads((output / "pipeline_metadata.json").read_text())
    assert metadata["stage3"]["status"] == "skipped" and metadata["execution_mode"] == "two_stage"
    assert set(metadata["files"]) == names - {"pipeline_metadata.json"}
    recall = json.loads((output / "recall_metrics.json").read_text())
    assert {row["stage"] for row in recall["results"]} == {"stage1", "stage2"}
    timing = json.loads((output / "timing_metrics.json").read_text())
    assert timing["status"] == "completed" and timing["totals"]["compute_complete"]
    assert timing["totals"]["compute_seconds"] is not None
    assert [row["stage"] for row in timing["stages"]] == ["stage1", "stage2"]
    assert execute.progress.n == execute.progress.total == 2 and execute.progress.closed
    assert execute.calls == ["stage1", "stage2", "release", ("shutdown", True)]


def test_main_preserves_normal_three_stage_execution(execute):
    def rerank(args, run, context, timing):
        execute.calls.append("stage3")
        return run
    execute.namespace["run_stage3"] = rerank
    output = execute.run(stop=False)
    assert "stage3" in execute.calls
    assert (output / "stage3_ranker_run.json").is_file()
    assert execute.progress.n == execute.progress.total == 3
    metadata = json.loads((output / "pipeline_metadata.json").read_text())
    assert "execution_mode" not in metadata and metadata["stage3"]["rerank_template"] == "query_relevant"


def test_failed_stage2_still_finalizes_reports_and_cleanup(execute):
    def failure(*a, **k):
        raise RuntimeError("retrieval failed")
    execute.namespace["run_stage2"] = failure
    with pytest.raises(RuntimeError, match="retrieval failed"):
        execute.run()
    report = json.loads((execute.tmp_path / "result/timing_metrics.json").read_text())
    assert report["status"] == "failed" and report["stages"][1]["status"] == "failed"
    assert execute.calls[-2:] == ["release", ("shutdown", False)] and execute.progress.closed


@pytest.mark.parametrize("internal", [False, True])
@pytest.mark.parametrize("group,datasets", [("ir", "flickr"), ("tbpr", "CUHK-PEDES ICFG-PEDES RSTPReid"),
                                            ("fused", "webqa remuq llava")])
def test_shell_defaults_and_mode_forwarding(tmp_path, internal, group, datasets):
    if BASH is None:
        pytest.skip("Bash not installed")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "bin").mkdir()
    for name in (f"{PREFIX}_{group}.sh", "stc_sparse_mode.sh", "lens_sparse_mode.sh",
                 "stage2_retrieval_mode.sh", "freeret_dense_mode.sh"):
        shutil.copyfile(ROOT / "scripts" / name, tmp_path / "scripts" / name)
    defaults = {k: v for k, v in CGP_NO_DEFAULTS.items() if not k.startswith(("RANKER_", "RERANK_"))}
    defaults["ADAPTIVE_OUTPUT_ROOT"] = "/root/autodl-fs/two_stage_adaptive_results"
    stub = '#!/usr/bin/env bash\nprintf "%s\\0" "$@" > args\n'
    stub += 'for name in ' + ' '.join((*defaults, "DATASETS", "IR_QUERY_TYPES")) + '; do\n'
    stub += 'printf "%s=%s\\0" "$name" "${!name-unset}"\ndone > configuration\n'
    for name in ("python", "deepspeed"):
        path = tmp_path / "bin" / name
        with path.open("w", encoding="utf-8", newline="\n") as f:
            f.write(stub)
        path.chmod(0o755)
    env = dict(os.environ)
    for name in {*defaults, "MODEL_FAMILY", "MODEL_PATH", "DATASET", "IR_QUERY_TYPE", "OUTPUT_DIR"}:
        env.pop(name, None)
    env.update(IRRELEVANT, PYTHON_BIN="./bin/python", DEEPSPEED_BIN="./bin/deepspeed", MSYS_NO_PATHCONV="1")
    if internal:
        env.update(MODEL_FAMILY="llava_llama", MODEL_PATH="model", DATASET=datasets.split()[0],
                   VIEW_SELECTION_FILE="saved selection", DENSE_CORPUS_DIR="dense", SPARSE_CORPUS_DIR="sparse",
                   OUTPUT_DIR="result")
    bin_path = (tmp_path / "bin").as_posix()
    if os.name == "nt":
        bin_path = "/" + bin_path[0].lower() + bin_path[2:]
    command = 'export PATH=' + shlex.quote(bin_path) + ':"$PATH"; exec /usr/bin/bash scripts/'
    command += f"{PREFIX}_{group}.sh" + (" --run-one" if internal else "")
    result = subprocess.run([BASH, "-c", command], cwd=tmp_path, env=env, capture_output=True,
                            text=True, encoding="utf-8", timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    args = (tmp_path / "args").read_bytes().decode().rstrip("\0").split("\0")
    assert "--stop-after-stage2" in args
    assert not any(arg.startswith(("--ranker", "--rerank", "--modify", "--use-rerank")) for arg in args)
    configuration = dict(row.split("=", 1) for row in (tmp_path / "configuration").read_bytes().decode().rstrip("\0").split("\0"))
    if internal:
        assert "src/three_stage_retrieval_pipeline_adaptive_views.py" in args
        for name, expected in {"--retrieval-llava-image-mode": "base", "--retrieval-dtype": "float16",
                               "--stage1-depth": "200", "--stage2-depth": "200", "--stage2-method": "hybrid",
                               "--hybrid-alpha": "0.7", "--view-probability-mass": "true",
                               "--view-probability-mass-threshold": "0.9"}.items():
            assert args[args.index(name) + 1] == expected
    else:
        for name, expected in defaults.items():
            assert configuration[name] == expected, name
        assert args[:2] == ["-B", "src/adaptive_retrieval_batch.py"]
        assert args[args.index("--task-group") + 1] == group
        assert args[args.index("--launcher") + 1].endswith(f"/{PREFIX}_{group}.sh")
        assert configuration["DATASETS"] == datasets
