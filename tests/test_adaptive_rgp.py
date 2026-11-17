"""Model-free checks for adaptive RGP/CGP-no launchers and job identity."""
import argparse
import ast
from contextlib import nullcontext
import copy
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS

import pytest


ROOT = Path(__file__).resolve().parents[1]
PREFIX = "three_stage_retrieval_pipeline_adaptive_views"
CASES = [("ir", "flickr", "image", "ir"), ("ir", "coco", "text", "ir"),
         ("tbpr", "CUHK-PEDES", "text", "tbpr"), ("fused", "webqa", "text", "t2it"),
         ("fused", "remuq", "image", "it2t"), ("fused", "llava", "image", "it2t")]
CGP_NO_DEFAULTS = dict(
    MODEL_FAMILIES="llava_llama", CHECKPOINT_ROOT="/root/autodl-fs/checkpoints", DATA_ROOT="/root/autodl-tmp",
    DENSE_OUTPUT_ROOT="/root/autodl-fs/dense_output", SPARSE_OUTPUT_ROOT="/root/autodl-fs/sparse_output",
    ADAPTIVE_OUTPUT_ROOT="/root/autodl-fs/three_stage_adaptive_results",
    LLAVA_IMAGE_MODE="base", RETRIEVAL_LLAVA_IMAGE_MODE="base", RANKER_LLAVA_IMAGE_MODE="base",
    DTYPE="float16", RETRIEVAL_DTYPE="float16", RANKER_DTYPE="float16", GPUS="2", NUM_NODES="1",
    USE_HIERARCHICAL_VIEWS="true", USE_VIEW_PROBABILITY_MASS="true", VIEW_PROBABILITY_MASS_THRESHOLD="0.9",
    VIEW_EXAMPLE_MODE="auto", VIEW_EXAMPLE_MAX_TOKENS="0", USE_VIEW_PMI="false", USE_VIEW_OVERLAP_SELECTION="false",
    USE_VIEW_TOKEN_VARIANTS="false", USE_VIEW_PREFIX_TOKENS="false",
    USE_STC_SPARSE="false", USE_LENS_SPARSE="false", USE_FREERET_DENSE="false",
    USE_FREERET_FULL_PROMPT="false", USE_FREERET_PROMPT_CONSTRAINTS="false",
    STAGE2_METHOD="hybrid", HYBRID_ALPHA="0.7", SELECT_HYBRID_ALPHA="false", STAGE1_DEPTH="200", STAGE2_DEPTH="200",
    RERANK_NUM="20", RERANK_BATCH_SIZE="20", RETRIEVAL_BATCH_SIZE="4", REPS_LOC="after_pad",
    SPARSE_LENGTH="30", TEXT_SPARSE_LENGTH="30", IMAGE_SPARSE_LENGTH="30", SPARSE_VALUE_TYPE="sum",
    SPARSE_LOWER_OR_UPPER="lower", SPARSE_VALUE_MEAN="false", IS_FILTERED="false", NUM_EXPENDED_TOKENS="0",
    QWEN_MIN_PIXELS="0", QWEN_MAX_PIXELS="0", DRY_RUN="false", OVERWRITE="true",
)
BASH = shutil.which("bash") if os.name != "nt" else next(
    (p for p in ("D:/Git/bin/bash.exe", "C:/Program Files/Git/bin/bash.exe") if Path(p).is_file()), None)


def load(filename, names, namespace, constants=(), source=None):
    path = ROOT / "src" / filename
    tree = ast.parse(source if source is not None else path.read_text(encoding="utf-8"))
    body = [n for n in tree.body if (
        isinstance(n, ast.ImportFrom) and n.module == "__future__") or (
        isinstance(n, ast.FunctionDef) and n.name in names) or (
        isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in constants for t in n.targets))]
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), namespace)
    return NS(**namespace)


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def batch(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "src"))
    for module in ("stc_sparse", "lens_sparse"):
        monkeypatch.setitem(sys.modules, module, NS(copy_options=lambda *a, **k: None,
                                                  mark_path=lambda p, config: p,
                                                  FIELD=module, validate_bundle=lambda *a: None))
    monkeypatch.setitem(sys.modules, "dense_representation", NS(
        configuration=lambda value, *a: value, mode_path=lambda path, *a: path,
        validate_artifact=lambda *a: None))
    views = load("adaptive_views.py", {"differences", "require_equal", "validate_artifact"}, {
        "Path": Path, "json": json, "digest": digest, "MANIFEST": "adaptive_views_manifest.json",
        "artifact_root": Path,
    })
    views.load_selection = lambda path, **k: json.loads(Path(path).read_text(encoding="utf-8"))
    views.is_overlap_selection = lambda selection: False
    views.checkpoint_identity = lambda path: {"path": str(path)}
    views.add_pool_mode = lambda value, mode: value
    views.pool_mode = lambda selection: None
    views.examples = NS(FIELD="example_sampling", add_configuration=lambda value, policy: value)
    views.atomic_json = atomic_json
    shuffle = NS(FIELD="option_shuffle", add_metadata=lambda value, policy: value, mark_path=lambda path, policy: path)
    mass = NS(is_selection=lambda selection: False, add_metadata=lambda value, selection: value)
    namespace = {"Path": Path, "json": json, "views": views, "option_shuffle": shuffle, "probability_mass": mass,
                 "SimpleNamespace": NS, "STC_NAMES": (), "LENS_NAMES": (), "os": os, "shlex": shlex,
                 "subprocess": subprocess, "sys": sys, "MissingInput": FileNotFoundError,
                 "annotation_files": lambda *a: [], "boolean": lambda value, *a: str(value).lower() == "true",
                 "resolve_sparse_index": lambda root, **k: root / "collection" / "index",
                 "resolved_path": lambda value, project, label: Path(value).resolve(),
                 "Job": lambda *values: NS(**dict(zip(("key", "output", "variables", "fingerprint", "configuration"), values)))}
    api = load("adaptive_retrieval_batch.py", {
        "job_types", "caption_modify_type", "launcher_rerank_template", "launcher_force_no_modify", "namespaces", "validate_corpus",
        "prepare_jobs", "report_summary", "run",
    }, namespace, ("FAMILIES", "RETRIEVAL_DEFAULTS", "PIPELINE_NAMES", "RGP_LAUNCHERS", "CGP_NO_LAUNCHERS"))
    with tempfile.TemporaryDirectory(prefix="rgp-") as directory:
        root = Path(directory)
        model = root / "model"
        model.mkdir()
        (model / "config.json").write_text("{}")
        roots = {name: root / name for name in ("dense", "sparse", "data", "results")}
        dense, sparse = roots["dense"] / "corpus", roots["sparse"] / "corpus"
        dense.mkdir(parents=True)
        (dense / "corpus_0.pkl").touch()
        collection = sparse / "collection"
        (collection / "index").mkdir(parents=True)
        (collection / "corpus_0.jsonl").touch()
        (collection / "index" / "segments_1").touch()
        selection = dict(fingerprint="selection", views_fingerprint="views", selection_split="test",
                         sampling={"rounds": 20, "seed": 42})
        selection_path = root / "selection.json"
        atomic_json(selection_path, selection)
        expected = dict(configuration={"method": ["disassembleeol_separate", "concat"],
                                       "selection_calibration": {"method": "pmi", "prior_prompt_mode": "instruction_only"}})
        expected["fingerprint"] = digest(expected["configuration"])
        for path in (dense, sparse):
            atomic_json(path / "adaptive_views_manifest.json", dict(
                **expected, num_shards=1, selection_fingerprint=selection["fingerprint"]))
        namespace["selection_location"] = lambda *a: (selection_path, {"sampling": selection["sampling"]}, None)
        backend = NS(load_model_identity=lambda *a, **k: {"family": "llava_mistral"},
                     artifact_dirs=lambda *a: (dense, sparse), metadata=lambda *a: copy.deepcopy(expected))
        controls = {name: "false" for name in api.PIPELINE_NAMES}
        controls.update(api.RETRIEVAL_DEFAULTS)
        controls.update(RETRIEVAL_DTYPE="float16", RANKER_DTYPE="float16", DTYPE="float16",
                        RETRIEVAL_LLAVA_IMAGE_MODE="anyres", RANKER_LLAVA_IMAGE_MODE="anyres", LLAVA_IMAGE_MODE="anyres",
                        REPS_LOC="after_pad", SPARSE_LENGTH="30", TEXT_SPARSE_LENGTH="30", IMAGE_SPARSE_LENGTH="30",
                        NUM_EXPENDED_TOKENS="0", QWEN_MIN_PIXELS="0", QWEN_MAX_PIXELS="0",
                        RANKER_QWEN_MIN_PIXELS="0", RANKER_QWEN_MAX_PIXELS="0", TBPR_TYPE="type",
                        SPARSE_VALUE_TYPE="sum", SPARSE_LOWER_OR_UPPER="lower", USE_STC_SPARSE="false",
                        GPUS="2", NUM_NODES="1", DEEPSPEED_BIN="deepspeed", OVERWRITE="false")

        def settings(group="ir", dataset="flickr", query="image"):
            return NS(group=group, families=("llava_mistral",), datasets=(dataset,), queries=(query,) if group == "ir" else (),
                      split="test", selection_split="test", models={"llava_mistral": model}, rankers={"llava_mistral": model},
                      roots=roots, controls=copy.deepcopy(controls), recover_missing_selection=False, selection_root=None,
                      overlap_only=False, output_dir=None, project=ROOT, bash="bash", dry_run=False,
                      environment={"MODIFY_TYPE": "modify_single", "USE_RERANK_PRIOR_CACHE": "true"},
                      options=NS(view_selection_modality="both", view_selection_pmi=True, view_overlap_selection=False,
                                 view_selection_hierarchical=False, view_example_mode="auto", view_example_max_tokens=0))

        yield NS(api=api, namespace=namespace, settings=settings, backend=backend, roots=roots)


@pytest.mark.parametrize("group,dataset,query,task", CASES)
def test_rgp_reuses_artifacts_and_separates_results(batch, group, dataset, query, task):
    settings = batch.settings(group, dataset, query)
    old, missing = batch.api.prepare_jobs(settings, backend=batch.backend, auto_modify=True)
    rgp, rgp_missing = batch.api.prepare_jobs(settings, backend=batch.backend, auto_modify=True,
                                           rerank_template="query_relevant")
    assert not missing and not rgp_missing
    old, rgp = old[0], rgp[0]
    assert old.configuration["modify_type"] == batch.api.caption_modify_type(task, query)
    assert "rerank_template" not in old.configuration and "use_rerank_prior_cache" not in old.configuration
    assert rgp.configuration["rerank_template"] == "query_relevant"
    assert rgp.configuration["modify_type"] == rgp.variables["JOB_MODIFY_TYPE"] == "no"
    assert rgp.configuration["use_rerank_prior_cache"] is False
    assert old.output != rgp.output and old.fingerprint != rgp.fingerprint
    for name in ("encoding_fingerprint", "views_fingerprint", "selection_fingerprint", "selection_cache_fingerprint"):
        assert old.configuration[name] == rgp.configuration[name]
    for name in ("VIEW_SELECTION_FILE", "DENSE_CORPUS_DIR", "SPARSE_CORPUS_DIR", "STAGE1_SPARSE_INDEX"):
        assert old.variables[name] == rgp.variables[name]
    # Removing only the RGP definition reproduces the exact old configuration/hash.
    restored = copy.deepcopy(rgp.configuration)
    restored.pop("rerank_template")
    restored.pop("use_rerank_prior_cache")
    restored["modify_type"] = old.configuration["modify_type"]
    assert restored == old.configuration and digest(restored) == old.fingerprint


@pytest.mark.parametrize("group", ["ir", "tbpr", "fused"])
def test_launcher_detection_preserves_existing_entries(batch, group):
    infer = batch.api.launcher_rerank_template
    assert infer(ROOT / "scripts" / f"{PREFIX}_rgp_{group}.sh", group) == "query_relevant"
    assert infer(ROOT / "scripts" / f"{PREFIX}_{group}.sh", group) == "what_caption_generation"
    assert infer(None, group) == "what_caption_generation"
    with pytest.raises(ValueError, match="does not match"):
        infer(f"{PREFIX}_rgp_ir.sh", "tbpr")


def test_batch_dry_run_and_metadata_name_rgp(batch, capsys):
    settings = batch.settings()
    launcher = ROOT / "scripts" / f"{PREFIX}_rgp_ir.sh"
    settings.dry_run = True
    assert batch.api.run(settings, launcher=launcher, backend=batch.backend,
                         runner=lambda *a, **k: pytest.fail("dry run launched retrieval")) == 0
    assert "Rerank: query_relevant modify_type=no" in capsys.readouterr().out
    assert not list(batch.roots["results"].rglob("*.json"))
    settings.dry_run = False
    calls = []

    def runner(command, **kwargs):
        calls.append((command, kwargs))
        assert kwargs["env"]["JOB_MODIFY_TYPE"] == "no"
        return NS(returncode=0)

    assert batch.api.run(settings, launcher=launcher, backend=batch.backend, runner=runner) == 0
    assert len(calls) == 1 and calls[0][0][1:] == [launcher.as_posix(), "--run-one"]
    metadata_path, = batch.roots["results"].rglob("batch_job.json")
    metadata = json.loads(metadata_path.read_text())
    assert metadata["configuration"]["rerank_template"] == "query_relevant"
    assert metadata["configuration"]["modify_type"] == "no"
    assert metadata["configuration"]["use_rerank_prior_cache"] is False
    assert metadata["fingerprint"] == digest(metadata["configuration"])


@pytest.mark.parametrize("group", ["ir", "tbpr", "fused"])
def test_cgp_no_launcher_policy_is_explicit(batch, group):
    launcher = ROOT / "scripts" / f"{PREFIX}_cgp_no_{group}.sh"
    assert batch.api.launcher_rerank_template(launcher, group) == "what_caption_generation"
    assert batch.api.launcher_force_no_modify(launcher, group)
    for old in (None, f"{PREFIX}_{group}.sh", f"{PREFIX}_rgp_{group}.sh"):
        assert not batch.api.launcher_force_no_modify(old, group)
    other = "tbpr" if group == "ir" else "ir"
    with pytest.raises(ValueError, match="does not match"):
        batch.api.launcher_force_no_modify(launcher, other)


@pytest.mark.parametrize("stage2", ["hybrid", "dense"])
@pytest.mark.parametrize("group,dataset,query,task", CASES)
def test_cgp_no_reuses_artifacts_with_distinct_result_identity(batch, stage2, group, dataset, query, task):
    settings = batch.settings(group, dataset, query)
    settings.controls["STAGE2_METHOD"] = stage2
    old, missing = batch.api.prepare_jobs(settings, backend=batch.backend, auto_modify=True)
    rgp, rgp_missing = batch.api.prepare_jobs(settings, backend=batch.backend, auto_modify=True,
                                           rerank_template="query_relevant")
    plain, plain_missing = batch.api.prepare_jobs(settings, backend=batch.backend, auto_modify=True, force_no_modify=True)
    assert not missing and not rgp_missing and not plain_missing
    old, rgp, plain = old[0], rgp[0], plain[0]
    assert plain.configuration["rerank_template"] == "what_caption_generation"
    assert plain.configuration["modify_type"] == plain.variables["JOB_MODIFY_TYPE"] == "no"
    assert plain.configuration["use_rerank_prior_cache"] is False
    assert plain.variables["USE_RERANK_PRIOR_CACHE"] == "false"
    assert len({job.output for job in (old, rgp, plain)}) == 3
    assert len({job.fingerprint for job in (old, rgp, plain)}) == 3
    for name in ("encoding_fingerprint", "views_fingerprint", "selection_fingerprint", "selection_cache_fingerprint"):
        assert old.configuration[name] == rgp.configuration[name] == plain.configuration[name]
    for name in ("VIEW_SELECTION_FILE", "DENSE_CORPUS_DIR", "SPARSE_CORPUS_DIR", "STAGE1_SPARSE_INDEX"):
        assert old.variables[name] == rgp.variables[name] == plain.variables[name]
    restored = copy.deepcopy(plain.configuration)
    restored.pop("rerank_template")
    restored.pop("use_rerank_prior_cache")
    restored["modify_type"] = batch.api.caption_modify_type(task, query)
    assert restored == old.configuration and digest(restored) == old.fingerprint


@pytest.mark.parametrize("options", [dict(auto_modify=False), dict(auto_modify=True, rerank_template="query_relevant")])
def test_cgp_no_rejects_inconsistent_internal_policy(batch, options):
    with pytest.raises(ValueError, match="dedicated CGP-no launcher"):
        batch.api.prepare_jobs(batch.settings(), backend=batch.backend, force_no_modify=True, **options)


@pytest.mark.parametrize("group,dataset,query", [("ir", "flickr", "image"), ("tbpr", "CUHK-PEDES", "text"),
                                                ("fused", "llava", "image")])
def test_cgp_no_batch_log_environment_and_metadata_agree(batch, capsys, group, dataset, query):
    settings = batch.settings(group, dataset, query)
    launcher = ROOT / "scripts" / f"{PREFIX}_cgp_no_{group}.sh"
    settings.dry_run = True
    assert batch.api.run(settings, launcher=launcher, backend=batch.backend,
                         runner=lambda *a, **k: pytest.fail("dry run launched retrieval")) == 0
    output = capsys.readouterr().out
    assert "Rerank: what_caption_generation modify_type=no" in output
    assert "USE_RERANK_PRIOR_CACHE=false" in output
    assert not list(batch.roots["results"].rglob("*.json"))
    settings.dry_run = False
    calls = []

    def runner(command, **kwargs):
        calls.append(command)
        assert kwargs["env"]["JOB_MODIFY_TYPE"] == "no"
        assert kwargs["env"]["USE_RERANK_PRIOR_CACHE"] == "false"
        return NS(returncode=0)

    assert batch.api.run(settings, launcher=launcher, backend=batch.backend, runner=runner) == 0
    assert calls == [[settings.bash, launcher.as_posix(), "--run-one"]]
    path, = batch.roots["results"].rglob("batch_job.json")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    assert metadata["configuration"]["rerank_template"] == "what_caption_generation"
    assert metadata["configuration"]["modify_type"] == "no"
    assert metadata["configuration"]["use_rerank_prior_cache"] is False
    assert metadata["fingerprint"] == digest(metadata["configuration"])


def test_stage3_dispatches_rgp(batch):
    api = load("three_stage_retrieval_pipeline_adaptive_views.py", {"_dispatch_stage3_rerank"}, {
        "argparse": argparse, "Any": object, "Run": dict, "SimpleNamespace": NS,
    })
    observed = []
    ranker = NS(rerank=lambda *a, **k: observed.append((a, k)),
                caption_generation_rerank=lambda *a, **k: pytest.fail("RGP used caption scoring"))
    api._dispatch_stage3_rerank(ranker, {"q": {"d": 1}},
        NS(rerank_template="query_relevant", rerank_type="pointwise", rerank_num=20, rerank_batch_size=4),
        None, None, None)
    assert observed[0][0][1:3] == ("pointwise", 20)
    assert observed[0][1] == {"rerank_batch_size": 4, "rerank_prompt_type": "query_relevant"}


@pytest.fixture
def pipeline(batch, monkeypatch):
    """Run the native parser, validation, preflight and dispatch without model imports."""
    for name in ("stc_sparse", "lens_sparse"):
        feature = sys.modules[name]
        feature.add_parser = lambda *a, **k: None
        feature.validate = lambda *a: None
        feature.bind_corpus = lambda *a: None
        feature.enabled = lambda *a: False
    sys.modules["dense_representation"].boolean = lambda value: str(value).lower() == "true"
    sys.modules["dense_representation"].prepare_pipeline = lambda *a: None
    mass = batch.namespace["probability_mass"]
    mass.enabled = lambda *a: False
    mass.validate_requested = lambda *a: None
    corpus = load("qwen_vl_pipeline.py", {"corpus_kind"}, {})
    monkeypatch.setitem(sys.modules, "qwen_vl_pipeline", corpus)
    backend = load("adaptive_view_encoding.py", {"preflight", "validate_encoding_method"}, {
        "Path": Path, "stc": sys.modules["stc_sparse"], "lens": sys.modules["lens_sparse"],
        "views": batch.namespace["views"], "probability_mass": mass,
        "option_shuffle": batch.namespace["option_shuffle"],
        "load_model_identity": batch.backend.load_model_identity, "metadata": batch.backend.metadata,
    }, ("ADAPTIVE_EOL_TYPES",))
    monkeypatch.setitem(sys.modules, "adaptive_view_encoding", backend)
    prior = load("rerank_prior_cache.py", {"validate_prior_cache_options"}, {})
    import stage2_retrieval_mode as stage2
    namespace = {
        "__name__": "_test_adaptive_rgp_pipeline", "argparse": argparse, "SimpleNamespace": NS,
        "Path": Path, "os": os, "sys": sys, "Any": object, "Run": dict,
        "configure_stage2_mode": stage2.configure, "stage2_method": stage2.method,
        "validate_prior_cache_options": prior.validate_prior_cache_options,
        "full_prompt": NS(enabled=lambda _: False), "prompt_constraints": NS(enabled=lambda _: False),
    }
    api = load("three_stage_retrieval_pipeline_adaptive_views.py", {
        "_parser", "_validate_args", "_validate_alpha_options", "_retrieval_namespaces",
        "retrieval_eol_type", "_find_corpus_pickles", "_dispatch_stage3_rerank",
    }, namespace, ("STAGE1_METHODS", "STAGE2_EOL_TYPE", "STAGE2_CALCULATE_TYPE", "DATASET_TASK_TYPES"))
    monkeypatch.setitem(sys.modules, namespace["__name__"], api)
    dense, sparse = batch.backend.artifact_dirs()
    selection, _, _ = batch.namespace["selection_location"]()

    def arguments(dataset="flickr", query="image", task="ir", extra=()):
        return api._parser().parse_args([
            "--dataset-name", dataset, "--query-type", query, "--task-type", task,
            "--adaptive-views", "--view-selection-file", str(selection),
            "--stage1-sparse-index", str(sparse), "--stage2-dense-index", str(dense),
            "--stage2-sparse-index", str(sparse),
            "--retrieval-model-name-or-path", str(batch.settings().models["llava_mistral"]),
            "--rerank-template", "query_relevant", *extra,
        ])

    def dispatch(args):
        calls = []

        def record(mode):
            def rerank(*positional, **kwargs):
                calls.append((mode, positional, kwargs))
                return {"q": {"d": 0.75}}
            return rerank

        ranker = NS(rerank=record("rgp"), caption_generation_rerank=record("cgp"))
        result = api._dispatch_stage3_rerank(ranker, {"q": {"d": 1}}, args, None, None, None)
        assert result == {"q": {"d": 0.75}}
        assert len(calls) == 1
        return calls[0]

    return NS(api=api, backend=backend, arguments=arguments, dispatch=dispatch, dense=dense, sparse=sparse)


@pytest.mark.parametrize("group,dataset,query,task", CASES)
def test_rgp_survives_native_validation_preflight_and_dispatch(pipeline, group, dataset, query, task):
    args = pipeline.arguments(dataset, query, task)
    pipeline.api._validate_args(args)
    assert args.rerank_template == "query_relevant"
    mode, positional, kwargs = pipeline.dispatch(args)
    assert mode == "rgp" and positional[1:3] == ("pointwise", 20)
    assert kwargs["rerank_prompt_type"] == "query_relevant"
    assert args.modify_type == "no" and not args.use_rerank_prior_cache


@pytest.mark.parametrize("template", ["relevant", "what_caption_generation"])
@pytest.mark.parametrize("group,dataset,query,task", CASES)
def test_native_preflight_preserves_legacy_cgp(pipeline, group, dataset, query, task, template):
    args = pipeline.arguments(dataset, query, task, ["--rerank-template", template])
    pipeline.api._validate_args(args)
    assert args.rerank_template == "what_caption_generation"
    mode, positional, kwargs = pipeline.dispatch(args)
    assert mode == "cgp" and positional[1:3] == ("pointwise", 20)
    assert kwargs["rerank_prompt_type"] == "what_caption_generation"
    assert positional[6].modify_type == "no"


def test_native_cgp_still_accepts_caption_prior_cache(pipeline):
    args = pipeline.arguments(extra=["--rerank-template", "what_caption_generation",
                                     "--modify-type", "modify_single", "--use-rerank-prior-cache"])
    pipeline.api._validate_args(args)
    mode, positional, kwargs = pipeline.dispatch(args)
    assert mode == "cgp" and positional[6].modify_type == "modify_single"
    assert positional[6].use_rerank_prior_cache and kwargs["prior_cache_prepared"]


@pytest.mark.parametrize("extra,preflight_error,native_error", [
    (["--rerank-type", "listwise"], "requires --rerank-type pointwise", "requires --rerank-type pointwise"),
    (["--modify-type", "modify_single"], "requires --modify-type no", "caption_generation"),
    (["--use-rerank-prior-cache"], "does not support --use-rerank-prior-cache", "requires --modify_type=modify_single"),
    (["--rerank-template", "unsupported"], "supports what_caption_generation", "supports what_caption_generation"),
])
def test_rgp_rejects_conflicting_options_before_model_loading(pipeline, monkeypatch, extra, preflight_error, native_error):
    def no_model(*a, **k):
        pytest.fail("Invalid reranking options reached model identity loading")

    monkeypatch.setitem(pipeline.backend.preflight.__globals__, "load_model_identity", no_model)
    with pytest.raises(ValueError, match=preflight_error):
        pipeline.backend.preflight(pipeline.arguments(extra=extra), pipeline.api)
    with pytest.raises(ValueError, match=native_error):
        pipeline.api._validate_args(pipeline.arguments(extra=extra))


@pytest.mark.parametrize("corruption", ["manifest", "shards"])
def test_rgp_preflight_still_validates_candidate_artifacts(pipeline, corruption):
    if corruption == "manifest":
        manifest = pipeline.sparse / "adaptive_views_manifest.json"
        value = json.loads(manifest.read_text(encoding="utf-8"))
        value["fingerprint"] = "corrupt"
        atomic_json(manifest, value)
        message = "Corrupt adaptive corpus manifest"
    else:
        (pipeline.dense / "corpus_1.pkl").touch()
        message = "Adaptive dense shard count mismatch"
    with pytest.raises(ValueError, match=message):
        pipeline.api._validate_args(pipeline.arguments())


@pytest.fixture
def shell(tmp_path):
    if BASH is None:
        pytest.skip("Bash not installed")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "bin").mkdir()
    (tmp_path / "logs").mkdir()
    for name in ("stc_sparse_mode.sh", "lens_sparse_mode.sh", "freeret_dense_mode.sh", "stage2_retrieval_mode.sh",
                 *(f"{PREFIX}_{mode}_{group}.sh" for mode in ("rgp", "cgp_no") for group in ("ir", "tbpr", "fused"))):
        shutil.copyfile(ROOT / "scripts" / name, tmp_path / "scripts" / name)
    stub = ('#!/usr/bin/env bash\n'
            'printf "%s\\0" "$@" > logs/args\n'
            'printf "%s" "${USE_RERANK_PRIOR_CACHE-unset}" > logs/prior_cache\n')
    stub += 'for name in ' + ' '.join((*CGP_NO_DEFAULTS, "DATASETS", "IR_QUERY_TYPES", "RERANK_TEMPLATE")) + '; do\n'
    stub += '    printf "%s=%s\\0" "$name" "${!name-unset}"\ndone > logs/configuration\n'
    for name in ("python", "deepspeed"):
        path = tmp_path / "bin" / name
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(stub)
        path.chmod(0o755)
    bin_path = (tmp_path / "bin").as_posix()
    if os.name == "nt":
        bin_path = "/" + bin_path[0].lower() + bin_path[2:]

    def launch(group, dataset, query, *, internal=True, family="llava_mistral", mode="rgp", defaults=False,
               rerank_template=None):
        env = dict(os.environ, MODEL_FAMILY=family, MODEL_PATH="models/" + family, DATASET=dataset,
                   IR_QUERY_TYPE=query, IR_QUERY_TYPES=query, DATASETS=dataset, MODEL_FAMILIES=family,
                   OUTPUT_DIR="saved results", VIEW_SELECTION_FILE="saved selection.json",
                   DENSE_CORPUS_DIR="saved dense", SPARSE_CORPUS_DIR="saved sparse", DRY_RUN="false",
                   USE_FREERET_DENSE="false", USE_FREERET_FULL_PROMPT="false", USE_FREERET_PROMPT_CONSTRAINTS="false",
                   USE_STC_SPARSE="false", USE_LENS_SPARSE="false", OVERWRITE="false", LLAVA_IMAGE_MODE="base",
                   MODIFY_TYPE="modify_single", JOB_MODIFY_TYPE="modify_single", USE_RERANK_PRIOR_CACHE="true",
                   PYTHON_BIN="./bin/python", DEEPSPEED_BIN="./bin/deepspeed", MSYS_NO_PATHCONV="1")
        env["LLAVA_IMAGE_MODE"] = "anyres" if family.startswith("qwen") else "base"
        env.pop("RERANK_TEMPLATE", None)
        if rerank_template is not None:
            env["RERANK_TEMPLATE"] = rerank_template
        if defaults:
            keys = {*CGP_NO_DEFAULTS, "MODEL_FAMILY", "MODEL_PATH", "RANKER_MODEL_PATH", "DATASET", "DATASETS",
                    "IR_QUERY_TYPE", "IR_QUERY_TYPES", "OUTPUT_DIR"}
            keys.update(name for name in env if name.startswith(("MODEL_PATH_", "RANKER_MODEL_PATH_")))
            for name in keys:
                env.pop(name, None)
        command = 'export PATH=' + shlex.quote(bin_path) + ':/usr/bin:/bin:"$PATH"; exec /usr/bin/bash scripts/'
        command += f"{PREFIX}_{mode}_{group}.sh" + (" --run-one" if internal else "")
        result = subprocess.run([BASH, "-c", command], cwd=tmp_path, env=env, capture_output=True,
                                text=True, encoding="utf-8", timeout=30)
        assert result.returncode == 0, result.stdout + result.stderr
        return (tmp_path / "logs" / "args").read_bytes().decode().rstrip("\0").split("\0")

    return launch


@pytest.mark.parametrize("family", ["llava_llama", "llava_mistral"])
@pytest.mark.parametrize("group,dataset,query,task", CASES)
def test_shell_sends_rgp_without_prior_correction(shell, tmp_path, family, group, dataset, query, task):
    args = shell(group, dataset, query, family=family)
    value = lambda key: args[args.index(key) + 1]
    assert "src/three_stage_retrieval_pipeline_adaptive_views.py" in args
    assert value("--rerank-template") == "query_relevant" and value("--modify-type") == "no"
    assert value("--rerank-type") == "pointwise" and "--use-rerank-prior-cache" not in args
    assert value("--task-type") == task and value("--query-type") == query
    assert value("--dataset-name") == dataset
    assert value("--view-selection-file") == "saved selection.json"
    assert value("--stage2-dense-index") == "saved dense"
    assert value("--stage2-sparse-index") == "saved sparse"
    assert value("--retrieval-llava-image-mode") == value("--ranker-llava-image-mode") == "base"
    assert (tmp_path / "logs" / "prior_cache").read_text() == "false"


@pytest.mark.parametrize("group,dataset,query", [("ir", "flickr", "image"), ("tbpr", "CUHK-PEDES", "text"),
                                                ("fused", "webqa", "text")])
def test_shell_forwards_its_own_rgp_launcher_to_batch(shell, tmp_path, group, dataset, query):
    args = shell(group, dataset, query, internal=False)
    value = lambda key: args[args.index(key) + 1]
    assert args[:2] == ["-B", "src/adaptive_retrieval_batch.py"]
    assert value("--task-group") == group
    assert value("--launcher").endswith(f"/{PREFIX}_rgp_{group}.sh")
    assert (tmp_path / "logs" / "prior_cache").read_text() == "false"


@pytest.mark.parametrize("family", ["llava_llama", "llava_mistral"])
@pytest.mark.parametrize("group,dataset,query,task", CASES)
def test_cgp_no_shell_reaches_native_caption_dispatch(shell, pipeline, tmp_path, family, group, dataset, query, task):
    command = shell(group, dataset, query, family=family, mode="cgp_no")
    entry = command.index("src/three_stage_retrieval_pipeline_adaptive_views.py")
    args = pipeline.api._parser().parse_args(command[entry + 1:])
    args.stage1_sparse_index = str(pipeline.sparse)
    args.stage2_sparse_index = str(pipeline.sparse)
    args.stage2_dense_index = str(pipeline.dense)
    args.view_selection_file = pipeline.arguments().view_selection_file
    args.output_dir = None
    pipeline.api._validate_args(args)
    mode, positional, kwargs = pipeline.dispatch(args)
    assert mode == "cgp" and positional[1:3] == ("pointwise", 20)
    assert kwargs["rerank_prompt_type"] == "what_caption_generation"
    assert positional[6].modify_type == "no" and not positional[6].use_rerank_prior_cache
    assert "prior_cache_prepared" not in kwargs and "--use-rerank-prior-cache" not in command
    assert args.task_type == task and args.query_type == query and args.dataset_name == dataset
    assert args.retrieval_llava_image_mode == args.ranker_llava_image_mode == "base"
    assert (tmp_path / "logs" / "prior_cache").read_text() == "false"


@pytest.mark.parametrize("group,datasets", [("ir", "flickr"), ("tbpr", "CUHK-PEDES ICFG-PEDES RSTPReid"),
                                           ("fused", "webqa remuq llava")])
def test_cgp_no_shell_defaults_and_batch_launcher(shell, tmp_path, group, datasets):
    args = shell(group, "ignored", "ignored", internal=False, mode="cgp_no", defaults=True)
    value = lambda key: args[args.index(key) + 1]
    assert args[:2] == ["-B", "src/adaptive_retrieval_batch.py"]
    assert value("--task-group") == group
    assert value("--launcher").endswith(f"/{PREFIX}_cgp_no_{group}.sh")
    records = (tmp_path / "logs" / "configuration").read_bytes().decode().rstrip("\0").split("\0")
    configuration = dict(row.split("=", 1) for row in records)
    for name, expected in CGP_NO_DEFAULTS.items():
        assert configuration[name] == expected, name
    assert configuration["DATASETS"] == datasets
    if group == "ir":
        assert configuration["IR_QUERY_TYPES"] == "image text"
    assert (tmp_path / "logs" / "prior_cache").read_text() == "false"


REVERSE_CASES = CASES[:3]
RANKER_FAMILIES = {
    "llava_llama": ("llava-hf-llama3-llava-next-8b-hf", "llava_next"),
    "llava_mistral": ("llava-hf-llava-v1.6-mistral-7b-hf", "llava_next"),
    "qwen2": ("Qwen-Qwen2-VL-7B-Instruct", "qwen2_vl"),
    "qwen25": ("Qwen-Qwen2.5-VL-7B-Instruct", "qwen2_5_vl"),
    "qwen3": ("Qwen-Qwen3-VL-8B-Instruct", "qwen3_vl"),
}


@pytest.fixture
def pointwise_templates(monkeypatch):
    """Execute production template selection and Qwen rendering without model weights."""
    names = {"llama3_template", "llava_mistral_template"}
    for prefix in ("", "mistral_"):
        for direction in ("text", "image"):
            for reverse in ("", "reverse_"):
                names.add(f"{prefix}{direction}_{reverse}query_relevant_prompt")
        for reverse in ("", "reverse_"):
            names.add(f"{prefix}person_retrieval_{reverse}query_relevant_prompt")
    templates = load("template.py", set(), {}, names)
    qwen = load("qwen_vl_prompts.py", {
        "model_family", "is_native_qwen_model", "render_user", "mistral_body", "rerank_template",
    }, {"baseline": templates}, (
        "PROMPT_VERSION", "QWEN2_PROMPT_VERSION", "QWEN3_PROMPT_VERSION", "PROMPT_VERSIONS", "MODEL_TYPES",
        "IMAGE", "IR_POINTWISE", "DIRECTED", "CAPTION_SUFFIXES"))
    monkeypatch.setitem(sys.modules, "qwen_vl_prompts", qwen)
    path = ROOT / "src/reranker.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "rerank")
    namespace = dict(vars(templates), nullcontext=nullcontext, _is_main_process=lambda: False,
        torch=NS(no_grad=nullcontext, cuda=NS(amp=NS(autocast=nullcontext))))
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    return NS(templates=templates, rerank=namespace["rerank"])


@pytest.mark.parametrize("template", ["query_relevant", "reverse_query_relevant"])
@pytest.mark.parametrize("family", RANKER_FAMILIES)
@pytest.mark.parametrize("group,dataset,query,task", REVERSE_CASES)
def test_native_stage3_selects_actual_directed_template(pipeline, pointwise_templates, template, family, group, dataset, query, task):
    args = pipeline.arguments(dataset, query, task, ["--rerank-template", template])
    pipeline.api._validate_args(args)
    assert args.rerank_template == template
    path, model_type = RANKER_FAMILIES[family]
    observed = []

    def batch_rerank(*positional, **kwargs):
        observed.append((positional, kwargs))
        return {"q": {"d": 0.8}}

    def render(conversations, **kwargs):
        assert kwargs == dict(tokenize=False, add_generation_prompt=True)
        assert len(conversations) == 1 and conversations[0]["role"] == "user"
        body = "".join(part["text"] if part["type"] == "text" else "<image>" for part in conversations[0]["content"])
        return "USER:" + body + "ASSISTANT:"

    ranker = NS(model=NS(config=NS(model_type=model_type)), query_type=query,
                processor=NS(apply_chat_template=render), _batch_rerank=batch_rerank,
                caption_generation_rerank=lambda *a, **k: pytest.fail("Directed RGP used caption scoring"))
    ranker.rerank = lambda *a, **k: pointwise_templates.rerank(ranker, *a, **k)
    result = pipeline.api._dispatch_stage3_rerank(ranker, {"q": {"d": 1}}, args, NS(),
        NS(task_type=task, fp16=False), NS(model_name_or_path=path))
    assert result == {"q": {"d": 0.8}} and len(observed) == 1
    prefix = "" if family == "llava_llama" else "mistral_"
    reverse = "reverse_" if template == "reverse_query_relevant" else ""
    name = f"{prefix}{query}_{reverse}query_relevant_prompt" if task == "ir" else f"{prefix}person_retrieval_{reverse}query_relevant_prompt"
    expected = getattr(pointwise_templates.templates, name)
    if family.startswith("qwen"):
        expected = "USER:" + expected[len("[INST]"):-len("[/INST]")] + "ASSISTANT:"
    assert observed[0][0][2] == expected
    assert observed[0][0][4] == template
    assert "'Yes' or 'No'" in expected


@pytest.mark.parametrize("group", ["ir", "tbpr"])
def test_reverse_launcher_and_legacy_policy_selection(batch, group):
    launcher = ROOT / "scripts" / f"{PREFIX}_rgp_{group}.sh"
    assert batch.api.launcher_rerank_template(launcher, group, "reverse_query_relevant") == "reverse_query_relevant"
    assert batch.api.launcher_rerank_template(launcher, group) == "query_relevant"
    for value in ("", "reverse_query_relevant ", "wrong", "what_caption_generation"):
        with pytest.raises(ValueError, match="RERANK_TEMPLATE"):
            batch.api.launcher_rerank_template(launcher, group, value)
    assert batch.api.launcher_rerank_template(f"{PREFIX}_{group}.sh", group, "reverse_query_relevant") == "what_caption_generation"
    assert batch.api.launcher_rerank_template(f"{PREFIX}_cgp_no_{group}.sh", group, "reverse_query_relevant") == "what_caption_generation"


@pytest.mark.parametrize("stage2", ["hybrid", "dense"])
@pytest.mark.parametrize("group,dataset,query,task", REVERSE_CASES)
def test_reverse_result_identity_reuses_encoding(batch, stage2, group, dataset, query, task):
    settings = batch.settings(group, dataset, query)
    settings.controls["STAGE2_METHOD"] = stage2
    normal, missing = batch.api.prepare_jobs(settings, backend=batch.backend, auto_modify=True, rerank_template="query_relevant")
    reverse, reverse_missing = batch.api.prepare_jobs(settings, backend=batch.backend, auto_modify=True,
                                                     rerank_template="reverse_query_relevant")
    assert not missing and not reverse_missing
    normal, reverse = normal[0], reverse[0]
    expected = copy.deepcopy(normal.configuration)
    expected["rerank_template"] = "reverse_query_relevant"
    assert expected == reverse.configuration
    assert normal.output != reverse.output and normal.fingerprint != reverse.fingerprint
    for name in ("VIEW_SELECTION_FILE", "DENSE_CORPUS_DIR", "SPARSE_CORPUS_DIR", "STAGE1_SPARSE_INDEX"):
        assert normal.variables[name] == reverse.variables[name]
    assert reverse.variables["RERANK_TEMPLATE"] == "reverse_query_relevant"
    assert reverse.variables["JOB_MODIFY_TYPE"] == "no" and reverse.variables["USE_RERANK_PRIOR_CACHE"] == "false"


@pytest.mark.parametrize("group,dataset,query", [("ir", "flickr", "image"), ("tbpr", "CUHK-PEDES", "text")])
def test_reverse_batch_uses_one_template_for_log_command_and_metadata(batch, capsys, group, dataset, query):
    settings = batch.settings(group, dataset, query)
    settings.environment["RERANK_TEMPLATE"] = "reverse_query_relevant"
    settings.dry_run = True
    launcher = ROOT / "scripts" / f"{PREFIX}_rgp_{group}.sh"
    assert batch.api.run(settings, launcher=launcher, backend=batch.backend,
                         runner=lambda *a, **k: pytest.fail("Dry run launched a ranker")) == 0
    output = capsys.readouterr().out
    assert "Rerank: reverse_query_relevant modify_type=no" in output
    assert "RERANK_TEMPLATE=reverse_query_relevant" in output
    assert not list(batch.roots["results"].rglob("*.json"))
    settings.dry_run = False

    def runner(command, **kwargs):
        assert command[1:] == [launcher.as_posix(), "--run-one"]
        assert kwargs["env"]["RERANK_TEMPLATE"] == "reverse_query_relevant"
        assert kwargs["env"]["JOB_MODIFY_TYPE"] == "no" and kwargs["env"]["USE_RERANK_PRIOR_CACHE"] == "false"
        return NS(returncode=0)

    assert batch.api.run(settings, launcher=launcher, backend=batch.backend, runner=runner) == 0
    path, = batch.roots["results"].rglob("batch_job.json")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    assert metadata["configuration"]["rerank_template"] == "reverse_query_relevant"
    assert metadata["configuration"]["modify_type"] == "no" and metadata["configuration"]["use_rerank_prior_cache"] is False
    assert metadata["fingerprint"] == digest(metadata["configuration"])


@pytest.mark.parametrize("dataset,query,task", [("webqa", "text", "t2it"), ("remuq", "image", "it2t"), ("llava", "image", "it2t")])
def test_reverse_fused_is_rejected_before_identity_loading(pipeline, batch, monkeypatch, dataset, query, task):
    def no_model(*a, **k):
        pytest.fail("Unsupported reverse task loaded a model identity")

    monkeypatch.setitem(pipeline.backend.preflight.__globals__, "load_model_identity", no_model)
    with pytest.raises(ValueError, match="supports IR and TBPR only"):
        pipeline.api._validate_args(pipeline.arguments(dataset, query, task, ["--rerank-template", "reverse_query_relevant"]))
    with pytest.raises(ValueError, match="supports IR and TBPR only"):
        batch.api.prepare_jobs(batch.settings("fused", dataset, query), backend=batch.backend,
                               auto_modify=True, rerank_template="reverse_query_relevant", identity_loader=no_model)
    with pytest.raises(ValueError, match="supports IR and TBPR only"):
        batch.api.launcher_rerank_template(f"{PREFIX}_rgp_fused.sh", "fused", "reverse_query_relevant")


@pytest.mark.parametrize("extra,preflight_error,native_error", [
    (["--rerank-type", "listwise"], "requires --rerank-type pointwise", "requires --rerank-type pointwise"),
    (["--modify-type", "modify_single"], "requires --modify-type no", "caption_generation"),
    (["--use-rerank-prior-cache"], "does not support --use-rerank-prior-cache", "requires --modify_type=modify_single"),
])
def test_reverse_conflicts_fail_before_identity_loading(pipeline, monkeypatch, extra, preflight_error, native_error):
    monkeypatch.setitem(pipeline.backend.preflight.__globals__, "load_model_identity",
                        lambda *a, **k: pytest.fail("Invalid reverse options loaded a model identity"))
    options = ["--rerank-template", "reverse_query_relevant", *extra]
    with pytest.raises(ValueError, match=preflight_error):
        pipeline.backend.preflight(pipeline.arguments(extra=options), pipeline.api)
    with pytest.raises(ValueError, match=native_error):
        pipeline.api._validate_args(pipeline.arguments(extra=options))


def test_unknown_template_error_shows_received_value(pipeline):
    value = "reverse_query_relevant "
    with pytest.raises(ValueError) as error:
        pipeline.api._validate_args(pipeline.arguments(extra=["--rerank-template", value]))
    assert repr(value) in str(error.value)


@pytest.mark.parametrize("family", RANKER_FAMILIES)
@pytest.mark.parametrize("group,dataset,query,task", REVERSE_CASES)
def test_reverse_shell_sends_selected_template(shell, tmp_path, family, group, dataset, query, task):
    arguments = shell(group, dataset, query, family=family, rerank_template="reverse_query_relevant")
    value = lambda key: arguments[arguments.index(key) + 1]
    assert value("--rerank-template") == "reverse_query_relevant"
    assert value("--modify-type") == "no" and value("--rerank-type") == "pointwise"
    assert value("--task-type") == task and value("--query-type") == query
    assert "--use-rerank-prior-cache" not in arguments
    assert (tmp_path / "logs" / "prior_cache").read_text() == "false"


@pytest.mark.parametrize("group,dataset,query", [("ir", "flickr", "image"), ("tbpr", "CUHK-PEDES", "text")])
@pytest.mark.parametrize("template", [None, "reverse_query_relevant"])
def test_rgp_shell_exports_template_to_batch(shell, tmp_path, group, dataset, query, template):
    shell(group, dataset, query, internal=False, rerank_template=template)
    records = (tmp_path / "logs" / "configuration").read_bytes().decode().rstrip("\0").split("\0")
    configuration = dict(row.split("=", 1) for row in records)
    assert configuration["RERANK_TEMPLATE"] == (template or "query_relevant")
