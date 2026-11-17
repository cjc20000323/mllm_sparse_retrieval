"""Model-free regressions for the adaptive original-text handoff and query settings."""
import argparse
import ast
import copy
import csv
import importlib.util
import json
import math
import os
import pickle
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = "disassembleeol_separate"
ORIGINAL = "disassembleeol_separate_origin_text"


def functions(filename, names, namespace, constants=()):
    source = ROOT / "src" / filename
    tree = ast.parse(source.read_text(encoding="utf-8"))
    body = [node for node in tree.body if (
        isinstance(node, ast.ImportFrom) and node.module == "__future__") or (
        isinstance(node, ast.FunctionDef) and node.name in names) or (
        isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id in constants
                                            for target in node.targets))]
    exec(compile(ast.Module(body=body, type_ignores=[]), str(source), "exec"), namespace)
    return SimpleNamespace(**namespace)


@pytest.fixture
def artifacts():
    spec = importlib.util.spec_from_file_location("adaptive_artifacts_probe", ROOT / "src/adaptive_artifacts.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def method():
    return functions("adaptive_view_encoding.py", {"validate_encoding_method"}, {
        "ADAPTIVE_EOL_TYPES": (DEFAULT, ORIGINAL),
    }).validate_encoding_method


@pytest.mark.parametrize("task,family", [("ir", "llava_mistral"), ("ir", "llava_llama")])
def test_original_method_supported(method, task, family):
    method(ORIGINAL, "concat", False, task, family=family)


@pytest.mark.parametrize("eol,calculate,manual,task,family", [
    ("prompteol", "concat", False, "ir", "llava_mistral"),
    (ORIGINAL, "separate", False, "ir", "llava_mistral"),
    (ORIGINAL, "concat", True, "ir", "llava_mistral"),
    (ORIGINAL, "concat", False, "tbpr", "llava_mistral"),
    (ORIGINAL, "concat", False, "ir", "qwen25"),
])
def test_unsupported_original_method_rejected(method, eol, calculate, manual, task, family):
    with pytest.raises(ValueError):
        method(eol, calculate, manual, task, family=family)


def test_original_defaults_remain_supported(method):
    for task in ("ir", "tbpr", "t2it", "it2t"):
        for family in ("llava_mistral", "llava_llama", "qwen25"):
            method(DEFAULT, "concat", False, task, family=family)


@pytest.fixture
def completed(tmp_path, artifacts):
    selection = {"views_fingerprint": "f" * 64, "selected": [{"label": "A", "aspect": "objects"}],
                 "selection_modality": "both",
                 "calibration": {"configuration": {"method": "pmi", "prior_prompt_mode": "instruction_only"}}}
    selection["fingerprint"] = artifacts._digest(selection)
    selection_file = tmp_path / "selection.json"
    selection_file.write_text(json.dumps(selection), encoding="utf-8")
    configuration = {"method": [ORIGINAL, "concat"], "kind": "text",
                     "views_fingerprint": selection["views_fingerprint"],
                     "selection_calibration": selection["calibration"]["configuration"]}
    metadata = {"configuration": configuration, "fingerprint": artifacts._digest(configuration)}
    roots = [tmp_path / "dense", tmp_path / "sparse"]
    for root in roots:
        root.mkdir()
        (root / artifacts.MANIFEST).write_text(json.dumps({
            **metadata, "num_shards": 2, "selection_fingerprint": selection["fingerprint"],
        }), encoding="utf-8")
    return roots, selection_file, metadata, selection["fingerprint"]


def test_completed_paths_publish_and_read(artifacts, completed, tmp_path):
    path = tmp_path / "paths.json"
    artifacts.publish(path, *completed)
    record = artifacts.read(path)
    assert record["configuration"]["method"] == [ORIGINAL, "concat"]
    assert Path(record["dense_root"]) == completed[0][0]
    assert Path(record["sparse_root"]) == completed[0][1]
    assert Path(record["selection_file"]) == completed[1]
    result = subprocess.run([sys.executable, "-B", str(ROOT / "src/adaptive_artifacts.py"),
                             str(path), "sparse_root"], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(completed[0][1])


def test_incomplete_encoding_does_not_publish(artifacts, completed, tmp_path):
    path = tmp_path / "paths.json"
    (completed[0][1] / artifacts.MANIFEST).unlink()
    with pytest.raises(FileNotFoundError):
        artifacts.publish(path, *completed)
    assert not path.exists()


@pytest.mark.parametrize("fail_second_manifest", [False, True])
def test_real_completion_publishes_only_after_all_manifests(completed, tmp_path, monkeypatch, fail_second_manifest):
    monkeypatch.syspath_prepend(str(ROOT / "src"))
    monkeypatch.setitem(sys.modules, "dense_representation", SimpleNamespace(
        write_manifest=lambda *a: None, enabled=lambda _: False))
    source = ROOT / "src/adaptive_view_encoding.py"
    run = next(node for node in ast.parse(source.read_text(encoding="utf-8")).body
               if isinstance(node, ast.FunctionDef) and node.name == "run_encode")
    complete = next(node for node in ast.walk(run) if isinstance(node, ast.FunctionDef) and node.name == "complete")
    roots, selection_file, metadata, _ = completed
    path = tmp_path / "published.json"
    for root in roots:
        (root / "adaptive_views_manifest.json").unlink()

    def atomic_json(filename, value):
        if fail_second_manifest and filename.parent == roots[1]:
            raise OSError("simulated incomplete encoding")
        filename.write_text(json.dumps(value), encoding="utf-8")

    namespace = {
        "expected": metadata, "world": 2, "selection_path": selection_file,
        "selection": json.loads(selection_file.read_text()), "roots": roots,
        "encoder": SimpleNamespace(last_prompts=[]), "model_args": SimpleNamespace(),
        "options": SimpleNamespace(adaptive_artifacts_file=str(path)),
        "views": SimpleNamespace(atomic_json=atomic_json, MANIFEST="adaptive_views_manifest.json"),
        **{name: SimpleNamespace(enabled=lambda _: False) for name in ("stc", "lens", "prompt_constraints", "full_prompt")},
    }
    exec(compile(ast.Module(body=[complete], type_ignores=[]), str(source), "exec"), namespace)
    if fail_second_manifest:
        with pytest.raises(OSError, match="simulated"):
            namespace["complete"]()
        assert not path.exists()
    else:
        namespace["complete"]()
        assert path.is_file()


@pytest.mark.parametrize("corruption", ["manifest", "selection", "record"])
def test_stale_or_corrupt_paths_rejected(artifacts, completed, tmp_path, corruption):
    path = tmp_path / "paths.json"
    artifacts.publish(path, *completed)
    if corruption == "manifest":
        target = completed[0][1] / artifacts.MANIFEST
        value = json.loads(target.read_text())
        value["configuration"]["method"] = [DEFAULT, "concat"]
    elif corruption == "selection":
        target = completed[1]
        value = json.loads(target.read_text())
        value["selected"] = [{"label": "B"}]
    else:
        target = path
        value = json.loads(target.read_text())
        value["configuration"]["method"] = [DEFAULT, "concat"]
    target.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError):
        artifacts.read(path)


def test_query_cache_and_corpus_fingerprints_separate_modes(monkeypatch, artifacts):
    empty_parser = lambda parser: None
    monkeypatch.setitem(sys.modules, "stc_sparse", SimpleNamespace(add_parser=empty_parser, copy_options=lambda *a, **k: None))
    monkeypatch.setitem(sys.modules, "lens_sparse", SimpleNamespace(add_parser=empty_parser, copy_options=lambda *a, **k: None))
    monkeypatch.setitem(sys.modules, "dense_representation", SimpleNamespace(boolean=lambda value: value == "true"))
    selection = {"fingerprint": "locked-selection", "views_fingerprint": "views"}
    monkeypatch.setitem(sys.modules, "adaptive_views", SimpleNamespace(load_selection=lambda *a, **k: selection))
    pipeline = functions("three_stage_retrieval_pipeline_adaptive_views.py", {
        "_parser", "_retrieval_namespaces", "_qwen_encoding_settings", "retrieval_eol_type",
    }, {"argparse": argparse, "SimpleNamespace": SimpleNamespace, "STAGE2_EOL_TYPE": DEFAULT,
        "STAGE1_METHODS": ("clip", "blip", "sparse_manual", "sparse_concat"),
        "full_prompt": SimpleNamespace(enabled=lambda _: False),
        "prompt_constraints": SimpleNamespace(enabled=lambda _: False)})
    args = pipeline._parser().parse_args(["--dataset-name", "flickr", "--task-type", "ir", "--query-type", "image"])
    assert pipeline.retrieval_eol_type(args) == DEFAULT
    args.task_type, args.query_type, args.dataset_name = "ir", "image", "flickr"
    args.adaptive_views, args.view_selection_file = True, "selection.json"
    mass = SimpleNamespace(validate_requested=lambda *a: None, is_selection=lambda _: False)
    views = SimpleNamespace(digest=artifacts._digest, model_identity=lambda *a, **k: {"family": "llava_mistral"},
                            load_selection=lambda *a, **k: selection)
    backend = functions("adaptive_view_encoding.py", {
        "validate_encoding_method", "_query_encoding_settings", "encode_queries",
    }, {"ADAPTIVE_EOL_TYPES": (DEFAULT, ORIGINAL), "views": views, "probability_mass": mass,
        "configure_processor": lambda *a: None})
    runtime = SimpleNamespace(encoder=SimpleNamespace(config=None), processor=None, query_cache={})
    baseline = backend._query_encoding_settings(args, pipeline, selection,
                                                eol_type=DEFAULT, calculate_type="concat", sparse_manual=False)
    original = backend._query_encoding_settings(args, pipeline, selection,
                                                eol_type=ORIGINAL, calculate_type="concat", sparse_manual=False)
    assert baseline["model"]["eol_type"] == DEFAULT
    assert original["model"]["eol_type"] == ORIGINAL
    assert artifacts._digest(baseline) != artifacts._digest(original)
    for eol, settings in ((DEFAULT, baseline), (ORIGINAL, original)):
        runtime.query_cache[("adaptive", artifacts._digest(settings), None)] = eol
        assert backend.encode_queries(args, runtime, pipeline, eol_type=eol, calculate_type="concat",
                                      sparse_manual=False, progress_prefix="test") == eol


def test_stage_mode_is_used_by_encoding_checks_and_metadata():
    source = (ROOT / "src/three_stage_retrieval_pipeline_adaptive_views.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for name in ("run_stage1", "run_stage2", "_run_sparse_concat_stage1", "main"):
        nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name]
        assert nodes, name
        calls = [node for function in nodes for node in ast.walk(function) if isinstance(node, ast.Call) and
                 isinstance(node.func, ast.Name) and node.func.id == "retrieval_eol_type"]
        assert calls, name
    backend = ast.parse((ROOT / "src/adaptive_view_encoding.py").read_text(encoding="utf-8"))
    run = next(node for node in backend.body if isinstance(node, ast.FunctionDef) and node.name == "run_encode")
    complete = next(node for node in ast.walk(run) if isinstance(node, ast.FunctionDef) and node.name == "complete")
    publish = next(node for node in ast.walk(complete) if isinstance(node, ast.Call) and
                   isinstance(node.func, ast.Name) and node.func.id == "publish")
    manifest = next(node for node in ast.walk(complete) if isinstance(node, ast.Call) and
                    isinstance(node.func, ast.Attribute) and node.func.attr == "atomic_json")
    assert manifest.lineno < publish.lineno


def test_result_metadata_reports_original_mode_and_pmi(completed):
    source = ROOT / "src/three_stage_retrieval_pipeline_adaptive_views.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    metadata_dict = next(node.value for node in ast.walk(main) if isinstance(node, ast.Assign) and
                         any(isinstance(target, ast.Name) and target.id == "metadata" for target in node.targets) and
                         isinstance(node.value, ast.Dict))
    sections = {key.value: value for key, value in zip(metadata_dict.keys, metadata_dict.values)}
    namespace = {"args": SimpleNamespace(retrieval_eol_type=ORIGINAL, stage1_method="sparse_concat",
                                          modify_type="no", use_rerank_prior_cache=False),
                 "STAGE2_EOL_TYPE": DEFAULT}
    functions("three_stage_retrieval_pipeline_adaptive_views.py", {"retrieval_eol_type"}, namespace)
    for section, field, expected in (("stage1", "eol_type", ORIGINAL), ("stage2", "eol_type", ORIGINAL),
                                      ("stage3", "modify_type", "no"), ("stage3", "use_rerank_prior_cache", False)):
        value = next(value for key, value in zip(sections[section].keys, sections[section].values) if key.value == field)
        assert eval(compile(ast.Expression(body=value), str(source), "eval"), namespace) == expected
    calibration = next(node for node in ast.walk(main) if isinstance(node, ast.If) and
                       ast.unparse(node.test) == "selection.get('calibration')")
    namespace.update(metadata={"adaptive_views": {}}, selection=json.loads(completed[1].read_text()))
    exec(compile(ast.Module(body=[calibration], type_ignores=[]), str(source), "exec"), namespace)
    config = namespace["metadata"]["adaptive_views"]["selection_calibration"]
    assert config["method"] == "pmi" and config["prior_prompt_mode"] == "instruction_only"


@pytest.fixture
def full_corpus_search(tmp_path, monkeypatch, artifacts):
    features = {}
    for name in ("stc_sparse", "lens_sparse"):
        features[name] = SimpleNamespace(FIELD=name, add_parser=lambda *a, **k: None,
            copy_options=lambda *a, **k: None, bind_search=lambda *a: None, validate_bundle=lambda *a: None)
        monkeypatch.setitem(sys.modules, name, features[name])
    monkeypatch.setitem(sys.modules, "dense_representation", SimpleNamespace(
        boolean=lambda value: value == "true", validate_artifact=lambda *a: None))
    pipeline = functions("three_stage_retrieval_pipeline_adaptive_views.py", {"_parser", "_retrieval_namespaces"}, {
        "argparse": argparse, "SimpleNamespace": SimpleNamespace, "STAGE2_EOL_TYPE": DEFAULT,
        "STAGE1_METHODS": ("clip", "blip", "sparse_manual", "sparse_concat"),
        "full_prompt": SimpleNamespace(enabled=lambda _: False),
        "prompt_constraints": SimpleNamespace(enabled=lambda _: False),
    })
    view_namespace = dict(Path=Path, json=json, digest=artifacts._digest,
                          artifact_root=lambda path: Path(path), MANIFEST=artifacts.MANIFEST)
    functions("adaptive_views.py", {"differences", "require_equal", "validate_artifact"}, view_namespace)
    view_namespace.update(SelectionFileResolver=lambda *a: SimpleNamespace(),
        is_overlap_selection=lambda _: False,
        load_selection=lambda path, **kwargs: json.loads(Path(path).read_text(encoding="utf-8")),
        validate_index_selection=lambda root, manifest, **kwargs:
            json.loads(Path(manifest["selection_file"]).read_text(encoding="utf-8")))
    views = SimpleNamespace(**view_namespace)
    search = functions("search_adaptive_views.py", {"parser", "boolean", "prepare", "direction"}, {
        "argparse": argparse, "copy": copy, "math": math, "Path": Path, "json": json, "pickle": pickle, "views": views,
        "stc": features["stc_sparse"], "lens": features["lens_sparse"], "artifact_root": lambda path: Path(path),
        "resolve_sparse_index": lambda path: Path(path) / "collection" / "index",
        "Experiment": lambda *values: SimpleNamespace(**dict(zip(
            ("args", "pipeline_args", "selection", "selection_path", "expected", "dense_root", "sparse_root", "index"), values))),
    }, constants=("ENCODING_DEFAULTS", "TASKS"))
    validator = functions("adaptive_view_encoding.py", {"validate_encoding_method"}, {
        "ADAPTIVE_EOL_TYPES": (DEFAULT, ORIGINAL),
    }).validate_encoding_method
    identity = {"family": "llava_mistral", "dtype": "float16", "checkpoint": {"path": "/models/llava"},
                "vision": {"llava_resolution": {"mode": "base"}}}
    selection = {"fingerprint": "locked", "views_fingerprint": "views", "selected": [{"label": "A"}],
                 "calibration": {"configuration": {"method": "pmi", "prior_prompt_mode": "instruction_only"}}}
    selection_file = tmp_path / "view_selection.json"
    selection_file.write_text(json.dumps(selection), encoding="utf-8")

    def metadata(model, data, task, kind, actual_identity, locked):
        configuration = dict(method=[model.eol_type, model.calculate_type], dataset=data.dataset_name,
            task=task, split=data.dataset_split, kind=kind, model=actual_identity,
            encoding={name: getattr(data, name) for name in (*search.ENCODING_DEFAULTS, "sparse_manual")})
        return dict(configuration=configuration, fingerprint=artifacts._digest(configuration))

    backend = SimpleNamespace(pipeline_module=lambda: pipeline, metadata=metadata,
        validate_encoding_method=validator, load_model_identity=lambda *a, **k: identity)
    monkeypatch.setitem(sys.modules, "adaptive_view_encoding", backend)
    monkeypatch.setitem(sys.modules, "qwen_vl_pipeline", SimpleNamespace(
        corpus_kind=lambda task, query: "text" if query == "image" else "image"))

    def index_check(root, manifest, expected, **kwargs):
        views.require_equal(manifest["configuration"], expected, "Sparse corpus")

    monkeypatch.setitem(sys.modules, "adaptive_sparse_index_batch", SimpleNamespace(check_artifact=index_check))

    def setup(eol=DEFAULT, query="image", family="llava_mistral"):
        identity["family"] = family
        roots = [tmp_path / "dense", tmp_path / "sparse"]
        for root in roots:
            root.mkdir(exist_ok=True)
        (roots[0] / "corpus_0.pkl").touch()
        (roots[1] / "collection" / "index").mkdir(parents=True, exist_ok=True)
        encoding = {**search.ENCODING_DEFAULTS, "sparse_length": 30, "text_sparse_length": 30,
                    "image_sparse_length": 30, "sparse_manual": False}
        configuration = dict(method=[eol, "concat"], dataset="flickr", task="ir", split="test",
            kind="text" if query == "image" else "image", model=copy.deepcopy(identity), encoding=encoding)
        for root in roots:
            manifest = dict(configuration=configuration, fingerprint=artifacts._digest(configuration), num_shards=1,
                            selection_file=str(selection_file))
            (root / artifacts.MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
        args = search.parser().parse_args(["--passage_reps", str(roots[0]), "--sparse_index", str(roots[1]),
                                           "--query_type", query, "--alpha", "0.7"])
        return args, roots

    def rewrite(roots, edit):
        for root in roots:
            path = root / artifacts.MANIFEST
            manifest = json.loads(path.read_text(encoding="utf-8"))
            edit(manifest["configuration"])
            manifest["fingerprint"] = artifacts._digest(manifest["configuration"])
            path.write_text(json.dumps(manifest), encoding="utf-8")

    return SimpleNamespace(search=search, backend=backend, pipeline=pipeline, setup=setup, rewrite=rewrite)


@pytest.mark.parametrize("eol", [DEFAULT, ORIGINAL])
@pytest.mark.parametrize("query", ["image", "text"])
def test_full_corpus_preflight_and_query_encoder_share_method(full_corpus_search, eol, query):
    probe = full_corpus_search
    args, _ = probe.setup(eol, query)
    experiment = probe.search.prepare(args, backend=probe.backend)
    assert experiment.pipeline_args.retrieval_eol_type == eol
    assert experiment.expected["configuration"]["method"] == [eol, "concat"]
    assert experiment.args.sparse_length == 30
    # Execute the production query-dispatch closure without loading weights or Lucene.
    source = ROOT / "src/search_adaptive_views.py"
    execute = next(node for node in ast.parse(source.read_text(encoding="utf-8")).body
                   if isinstance(node, ast.FunctionDef) and node.name == "execute")
    retrieve = next(node for node in ast.walk(execute) if isinstance(node, ast.FunctionDef) and node.name == "retrieve")
    observed = []
    probe.backend.configure_processor = lambda *a: None

    def encoder(model, processor, model_args, data, task, selection):
        observed.append(model_args.eol_type)
        return SimpleNamespace(last_prompts=["example"], method=model_args.eol_type)

    probe.backend.AdaptiveEncoder = encoder
    probe.pipeline.build_dataset = lambda args: None
    probe.pipeline._query_records = lambda args, dataset: [{"id": "q"}]
    probe.pipeline._qwen_query_records = lambda args, records: records
    namespace = dict(api=probe.pipeline, backend=probe.backend,
        runtime=SimpleNamespace(encoder=SimpleNamespace(config=None), processor=None),
        context=SimpleNamespace(rank=0, local_rank=0, world_size=1),
        CorpusSearch=lambda *a, **k: SimpleNamespace(retrieve=lambda encoder, records: ({"q": {}}, {"q": {}}), close=lambda: None))
    exec(compile(ast.Module(body=[retrieve], type_ignores=[]), str(source), "exec"), namespace)
    runs, examples, ids = namespace["retrieve"](experiment)
    assert observed == [eol] and ids == ["q"] and examples == ["example"]


@pytest.mark.parametrize("method,manual,family", [
    (None, False, "llava_mistral"),
    (["prompteol", "concat"], False, "llava_mistral"),
    ([DEFAULT, "separate"], False, "llava_mistral"),
    ([DEFAULT, "concat"], True, "llava_mistral"),
    ([ORIGINAL, "concat"], False, "qwen25"),
])
def test_full_corpus_rejects_unsupported_manifest_method(full_corpus_search, method, manual, family):
    probe = full_corpus_search
    args, roots = probe.setup(family=family)
    probe.rewrite(roots, lambda config: config.update(method=method, encoding={**config["encoding"], "sparse_manual": manual}))
    with pytest.raises(ValueError):
        probe.search.prepare(args, backend=probe.backend)


def test_full_corpus_rejects_mismatched_dense_sparse_modes(full_corpus_search):
    probe = full_corpus_search
    args, roots = probe.setup()
    probe.rewrite(roots[1:], lambda config: config.update(method=[ORIGINAL, "concat"]))
    with pytest.raises(ValueError, match="Dense/sparse corpus configuration"):
        probe.search.prepare(args, backend=probe.backend)


@pytest.mark.parametrize("missing", ["passage_reps", "sparse_index"])
def test_similarity_rejects_single_index_before_loading_model(full_corpus_search, missing):
    probe = full_corpus_search
    args, _ = probe.setup()
    args.retrieval_similarity_stats = True
    setattr(args, missing, None)
    with pytest.raises(ValueError, match="requires both passage_reps and sparse_index"):
        probe.search.prepare(args, identity_loader=lambda *a, **k: pytest.fail("model identity loaded"),
                             backend=probe.backend)


@pytest.mark.parametrize("eol", [DEFAULT, ORIGINAL])
def test_similarity_dry_run_does_not_write_reports(full_corpus_search, tmp_path, monkeypatch, capsys, eol):
    monkeypatch.syspath_prepend(str(ROOT / "src"))
    probe = full_corpus_search
    args, _ = probe.setup(eol)
    args.dry_run = args.retrieval_similarity_stats = True
    args.search_output_root = str(tmp_path / "planned")
    namespace = probe.search.prepare.__globals__
    for feature in (namespace["stc"], namespace["lens"]):
        feature.mark_path = lambda path, config: path
    sys.modules["dense_representation"].mode_path = lambda path, *a: path
    probe.pipeline._initialise_distributed = lambda *a: pytest.fail("dry run initialized runtime")
    search = functions("search_adaptive_views.py", {
        "execute", "prepare_validation", "run_configuration", "output_location", "similarity_configuration",
    }, namespace)
    search.execute(args)
    plan = json.loads(capsys.readouterr().out)
    assert plan["configuration"]["retrieval_similarity_stats"]["top_k"] == [10, 20, 50, 100]
    assert plan["configuration"]["encoding"]["method"] == [eol, "concat"]
    assert not Path(args.search_output_root).exists()


def test_similarity_uses_all_gathered_queries(full_corpus_search, tmp_path, monkeypatch):
    from types import ModuleType
    monkeypatch.syspath_prepend(str(ROOT / "src"))
    probe = full_corpus_search
    args, _ = probe.setup()
    experiment = probe.search.prepare(args, backend=probe.backend)
    search = functions("search_adaptive_views.py", {
        "gather_run", "flat_run", "write_similarity_report", "similarity_configuration",
    }, probe.search.prepare.__globals__, constants=("SIMILARITY_REPORT_NAME",))
    dense_parts = [{"q0": {"docs": {"a": 2., "b": 1.}}}, {"q1": {"docs": {"a": 2., "b": 1.}}}]
    sparse_parts = [{"q0": {"docs": {"a": 2., "b": 1.}}}, {"q1": {"docs": {"a": 1., "b": 2.}}}]
    parts = iter((dense_parts, sparse_parts))
    torch, dist = ModuleType("torch"), ModuleType("torch.distributed")
    torch.distributed = dist
    dist.all_gather_object = lambda gathered, local: gathered.__setitem__(slice(None), next(parts))
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.distributed", dist)
    context = SimpleNamespace(is_distributed=True, world_size=2)
    dense = search.gather_run(dense_parts[0], context)
    sparse = search.gather_run(sparse_parts[0], context)
    result = search.write_similarity_report(experiment, {
        "dense": search.flat_run(dense), "sparse": search.flat_run(sparse),
    }, tmp_path / "report")
    json.dumps(result, allow_nan=False)
    with Path(result["file"]).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert {row["query_id"] for row in rows if row["table_type"] == "per_query"} == {"q0", "q1"}
    summary, = [row for row in rows if row["table_type"] == "summary"]
    assert float(summary["Spearman@10"]) == pytest.approx(0.)
    assert int(summary["queries_with_dense_and_sparse"]) == 2


def test_similarity_is_not_written_by_other_workers():
    source = ROOT / "src/search_adaptive_views.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    execute = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "execute")
    finish = next(node for node in ast.walk(execute) if isinstance(node, ast.FunctionDef) and node.name == "finish")
    namespace = dict(context=SimpleNamespace(is_main=False),
                     write_similarity_report=lambda *a: pytest.fail("non-main worker wrote a report"))
    exec(compile(ast.Module(body=[finish], type_ignores=[]), str(source), "exec"), namespace)
    namespace["finish"]()


@pytest.mark.parametrize("eol", [DEFAULT, ORIGINAL])
@pytest.mark.parametrize("query", ["image", "text"])
@pytest.mark.parametrize("stats", [False, True])
@pytest.mark.parametrize("overwrite", [False, True])
def test_full_corpus_writes_three_runs_with_fixed_alpha(full_corpus_search, tmp_path, monkeypatch, eol, query, stats, overwrite):
    monkeypatch.syspath_prepend(str(ROOT / "src"))
    probe = full_corpus_search
    args, roots = probe.setup(eol, query)
    args.search_output_root = str(tmp_path / "results")
    args.retrieval_similarity_stats = stats
    if overwrite:
        args.overwrite = True
        args.output_dir = str(tmp_path / "results" / "explicit")
        Path(args.output_dir).mkdir(parents=True)
        (Path(args.output_dir) / "dense_sparse_similarity_test.csv").write_text("stale report")
        (Path(args.output_dir) / "test_runs_no_manual.pkl").write_bytes(b"stale report")
    context = SimpleNamespace(rank=0, local_rank=0, world_size=1, is_main=True, is_distributed=False)
    probe.pipeline._initialise_distributed = lambda _: context
    probe.pipeline._shutdown_distributed = lambda *a, **k: None
    probe.pipeline._load_retrieval_runtime = lambda *a: SimpleNamespace(encoder=SimpleNamespace(config=None), processor=None)
    probe.pipeline._release_retrieval_runtime = lambda *a: None
    probe.pipeline.build_dataset = lambda _: None
    probe.pipeline._query_records = lambda *a: [{"id": "q"}]
    probe.pipeline._qwen_query_records = lambda args, records: records
    probe.pipeline._build_recall_evaluation = lambda *a: {"q": "d1"}

    def evaluate_recall(run, gold, **kwargs):
        ranking = sorted(run["q"], key=run["q"].get, reverse=True)
        return [{"category": "all", "recall": {k: float(gold["q"] in ranking[:k]) for k in (1, 5, 10)}}]

    probe.pipeline.evaluate_recall = evaluate_recall
    probe.backend.configure_processor = lambda *a: None
    observed = []

    def encoder(model, processor, model_args, data, task, selection):
        observed.append(model_args.eol_type)
        return SimpleNamespace(last_prompts=["query prompt"])

    probe.backend.AdaptiveEncoder = encoder
    dense = {"q": {"docs": {"d1": 10., "d2": 4., "d3": 0.}, "min_score": 0., "max_score": 10.}}
    sparse = {"q": {"docs": {"d1": 0., "d2": 10., "d3": 0.}, "min_score": 0., "max_score": 10.}}
    namespace = probe.search.prepare.__globals__
    namespace.update(csv=csv, CorpusSearch=lambda *a, **k:
                     SimpleNamespace(retrieve=lambda *a: (dense, sparse), close=lambda: None))
    namespace["views"].atomic_json = lambda path, value: path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")
    for feature in (namespace["stc"], namespace["lens"]):
        feature.enabled = lambda _: False
        feature.mark_path = lambda path, _: path
    sys.modules["dense_representation"].mode_path = lambda path, *a: path
    # Completing the run requires the saved per-view probabilities in metadata.
    selection_file = Path(json.loads((roots[0] / "adaptive_views_manifest.json").read_text())["selection_file"])
    selection = json.loads(selection_file.read_text())
    selection.update(pool=[{"label": "A"}], scores=[1.])
    selection_file.write_text(json.dumps(selection), encoding="utf-8")
    hybrid = functions("hybrid.py", {"fuse", "write_trec_run"}, {})
    monkeypatch.setitem(sys.modules, "hybrid", hybrid)
    search = functions("search_adaptive_views.py", {
        "execute", "collective", "gather_run", "prepare_validation", "run_configuration", "output_location",
        "flat_run", "evaluate", "write_csv", "metric_rows", "similarity_configuration", "write_similarity_report",
    }, namespace, constants=("SIMILARITY_REPORT_NAME", "OUTPUT_FILES"))

    def unexpected_sweep(*a):
        pytest.fail("Fixed alpha must not run the alpha sweep")

    namespace["select_alpha"] = unexpected_sweep
    search.execute(args)
    metadata_path, = Path(args.search_output_root).rglob("run_metadata.json")
    output = metadata_path.parent
    metadata = json.loads(metadata_path.read_text())
    assert metadata["status"] == "complete" and observed == [eol]
    assert metadata["configuration"]["encoding"]["method"] == [eol, "concat"]
    assert metadata["configuration"]["direction"] == ("image_to_text" if query == "image" else "text_to_image")
    assert metadata["configuration"]["automatic_alpha"] is False
    assert metadata["alpha"] == .7 and metadata["alpha_selection_source"] == "fixed"
    assert metadata["query_count"] == 1 and metadata["selected_views"][0]["probability"] == 1.
    assert not (output / "alpha_sweep.csv").exists()
    for name in ("dense", "sparse", "hybrid"):
        assert (output / f"{name}_run.json").is_file() and (output / f"{name}_run.trec").is_file()
    fused = json.loads((output / "hybrid_run.json").read_text())["q"]
    assert fused == pytest.approx({"d1": .7, "d2": .58, "d3": 0.})
    report = json.loads((output / "recall_metrics.json").read_text())
    assert report["metrics"]["dense"]["Recall@1"] == 100.
    assert report["metrics"]["sparse"]["Recall@1"] == 0.
    assert report["metrics"]["hybrid"]["Recall@1"] == 100.
    with (output / "recall_metrics.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert {row["method"] for row in rows} == {"dense", "sparse", "hybrid"}
    assert all(row["alpha"] == "0.7" for row in rows)
    similarity = output / "dense_sparse_similarity_test.csv"
    saved_runs = output / "test_runs_no_manual.pkl"
    assert similarity.is_file() == stats
    assert saved_runs.is_file() == stats
    if stats:
        assert metadata["retrieval_similarity_stats"]["file"] == str(similarity.resolve())
        assert metadata["retrieval_similarity_stats"]["run_file"] == str(saved_runs.resolve())
        assert metadata["configuration"]["retrieval_similarity_stats"] == {
            "version": 2, "top_k": [10, 20, 50, 100], "rbo_p": .9}
        with similarity.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        summary, = [row for row in rows if row["table_type"] == "summary"]
        assert summary["condition"] == "no_manual"
        assert float(summary["Overlap@10"]) == 1.
        assert float(summary["Spearman@10"]) == pytest.approx(0.)
        info = {row["item"]: row["value"] for row in rows if row["table_type"] == "metadata"}
        assert info["no_manual_selection_fingerprint"] == "locked"
        assert info["no_manual_run_metadata_file"] == str(metadata_path)
        assert info["no_manual_run_file"] == str(saved_runs)
        assert info["no_manual_sparse_manual"] == "False" and info["no_manual_llava_image_mode"] == "base"
        with saved_runs.open("rb") as stream:
            payload = pickle.load(stream)
        assert payload["dense_run"]["q"]["docs"] == dense["q"]["docs"]
        assert payload["sparse_run"]["q"]["docs"] == sparse["q"]["docs"]
        assert payload["metadata"]["adaptive_views"] is True
    else:
        assert "retrieval_similarity_stats" not in metadata["configuration"]
        assert "retrieval_similarity_stats" not in metadata


@pytest.mark.parametrize("eol", [DEFAULT, ORIGINAL])
@pytest.mark.parametrize("query", ["image", "text"])
def test_adaptive_report_matches_ordinary_and_offline_reports(full_corpus_search, tmp_path, monkeypatch, capsys, eol, query):
    monkeypatch.syspath_prepend(str(ROOT / "src"))
    import compute_retrieval_similarity as metrics
    probe = full_corpus_search
    args, _ = probe.setup(eol, query)
    experiment = probe.search.prepare(args, backend=probe.backend)
    search = functions("search_adaptive_views.py", {
        "write_similarity_report", "similarity_configuration", "run_configuration",
    }, probe.search.prepare.__globals__, constants=("SIMILARITY_REPORT_NAME",))
    disabled = search.run_configuration(experiment)
    assert "retrieval_similarity_stats" not in disabled
    experiment.args.retrieval_similarity_stats = True
    enabled = search.run_configuration(experiment)
    version1 = copy.deepcopy(enabled)
    version1["retrieval_similarity_stats"]["version"] = 1
    assert enabled["encoding"] == disabled["encoding"]
    assert probe.search.prepare.__globals__["views"].digest(version1) != probe.search.prepare.__globals__["views"].digest(enabled)
    disabled_again = copy.deepcopy(enabled)
    disabled_again.pop("retrieval_similarity_stats")
    assert disabled_again == disabled

    scores = {"a": 3., "b": 2., "c": 1.}
    dense = {"same": scores, "reverse": scores, "ties": dict.fromkeys(scores, 1.),
             "empty": {}, "one_common": {"a": 2., "b": 1.}, "disjoint": {"a": 2., "b": 1.}}
    sparse = {"same": scores, "reverse": {"a": 1., "b": 2., "c": 3.}, "ties": dict.fromkeys(scores, 1.),
              "empty": scores, "one_common": {"a": 2., "c": 1.}, "disjoint": {"c": 2., "d": 1.}}
    adaptive_output = tmp_path / "adaptive"
    result = search.write_similarity_report(experiment, {"dense": dense, "sparse": sparse}, adaptive_output)
    json.dumps(result, allow_nan=False)

    ordinary = functions("search_similarity.py", {
        "_save_test_runs_for_similarity", "_gather_test_run_for_similarity", "_similarity_stats_configuration",
    }, dict(os=os, pickle=pickle, ENABLE_RETRIEVAL_SIMILARITY_STATS=True,
            dist=SimpleNamespace(get_world_size=lambda: 1, get_rank=lambda: 0,
                                 all_gather_object=lambda gathered, run: gathered.__setitem__(0, run))),
        constants=("RETRIEVAL_SIMILARITY_STATS_DIR_NAME", "RETRIEVAL_SIMILARITY_CONFIGURATIONS"))
    ordinary_output = tmp_path / "ordinary" / "0_no_manual_30_sum_llava_base" / "best.csv"
    ordinary._save_test_runs_for_similarity(
        {qid: {"docs": values} for qid, values in dense.items()},
        {qid: {"docs": values} for qid, values in sparse.items()}, str(ordinary_output),
        SimpleNamespace(eol_type=eol, calculate_type="concat"),
        SimpleNamespace(dataset_name="flickr", sparse_manual=False, llava_image_mode="base"),
        SimpleNamespace(query_type=query, depth=1000))
    ordinary_csv, = (tmp_path / "ordinary").rglob("dense_sparse_similarity_test.csv")
    adaptive_csv = Path(result["file"])

    def read(path):
        with path.open(encoding="utf-8-sig", newline="") as stream:
            return list(csv.DictReader(stream))

    ordinary_rows, adaptive_rows = read(ordinary_csv), read(adaptive_csv)
    assert list(ordinary_rows[0]) == list(adaptive_rows[0])
    for row_type in ("summary", "per_query"):
        assert [row for row in ordinary_rows if row["table_type"] == row_type] == [
            row for row in adaptive_rows if row["table_type"] == row_type]
    info = {row["item"]: row["value"] for row in adaptive_rows if row["table_type"] == "metadata"}
    reference = {row["item"]: row["value"] for row in ordinary_rows if row["table_type"] == "metadata"}
    assert set(reference) <= set(info)
    for key, value in reference.items():
        if key not in {"no_manual_run_file", "no_manual_test_run_directory"}:
            assert info[key] == value
    assert info["no_manual_run_file"] == result["run_file"]
    assert info["no_manual_test_run_directory"] == str(adaptive_output)
    assert info["no_manual_adaptive_views"] == "True"
    loaded_dense, loaded_sparse, metadata = metrics.load_saved_runs(result["run_file"])
    assert loaded_dense == dense and loaded_sparse == sparse
    assert metadata["sparse_manual"] is False and metadata["adaptive_views"] is True
    offline = metrics.compute_from_directory(adaptive_output, output_path=tmp_path / "offline.csv", allow_partial=True)
    assert Path(offline).read_bytes() == adaptive_csv.read_bytes()
    log = capsys.readouterr().out
    assert f"Saved test runs for similarity analysis to: {result['run_file']}" in log
    assert f"Saved dense/sparse similarity report to: {result['file']}" in log


BASH = shutil.which("bash") if os.name != "nt" else next(
    (path for path in ("D:/Git/bin/bash.exe", "C:/Program Files/Git/bin/bash.exe") if Path(path).is_file()), None)


@pytest.mark.skipif(BASH is None, reason="Bash not installed")
@pytest.mark.parametrize("target,query", [("text", "image"), ("image", "text")])
@pytest.mark.parametrize("search_phase", ["search", "search_similarity"])
@pytest.mark.parametrize("suffix,eol,namespace", [
    ("concat_origin_text", ORIGINAL, "adaptive_origin_text"),
    ("adaptive_concat", DEFAULT, "adaptive_concat"),
])
@pytest.mark.parametrize("mass_env,mass,threshold,selection_name", [
    ({}, "true", "0.9", "flickr_view_selection_pm_0.9.json"),
    ({"USE_VIEW_PROBABILITY_MASS": "false"}, "false", "0.9", "flickr_view_selection.json"),
    ({"USE_VIEW_PROBABILITY_MASS": "TRUE", "VIEW_PROBABILITY_MASS_THRESHOLD": "0.95"},
     "true", "0.95", "flickr_view_selection_pm_0.95.json"),
    ({"VIEW_SELECTION_FILE": "existing probability source.json", "VIEW_PROBABILITY_MASS_THRESHOLD": "0.8"},
     "true", "0.8", "existing probability source.json"),
])
def test_three_script_handoff_and_full_corpus_arguments(tmp_path, monkeypatch, target, query,
                                                       suffix, eol, namespace, mass_env, mass, threshold, selection_name, search_phase):
    (tmp_path / "scripts").mkdir()
    (tmp_path / "bin").mkdir()
    (tmp_path / "logs").mkdir()
    phases = ("encode", "sparse_index", search_phase)
    for phase in phases:
        name = f"{phase}_llava_base_{suffix}.sh"
        text = (ROOT / "scripts" / name).read_text(encoding="utf-8")
        with (tmp_path / "scripts" / name).open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(text.replace("ENCODE_TYPE=text", f"ENCODE_TYPE={target}", 1))
    mock = '''#!/usr/bin/env bash
if [[ "$2" == src/adaptive_artifacts.py ]]; then
  [[ "${MOCK_PATH_FAIL:-0}" != 1 ]] || exit 2
  case "$4" in
    dense_root) echo 'saved dense' ;;
    sparse_root) echo 'saved sparse' ;;
    selection_file) echo 'saved selection.json' ;;
    *) exit 3 ;;
  esac
  exit 0
fi
printf '%s\\0' "$@" > "logs/$MOCK_PHASE.args"
printf '%s' "${OPENAI_API_KEY-unset}" > "logs/$MOCK_PHASE.env"
'''
    for binary in ("python", "deepspeed"):
        path = tmp_path / "bin" / binary
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(mock)
        path.chmod(0o755)
    bin_path = (tmp_path / "bin").as_posix()
    if os.name == "nt":
        bin_path = "/" + bin_path[0].lower() + bin_path[2:]
    captured = {}
    for phase in phases:
        env = dict(os.environ, MOCK_PHASE=phase, MSYS_NO_PATHCONV="1")
        env.pop("OPENAI_API_KEY", None)
        for name in ("USE_VIEW_PROBABILITY_MASS", "VIEW_PROBABILITY_MASS_THRESHOLD", "VIEW_SELECTION_FILE"):
            env.pop(name, None)
        env.update(mass_env)
        command = 'export PATH=' + shlex.quote(bin_path) + ':/usr/bin:/bin:"$PATH"; exec /usr/bin/bash scripts/' + phase + '_llava_base_' + suffix + '.sh'
        result = subprocess.run([BASH, "-c", command], cwd=tmp_path, env=env,
                                capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stdout + result.stderr
        captured[phase] = (tmp_path / "logs" / f"{phase}.args").read_bytes().decode().rstrip("\0").split("\0")
    encode = captured["encode"]
    value = lambda flag: encode[encode.index(flag) + 1]
    assert "src/encode_adaptive_views.py" in encode and "--adaptive_views" in encode
    assert value("--eol_type") == eol and value("--calculate_type") == "concat"
    assert value("--encode_type") == target
    assert value("--adaptive_artifacts_file").endswith(f"flickr_{target}.json")
    assert f"/output/{namespace}/" in value("--adaptive_artifacts_file")
    assert value("--view_selection_file").endswith(selection_name)
    assert value("--view_probability_mass") == mass
    assert value("--view_probability_mass_threshold") == threshold
    assert value("--view_selection_pmi") == "true"
    assert value("--view_pmi_prior_prompt_mode") == "instruction_only"
    assert value("--view_selection_modality") == "both"
    assert value("--view_selection_rounds") == "20" and value("--view_selection_seed") == "42"
    for flag in ("--view_selection_hierarchical", "--view_selection_token_variants",
                 "--view_selection_prefix_tokens", "--view_selection_shuffle", "--sparse_manual"):
        assert value(flag) == "false"
    index = captured["sparse_index"]
    assert index[:3] == ["-B", "src/adaptive_views.py", "index"]
    assert index[index.index("--sparse-root") + 1] == "saved sparse"
    # Parse the captured command using the independent search argument parser.
    for module in ("stc_sparse", "lens_sparse"):
        monkeypatch.setitem(sys.modules, module, SimpleNamespace(add_parser=lambda p, **k: None))
    parser = functions("search_adaptive_views.py", {"parser", "boolean"}, {
        "argparse": argparse, "stc": sys.modules["stc_sparse"], "lens": sys.modules["lens_sparse"],
    }, constants=("ENCODING_DEFAULTS",)).parser()
    search = captured[search_phase]
    entry = search.index("src/search_adaptive_views.py")
    args = parser.parse_args(search[entry + 1:])
    assert args.query_type == query
    assert args.retrieval_similarity_stats == (search_phase == "search_similarity")
    assert args.view_probability_mass is None and args.view_probability_mass_threshold is None
    assert args.passage_reps == "saved dense" and args.sparse_index == "saved sparse"
    assert args.view_selection_file == "saved selection.json"
    assert args.depth == 1000 and args.alpha == .7 and args.use_fixed_alpha
    assert args.llava_image_mode == "base" and args.dtype == "float16"
    assert args.view_selection_pmi is None and args.view_pmi_prior_prompt_mode is None
    assert args.search_output_root == "/root/gpufree-share/adaptive_search_results" and args.output_dir is None
    assert not any(value.startswith(("--stage", "--ranker", "--rerank", "--modify")) for value in search)
    assert (tmp_path / "logs" / f"{search_phase}.env").read_text() == "local-placeholder"
    # A missing/invalid handoff must stop before launching an index or model.
    for phase in ("sparse_index", search_phase):
        log = tmp_path / "logs" / f"{phase}.args"
        log.unlink()
        env = dict(os.environ, MOCK_PHASE=phase, MOCK_PATH_FAIL="1", MSYS_NO_PATHCONV="1")
        command = 'export PATH=' + shlex.quote(bin_path) + ':/usr/bin:/bin:"$PATH"; exec /usr/bin/bash scripts/' + phase + '_llava_base_' + suffix + '.sh'
        result = subprocess.run([BASH, "-c", command], cwd=tmp_path, env=env,
                                capture_output=True, text=True, timeout=30)
        assert result.returncode != 0 and not log.exists()
