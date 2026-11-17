"""CPU regressions for the independent search launcher (no model weights needed)."""
import ast
import csv
import json
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


def definitions(relative_path, namespace, constants=()):
    """Load production helpers without importing the GPU entry point's dependencies."""
    path = ROOT / relative_path
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) or (
        isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in constants
            for target in node.targets))]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return SimpleNamespace(**namespace)


@pytest.fixture
def resolution():
    return definitions("src/llava_resolution.py", {"Path": Path, "json": json},
                       ("MODES", "VERSION", "SUFFIX", "MANIFEST_SUFFIX"))


def config(model_type="llava_next"):
    return SimpleNamespace(
        model_type=model_type, vision_feature_select_strategy="default",
        vision_config=SimpleNamespace(image_size=336, patch_size=14,
                                      model_type="clip_vision_model"))


def manifest(resolution, path, mode):
    path.mkdir(parents=True, exist_ok=True)
    report = resolution.resolution_metadata(config(), mode)
    resolution.sidecar_path(path).write_text(json.dumps(report), encoding="utf-8")
    return report


def test_checkpoint_preflight(resolution, monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoConfig=SimpleNamespace(from_pretrained=lambda _: config("llava"))))
    with pytest.raises(ValueError, match="LLaVA-NeXT"):
        resolution.validate_checkpoint("checkpoint", "base")
    resolution.validate_checkpoint("checkpoint", "anyres")


def test_image_manifest_and_lucene_paths(resolution, tmp_path):
    corpus = tmp_path / "images_llava_base"
    report = manifest(resolution, corpus, "base")
    index = corpus / "collection" / "index"
    index.mkdir(parents=True)
    shard = corpus / "corpus.0.pkl"
    shard.touch()
    for path in (corpus, corpus / "collection", index, shard):
        resolution.validate_artifact(path, report)
    for path in (corpus, corpus / "collection", index):
        assert Path(resolution.lucene_index_dir(path)) == index
    with pytest.raises(ValueError, match="mismatch"):
        resolution.validate_artifact(corpus, resolution.resolution_metadata(config(), "anyres"))
    missing = tmp_path / "legacy"
    missing.mkdir()
    with pytest.raises(ValueError, match="Missing matching"):
        resolution.validate_artifact(missing, report)
    resolution.validate_artifact(missing, report, has_images=False)
    resolution.validate_artifact(missing, resolution.resolution_metadata(config(), "anyres"))
    direct = tmp_path / "direct" / "index"
    direct.mkdir(parents=True)
    assert Path(resolution.lucene_index_dir(direct.parent)) == direct


def test_search_checks_used_image_artifacts_and_reuses_text(resolution, tmp_path):
    image = tmp_path / "image"
    report = manifest(resolution, image, "base")
    wrong = tmp_path / "wrong"
    manifest(resolution, wrong, "anyres")
    # Exercise the real configure_search orchestration with a CPU-only model setup.
    resolution.configure_search.__globals__["configure"] = lambda *args: "base processor"
    model = SimpleNamespace(_llava_resolution_report=report)
    data = SimpleNamespace(llava_image_mode="base")
    search = SimpleNamespace(query_type="text", passage_reps=str(image),
                             sparse_index=str(image), val_passage_reps=str(wrong),
                             val_sparse_index=str(wrong), use_fixed_alpha=True)
    assert resolution.configure_search(model, None, data, search, "ir") == "base processor"
    search.use_fixed_alpha = False
    with pytest.raises(ValueError, match="mismatch"):
        resolution.configure_search(model, None, data, search, "ir")
    search.query_type = "image"
    # Pure-text candidates are shared even when their sidecar reports AnyRes.
    search.passage_reps = search.sparse_index = str(wrong)
    assert resolution.configure_search(model, None, data, search, "ir") == "base processor"


@pytest.mark.parametrize("mode", ["anyres", "base"])
def test_saved_runs_and_reports_are_isolated(resolution, tmp_path, monkeypatch, mode):
    monkeypatch.syspath_prepend(str(ROOT / "src"))
    dist = SimpleNamespace(get_world_size=lambda: 1, get_rank=lambda: 0,
                           all_gather_object=lambda gathered, run: gathered.__setitem__(0, run))
    helpers = definitions("src/search_similarity.py", {
        "os": os, "pickle": pickle, "dist": dist, "ENABLE_RETRIEVAL_SIMILARITY_STATS": True,
    }, ("RETRIEVAL_SIMILARITY_STATS_DIR_NAME", "RETRIEVAL_SIMILARITY_CONFIGURATIONS"))
    model = SimpleNamespace(eol_type="prompteol", calculate_type="separate")
    search = SimpleNamespace(query_type="image", depth=1000)
    run = {"q": {"docs": {str(i): float(20 - i) for i in range(20)}}}
    # Keep both modes in this test's same parent to catch accidental file reuse.
    for current_mode in ("anyres", "base"):
        for manual in (False, True):
            label = "manual" if manual else "no_manual"
            leaf = tmp_path / f"0_{label}_30_sum"
            output = Path(resolution.search_result_dir(str(leaf), current_mode)) / "best.csv"
            data = SimpleNamespace(dataset_name="flickr", sparse_manual=manual,
                                   llava_image_mode=current_mode)
            helpers._save_test_runs_for_similarity(run, run, str(output), model, data, search)
            if not manual:
                single = next(path for path in tmp_path.rglob("dense_sparse_similarity_test.csv")
                              if path.parent.name.endswith("_llava_base") == (current_mode == "base"))
                with single.open(encoding="utf-8-sig", newline="") as handle:
                    conditions = {row["condition"] for row in csv.DictReader(handle) if row["table_type"] == "summary"}
                assert conditions == {"no_manual"}
    reports = list(tmp_path.rglob("dense_sparse_similarity_test.csv"))
    assert len(reports) == 2
    report_path = next(path for path in reports if path.parent.name.endswith("_llava_base") == (mode == "base"))
    with report_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    summaries = [row for row in rows if row["table_type"] == "summary"]
    assert {row["condition"] for row in summaries} == {"manual", "no_manual", "manual - no_manual"}
    for row in summaries:
        expected = 0 if row["condition"] == "manual - no_manual" else 1
        assert float(row["Overlap@10"]) == pytest.approx(expected)
        assert float(row["Spearman@10"]) == pytest.approx(expected)
        assert float(row["RBO@10"]) == pytest.approx(expected)
    metadata = {row["item"]: row["value"] for row in rows if row["table_type"] == "metadata"}
    assert metadata["manual_llava_image_mode"] == metadata["no_manual_llava_image_mode"] == mode
    for label in ("manual", "no_manual"):
        with (report_path.parent / f"test_runs_{label}.pkl").open("rb") as handle:
            assert pickle.load(handle)["metadata"]["llava_image_mode"] == mode
    # Disabling statistics must avoid gathers and files.
    helpers._save_test_runs_for_similarity.__globals__["ENABLE_RETRIEVAL_SIMILARITY_STATS"] = False
    dist.all_gather_object = lambda *args: pytest.fail("statistics disabled")
    helpers._save_test_runs_for_similarity(run, run, "unused/best.csv", model, data, search)


@pytest.mark.parametrize("manual", [False, True])
@pytest.mark.parametrize("eol,calculate", [
    ("prompteol", "separate"),
    ("disassembleeol_separate", "concat"),
    ("disassembleeol_separate_origin_text", "concat"),
])
def test_single_condition_reports_disagreement(tmp_path, monkeypatch, manual, eol, calculate):
    monkeypatch.syspath_prepend(str(ROOT / "src"))
    helpers = definitions("src/search_similarity.py", {
        "os": os, "pickle": pickle, "ENABLE_RETRIEVAL_SIMILARITY_STATS": True,
        "dist": SimpleNamespace(get_world_size=lambda: 1, get_rank=lambda: 0,
                                all_gather_object=lambda gathered, run: gathered.__setitem__(0, run)),
    }, ("RETRIEVAL_SIMILARITY_STATS_DIR_NAME", "RETRIEVAL_SIMILARITY_CONFIGURATIONS"))
    label = "manual" if manual else "no_manual"
    output = tmp_path / f"0_{label}_30_sum_llava_base" / "best.csv"
    dense = {"q": {"docs": {f"d{i}": float(4-i) for i in range(4)}}}
    sparse = {"q": {"docs": {f"d{i}": float(i) for i in range(4)}}}
    helpers._save_test_runs_for_similarity(dense, sparse, str(output),
        SimpleNamespace(eol_type=eol, calculate_type=calculate),
        SimpleNamespace(dataset_name="flickr", sparse_manual=manual, llava_image_mode="base"),
        SimpleNamespace(query_type="image", depth=1000))
    report, = tmp_path.rglob("dense_sparse_similarity_test.csv")
    with report.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    summary, = [row for row in rows if row["table_type"] == "summary"]
    assert summary["condition"] == label
    for depth in (10, 20, 50, 100):
        assert float(summary[f"Overlap@{depth}"]) == 1.
        assert float(summary[f"Spearman@{depth}"]) == pytest.approx(-1.)
        assert 0. <= float(summary[f"RBO@{depth}"]) <= 1.
    assert (report.parent / f"test_runs_{label}.pkl").is_file()
    other = "no_manual" if manual else "manual"
    assert not (report.parent / f"test_runs_{other}.pkl").exists()


def test_entry_point_wiring_and_all_result_branches():
    source = (ROOT / "src/search_similarity.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    calls = [node for node in ast.walk(main) if isinstance(node, ast.Call)]
    preflight = next(node for node in calls if isinstance(node.func, ast.Name) and node.func.id == "validate_checkpoint")
    setup = next(node for node in calls if isinstance(node.func, ast.Name) and node.func.id == "configure_search")
    device = next(node for node in calls if ast.get_source_segment(source, node.func) == "torch.device")
    wrapper = next(node for node in calls if isinstance(node.func, ast.Name) and node.func.id == "MLLMRetrievalModel")
    assert preflight.lineno < device.lineno < setup.lineno < wrapper.lineno
    lucene = [node for node in calls if isinstance(node.func, ast.Name) and node.func.id == "LuceneImpactSearcher"]
    assert len(lucene) == 2
    assert all(isinstance(node.args[0], ast.Call) and node.args[0].func.id == "lucene_index_dir" for node in lucene)
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    paths = [node for node in ast.walk(main) if isinstance(node, ast.BinOp) and
             isinstance(node.left, ast.Name) and node.left.id == "path_prefix"]
    assert len(paths) == 35
    assert all(isinstance(parents[node], ast.Call) and parents[node].func.id == "llava_result_dir" for node in paths)


BASH = shutil.which("bash") if os.name != "nt" else next(
    (path for path in ("D:/Git/bin/bash.exe", "C:/Program Files/Git/bin/bash.exe") if Path(path).is_file()), None)


@pytest.fixture
def launch_shell(tmp_path):
    if BASH is None:
        pytest.skip("Bash not installed")
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    script = scripts / "search_similarity.sh"
    (tmp_path / "logs").mkdir()
    (tmp_path / "bin").mkdir()
    mock = tmp_path / "bin" / "deepspeed"
    with mock.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write('#!/usr/bin/env bash\n'
                     'printf "%s\\0" "$@" >> logs/args\n'
                     'printf "%s" "$ENABLE_RETRIEVAL_SIMILARITY_STATS" > logs/stats\n'
                     'printf "%s" "$OPENAI_API_KEY" > logs/key\n')
    mock.chmod(0o755)
    bin_path = (tmp_path / "bin").as_posix()
    if os.name == "nt":
        bin_path = "/" + bin_path[0].lower() + bin_path[2:]

    def launch(target="text", manual=False, eol="prompteol", calculate="separate", stats=True, argument=None, key=None):
        text = (ROOT / "scripts/search_similarity.sh").read_text(encoding="utf-8")
        for before, after in (
            ("ENCODE_TYPE=text", f"ENCODE_TYPE={target}"),
            ("SPARSE_MANUAL=false", f"SPARSE_MANUAL={str(manual).lower()}"),
            ("EOL_TYPE=prompteol", f"EOL_TYPE={eol}"),
            ("CALCULATE_TYPE=separate", f"CALCULATE_TYPE={calculate}"),
            ("ENABLE_RETRIEVAL_SIMILARITY_STATS=true", f"ENABLE_RETRIEVAL_SIMILARITY_STATS={str(stats).lower()}"),
        ):
            text = text.replace(before, after, 1)
        with script.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        env = dict(os.environ, MSYS_NO_PATHCONV="1")
        env.pop("OPENAI_API_KEY", None)
        if key:
            env["OPENAI_API_KEY"] = key
        command = 'export PATH=' + shlex.quote(bin_path) + ':/usr/bin:/bin:"$PATH"; exec /usr/bin/bash scripts/search_similarity.sh'
        if argument is not None:
            command += " " + shlex.quote(argument)
        return subprocess.run([BASH, "-c", command], cwd=tmp_path, env=env,
                              capture_output=True, text=True, encoding="utf-8", timeout=30)

    return launch


@pytest.mark.parametrize("manual", [False, True])
@pytest.mark.parametrize("target,query", [("text", "image"), ("image", "text")])
@pytest.mark.parametrize("eol,calculate", [
    ("prompteol", "separate"),
    ("disassembleeol_separate", "concat"),
    ("disassembleeol_separate_origin_text", "concat"),
])
def test_shell_directions_and_manual_settings(tmp_path, launch_shell, target, query, manual, eol, calculate):
    result = launch_shell(target, manual, eol, calculate)
    assert result.returncode == 0, result.stdout + result.stderr
    args = (tmp_path / "logs" / "args").read_bytes().decode().rstrip("\0").split("\0")
    value = lambda flag: args[args.index(flag) + 1]
    assert args.count("src/search_similarity.py") == 1
    assert value("--llava_image_mode") == "base" and "--fp16" in args
    assert value("--model_name_or_path") == "/root/gpufree-share/checkpoints/llava-hf-llava-v1.6-mistral-7b-hf"
    assert value("--sparse_manual") == str(manual).lower()
    for flag, root in (("--passage_reps", "dense_output"), ("--sparse_index", "sparse_output")):
        path = value(flag)
        assert path.startswith(f"/root/gpufree-share/{root}/")
        assert f"/{target}/no_filter/{calculate}/prompt_5/test/" in path
        assert f"after_pad_{eol}_lower_no_mean" in path
        assert "_llava_base" in path if target == "image" else "_llava_base" not in path
        assert "_manual_30_30_" in path if manual else "_no_manual_30_" in path
        assert path.endswith("/index") == (flag == "--sparse_index")
    assert value("--query_type") == query and value("--encode_type") == target
    assert value("--dataset_name") == "flickr" and value("--dataset_split") == "test"
    assert value("--task_type") == "ir" and value("--prompt_type") == "prompt_5"
    assert value("--eol_type") == eol and value("--calculate_type") == calculate
    assert value("--depth") == "1000" and value("--alpha") == "0.7" and value("--beta") == "0.3"
    assert "--use_fixed_alpha" in args
    for flag in ("--sparse_length", "--text_sparse_length", "--image_sparse_length"):
        assert value(flag) == "30"
    for flag in ("--use_freeret_dense", "--use_freeret_full_prompt", "--use_freeret_prompt_constraints"):
        assert value(flag) == "false"
    assert (tmp_path / "logs" / "stats").read_text() == "true"
    assert (tmp_path / "logs" / "key").read_text() == "local-placeholder"


def test_shell_can_disable_stats_and_preserves_existing_key(tmp_path, launch_shell):
    result = launch_shell(stats=False, key="local-test-value")
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "logs" / "stats").read_text() == "false"
    assert (tmp_path / "logs" / "key").read_text() == "local-test-value"


@pytest.mark.parametrize("argument", ["both", "no_manual", "manual"])
def test_shell_rejects_old_position_arguments(tmp_path, launch_shell, argument):
    result = launch_shell(argument=argument)
    assert result.returncode == 2 and "SPARSE_MANUAL" in result.stderr
    assert not (tmp_path / "logs" / "args").exists()
