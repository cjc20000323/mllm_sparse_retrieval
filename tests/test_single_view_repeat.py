"""CPU checks for the selected-count single-view sparse -> hybrid control."""
import argparse
import ast
import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import string
import subprocess
import sys
from contextlib import nullcontext
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from test_adaptive_rgp import BASH, ROOT, batch, digest, load, pipeline


@pytest.fixture(scope="module", autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def repeat(monkeypatch):
    spec = importlib.util.spec_from_file_location("single_view_repeat", ROOT / "src/single_view_repeat.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setitem(sys.modules, "single_view_repeat", module)
    return module


def selection(k):
    selected = [dict(label=str(i), aspect=f"aspect-{i}") for i in range(k)]
    return dict(selected=selected, selected_count=k, pool=list(range(10)),
                fingerprint=digest(selected), views_fingerprint=digest(dict(views=selected)),
                selection_modality="both")


def data_args(**changes):
    values = dict(dataset_name="flickr", dataset_split="test", sparse_value_mean=False,
                  sparse_logits_multiply_view_count=False, sparse_manual=False,
                  sparse_length=2, text_sparse_length=2, image_sparse_length=2,
                  is_filtered=False, sparse_lower_or_upper="lower", num_expended_tokens=0,
                  sparse_value_type="sum", print_sparse=False, reps_loc="after_pad", tbpr_type="origin_type", cir_type="type")
    return NS(**dict(values, **changes))


def model_args(enabled=True, **changes):
    values = dict(single_view_repeat_selected_count=enabled, eol_type="disassembleeol_separate",
                  calculate_type="concat", model_name_or_path="llava-hf-llava-v1.6-mistral-7b-hf")
    return NS(**dict(values, **changes))


@pytest.mark.parametrize("k", [1, 3, 5, 7])
def test_count_and_namespace_come_from_locked_selection(repeat, k):
    locked = selection(k)
    config = repeat.configuration(model_args(), data_args(), "ir", locked, family="llava_mistral")
    assert config["count"] == k and config["count_source"] == "selected"
    assert config["selection_fingerprint"] == locked["fingerprint"]
    path = Path("views_llava_base_dense_freeret_pre_mlp")
    marked = repeat.mark_path(path, config)
    assert f"single_view_repeat_k{k}_{locked['fingerprint'][:12]}" in marked.name
    assert marked.name.endswith("_llava_base_dense_freeret_pre_mlp")
    assert repeat.mark_path(marked, config) == marked
    assert repeat.configuration(model_args(False), data_args(), "ir", locked) is None
    assert repeat.mark_path(path, None) == path


@pytest.mark.parametrize("change", [dict(selected=[]), dict(selected_count=4), dict(selected_count=True),
                                   dict(fingerprint=""), dict(overlap_configuration={})])
def test_bad_or_inconsistent_counts_are_rejected(repeat, change):
    with pytest.raises(ValueError):
        repeat.selection_configuration(dict(selection(3), **change))


@pytest.mark.parametrize("model,data,task,family", [
    (model_args(), data_args(), "ir", "qwen2_5_vl"),
    (model_args(), data_args(dataset_name="RSTPReid"), "tbpr", "llava_mistral"),
    (model_args(), data_args(sparse_value_mean=True), "ir", "llava_mistral"),
    (model_args(), data_args(sparse_logits_multiply_view_count=True), "ir", "llava_mistral"),
    (model_args(use_stc_sparse=True), data_args(), "ir", "llava_mistral"),
    (model_args(use_lens_sparse=True), data_args(), "ir", "llava_mistral"),
    (model_args(use_freeret_full_prompt=True), data_args(), "ir", "llava_mistral"),
    (model_args(eol_type="disassembleeol_separate_origin_text"), data_args(), "ir", "llava_mistral"),
])
def test_unsupported_combinations_fail(repeat, model, data, task, family):
    with pytest.raises(ValueError):
        repeat.configuration(model, data, task, selection(3), family=family)


@pytest.fixture
def legacy(monkeypatch, repeat):
    api = load("encode.py", {"get_sparse_logit_multiplier", "validate_sparse_logits_multiply_view_count",
                            "get_img_valid_tokens_values", "get_text_valid_tokens_values"}, {
        "np": np, "torch": torch, "dist": NS(get_rank=lambda: 0), "word_tokenize": str.split,
        "stopwords": NS(words=lambda _: []), "string": string,
        "SPARSE_LOGIT_X5_MODELS": ("llava-hf-llama3-llava-next-8b-hf", "llava-hf-llava-v1.6-mistral-7b-hf"),
    })
    monkeypatch.setitem(sys.modules, "encode", api)
    return api


class Tokenizer:
    def get_vocab(self):
        return {"alpha": 0, "beta": 1, "other": 2, "<s>": 3, "</s>": 4,
                "<|begin_of_text|>": 3, "<|end_of_text|>": 4}

    def encode(self, word, **kwargs):
        return [self.get_vocab()[word]]


@pytest.mark.parametrize("k", [1, 3, 5, 7])
def test_actual_prompteol_quantization_and_legacy_x5_match(repeat, legacy, k):
    encoder = NS(model_args=model_args(), data=data_args(), selection=selection(k),
                 processor=NS(tokenizer=Tokenizer()))
    row = torch.tensor([[0.1234, 0.5555, 0.3001]])
    vectors = repeat.sparse_vectors(encoder, [{"text": "alpha beta"}, {"image": "image.jpg"}], [row, row])
    assert vectors == [
        {"beta": int(np.rint(float(row[0, 1])*k*100)), "alpha": int(np.rint(float(row[0, 0])*k*100))},
        {"beta": int(np.rint(float(row[0, 1])*k*100)), "other": int(np.rint(float(row[0, 2])*k*100))},
    ]
    if k in {1, 5}:
        old_data = data_args(sparse_logits_multiply_view_count=k == 5)
        old_model = model_args(False, eol_type="prompteol", calculate_type="separate")
        vocab = {v: key for key, v in Tokenizer().get_vocab().items()}
        tokens, weights = legacy.get_text_valid_tokens_values(
            "alpha beta", Tokenizer(), row[0], vocab, old_data, None, old_model)
        assert vectors[0] == dict(zip(tokens, weights))
    assert not hasattr(encoder.data, "selected_view_count")


def test_saved_count_and_source_cannot_be_changed(repeat):
    locked = selection(3)
    config = dict(dataset="flickr", task="ir", model={"family": "llava_mistral"},
                  single_view_repeat=repeat.selection_configuration(locked))
    repeat.validate_saved(config, locked)
    for key, value in (("count", 5), ("selection_fingerprint", "different")):
        broken = copy.deepcopy(config)
        broken["single_view_repeat"][key] = value
        with pytest.raises(ValueError):
            repeat.validate_saved(broken, locked)


@pytest.fixture
def encoder_api(monkeypatch, repeat):
    off = NS(enabled=lambda _: False, validate=lambda *a, **k: None)
    constraints = NS(enabled=lambda _: False, augment=lambda block, *a: block,
                     augment_context=lambda context, *a: context)
    monkeypatch.setitem(sys.modules, "freeret_full_prompt", off)
    monkeypatch.setitem(sys.modules, "freeret_prompt_constraints", constraints)
    monkeypatch.setitem(sys.modules, "dense_representation", NS(configure_model=lambda *a: None, enabled=lambda _: False))
    monkeypatch.setitem(sys.modules, "qwen_vl_encoding", NS(
        ViewBatch=lambda inputs, spans, prompts: NS(inputs=inputs, spans=spans, prompts=prompts),
        read_image=lambda image: image, forward_views=lambda *a, **k: pytest.fail("unexpected Qwen forward"),
        prepare_views=lambda *a, **k: pytest.fail("unexpected Qwen preparation")))
    template_spec = importlib.util.spec_from_file_location("template", ROOT / "src/template.py")
    template = importlib.util.module_from_spec(template_spec)
    template_spec.loader.exec_module(template)
    monkeypatch.setitem(sys.modules, "template", template)
    api = load("adaptive_view_encoding.py", {"context_and_whole", "instructions", "validate_sparse_print"}, {
        "copy": copy, "torch": torch, "F": F, "full_prompt": off, "prompt_constraints": constraints,
        "stc": off, "lens": off, "configure_processor": lambda *a: None,
        "print_sparse_views": lambda *a: None,
        "validate_encoding_method": lambda *a, **k: None,
        "views": NS(family_of=lambda config: config.family, view_pool=lambda _: list(range(10)), pool_mode=lambda _: None),
    })
    namespace = vars(api)
    node = next(node for node in ast.parse((ROOT / "src/adaptive_view_encoding.py").read_text(encoding="utf-8")).body
                if isinstance(node, ast.ClassDef) and node.name == "AdaptiveEncoder")
    exec(compile(ast.Module(body=[node], type_ignores=[]), "AdaptiveEncoder", "exec"), namespace)
    return NS(**namespace)


@pytest.mark.parametrize("k", [1, 3, 5, 7])
@pytest.mark.parametrize("image", [False, True])
@pytest.mark.parametrize("family", ["llava_mistral", "llava_llama"])
def test_encoder_uses_one_view_one_forward_and_preserves_dense(encoder_api, k, image, family):
    marker = "<s>" if family == "llava_mistral" else "<|begin_of_text|>"
    class Processor:
        tokenizer = Tokenizer()

        def __call__(self, text, **kwargs):
            rows = [[3, 0, 4]*prompt.count(marker) for prompt in text]
            return dict(input_ids=torch.tensor(rows), attention_mask=torch.ones(len(rows), len(rows[0]), dtype=torch.long))

    class Model:
        config = NS(family=family)
        calls = 0

        def get_input_embeddings(self):
            return NS(weight=torch.zeros(1))

        def __call__(self, **inputs):
            self.calls += 1
            batch, length = inputs["input_ids"].shape
            return NS(logits=torch.tensor([-1., 0.3, 0.8]).repeat(batch, length, 1),
                      hidden_states=[torch.tensor([3., 4.]).repeat(batch, length, 1)])

    record = {"id": "one", **({"image": "image.jpg"} if image else {"text": "alpha beta"})}
    outputs = []
    for enabled in (False, True):
        model = Model()
        encoder = encoder_api.AdaptiveEncoder(model, Processor(), model_args(enabled), data_args(), "ir", selection(k))
        sparse, dense = encoder.encode([record])
        assert model.calls == 1
        assert sparse[0].shape == ((1 if enabled else k), 3)
        assert encoder.last_prompts[0].count(marker) == (1 if enabled else k+1)
        if enabled:
            assert "aspect-" not in encoder.last_prompts[0]
        outputs.append(dense)
    torch.testing.assert_close(outputs[0], outputs[1])
    torch.testing.assert_close(outputs[1], torch.tensor([[0.6, 0.8]]))


def test_batch_namespaces_forward_mode_and_isolate_output(batch, repeat, monkeypatch):
    settings = batch.settings()
    settings.stop_after_stage2 = True
    old, _ = batch.api.prepare_jobs(settings, backend=batch.backend)
    locked = selection(3)
    locked.update(selection_split="test", sampling={"rounds": 20, "seed": 42})
    selection_path = Path(old[0].variables["VIEW_SELECTION_FILE"])
    selection_path.write_text(json.dumps(locked), encoding="utf-8")
    metadata = batch.backend.metadata

    def repeated_metadata(model, data, task, kind, identity, selection):
        result = metadata(model, data, task, kind, identity, selection)
        result["configuration"][repeat.FIELD] = repeat.configuration(model, data, task, selection, family=identity["family"])
        result["fingerprint"] = digest(result["configuration"])
        return result

    monkeypatch.setattr(batch.backend, "metadata", repeated_metadata)
    monkeypatch.setitem(batch.namespace, "validate_corpus", lambda *a, **k: {"num_shards": 1})
    settings.controls[repeat.ENV] = "true"
    jobs, missing = batch.api.prepare_jobs(settings, backend=batch.backend)
    assert not missing
    job, = jobs
    assert job.variables[repeat.ENV] == "true"
    assert job.configuration[repeat.FIELD]["count"] == 3
    assert job.output != old[0].output and "single_view_repeat_k3" in job.output.name
    assert job.configuration["execution_mode"] == "two_stage"


def test_query_cache_keys_separate_mode_and_count(repeat):
    api = load("adaptive_view_encoding.py", {"_query_encoding_settings"}, {
        "probability_mass": NS(validate_requested=lambda *a: None, is_selection=lambda _: False)})
    pipeline = NS(_qwen_encoding_settings=lambda *a: dict(existing=True))
    kwargs = dict(eol_type="disassembleeol_separate", calculate_type="concat", sparse_manual=False)
    base = api._query_encoding_settings(NS(), pipeline, selection(3), **kwargs)
    assert base == {"existing": True}
    one = api._query_encoding_settings(model_args(), pipeline, selection(1), **kwargs)
    three = api._query_encoding_settings(model_args(), pipeline, selection(3), **kwargs)
    assert len({digest(base), digest(one), digest(three)}) == 3


@pytest.mark.parametrize("kind", ["image", "text"])
def test_encoding_manifest_and_corpus_paths_separate_modes(encoder_api, repeat, monkeypatch, kind):
    optional = NS(enabled=lambda _: False, configuration=lambda _: None, mark_path=lambda path, config: path)
    dense = NS(enabled=lambda _: False, mode_path=lambda path, *a: path, original_path=lambda path: path,
               configuration=lambda config, *a: config)
    monkeypatch.setitem(sys.modules, "dense_representation", dense)
    monkeypatch.setitem(sys.modules, "qwen_vl_pipeline", NS(artifact_dirs=lambda *a: [Path("dense"), Path("sparse")]))
    monkeypatch.setitem(sys.modules, "llava_resolution", NS(
        artifact_dir=lambda path, mode: path.with_name(path.name + "_llava_base") if mode == "base" else path))
    views = NS(encoding_version=lambda _: 18, digest=digest, is_pmi_selection=lambda _: False,
               pool_mode=lambda _: None, add_pool_mode=lambda config, mode: config,
               examples=NS(FIELD="example_sampling", add_configuration=lambda config, policy: config))
    api = load("adaptive_view_encoding.py", {"artifact_dirs", "encoding_configuration", "metadata"}, {
        "views": views, "stc": optional, "lens": optional,
        "prompt_constraints": NS(validate_selection=lambda *a: None, enabled=lambda _: False),
        "full_prompt": NS(enabled=lambda _: False),
        "option_shuffle": NS(FIELD="option_shuffle", mark_path=lambda path, config: path,
                             add_metadata=lambda config, policy: config),
        "probability_mass": NS(add_metadata=lambda config, selection: config),
    })
    data, locked = data_args(llava_image_mode="base"), selection(3)
    identity = dict(family="llava_mistral")
    old = api.metadata(model_args(False), data, "ir", kind, identity, locked)
    absent = model_args(False)
    del absent.single_view_repeat_selected_count
    assert api.metadata(absent, data, "ir", kind, identity, locked) == old
    new = api.metadata(model_args(), data, "ir", kind, identity, locked)
    assert new["fingerprint"] != old["fingerprint"]
    assert new["configuration"].pop(repeat.FIELD)["count"] == 3
    assert new["configuration"] == old["configuration"]
    original = api.artifact_dirs(model_args(False), data, NS(task_type="ir"), locked)
    repeated = api.artifact_dirs(model_args(), data, NS(task_type="ir"), locked)
    assert set(original).isdisjoint(repeated)
    assert all("_single_view_repeat_k3_" in path.name and path.name.endswith("_llava_base") for path in repeated)


def test_index_checks_locked_count_and_marked_directory(repeat, monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "adaptive_views", NS(
        resolve_token_variants=lambda *a, **k: False, resolve_prefix_tokens=lambda *a, **k: False,
        view_pool=lambda _: list(range(10)), pool_mode=lambda _: None))
    monkeypatch.setitem(sys.modules, "adaptive_view_shuffle", NS(
        FIELD="option_shuffle", validate=lambda *a, **k: None, mark_path=lambda path, config: path))
    monkeypatch.setitem(sys.modules, "dense_representation", NS(
        SUFFIX="_dense_freeret_pre_mlp", PROMPT_SUFFIX="_freeret_constraints_v1", FULL_PROMPT_SUFFIX="_freeret_full_prompt_v1",
        validate_artifact=lambda *a: None))
    for name in ("stc_sparse", "lens_sparse"):
        monkeypatch.setitem(sys.modules, name, NS(FIELD=name, mark_path=lambda path, config: path,
                                                validate_bundle=lambda *a: None))
    api = load("adaptive_sparse_index_batch.py", {"check_artifact"}, {
        "Path": Path, "re": __import__("re"), "digest": digest, "MANIFEST": "adaptive_views_manifest.json",
        "PREFIX_VERSION": 24, "VARIANT_VERSION": 23,
        "collection_plan": lambda *a: None, "canonical_index": lambda root: root / "collection/index",
    })
    locked = selection(3)
    config = dict(version=18, model=dict(family="llava_mistral"), dataset="flickr", task="ir",
                  views_fingerprint=locked["views_fingerprint"], single_view_repeat=repeat.selection_configuration(locked))
    root = tmp_path / locked["views_fingerprint"][:16]
    marked = repeat.mark_path(root, config[repeat.FIELD])
    manifest = dict(configuration=config, fingerprint=digest(config))
    assert api.check_artifact(marked, manifest, config, selection=locked) == marked / "collection/index"
    with pytest.raises(ValueError, match="Directory/view fingerprint"):
        api.check_artifact(root, manifest, config, selection=locked)
    inconsistent = copy.deepcopy(config)
    inconsistent[repeat.FIELD]["count"] = 5
    with pytest.raises(ValueError, match="locked selection/count"):
        api.check_artifact(marked, dict(fingerprint=digest(inconsistent)), inconsistent, selection=locked)


@pytest.mark.parametrize("stop,method", [(False, "hybrid"), (True, "dense")])
def test_batch_rejects_other_pipelines_before_launch(batch, repeat, stop, method):
    settings = batch.settings()
    settings.stop_after_stage2 = stop
    settings.controls.update(SINGLE_VIEW_REPEAT_SELECTED_COUNT="true", STAGE2_METHOD=method)
    with pytest.raises(ValueError, match="two-stage sparse -> hybrid"):
        batch.api.prepare_jobs(settings, backend=batch.backend)


def test_pipeline_switch_requires_two_stage_hybrid(pipeline, repeat):
    args = pipeline.arguments(extra=["--single-view-repeat-selected-count"])
    with pytest.raises(ValueError, match="two-stage"):
        pipeline.api._validate_args(args)
    args.stop_after_stage2 = True
    args.stage2_method = "dense"
    with pytest.raises(ValueError, match="sparse -> hybrid"):
        pipeline.api._validate_args(args)


@pytest.mark.parametrize("name", ["sparse_value_mean", "sparse_logits_multiply_view_count", "use_stc_sparse", "use_lens_sparse"])
def test_pipeline_rejects_incompatible_modes_before_loading(pipeline, repeat, name):
    args = pipeline.arguments(extra=["--single-view-repeat-selected-count", "--stop-after-stage2"])
    setattr(args, name, True)
    with pytest.raises(ValueError, match="incompatible"):
        pipeline.api._validate_args(args)


@pytest.mark.parametrize("name", ["view_overlap_selection", "use_stc_sparse", "use_lens_sparse"])
def test_encoder_rejects_incompatible_modes_before_loading(repeat, monkeypatch, name):
    monkeypatch.setitem(sys.modules, "qwen_vl_pipeline", NS(init_process=lambda: pytest.fail("distributed initialization")))
    api = load("adaptive_view_encoding.py", {"run_encode"}, {})
    with pytest.raises(ValueError, match="incompatible"):
        api.run_encode(model_args(), data_args(), NS(task_type="ir"), NS(**{name: True}))


def test_disabled_batch_switch_preserves_existing_fingerprint(batch):
    settings = batch.settings()
    settings.stop_after_stage2 = True
    original, _ = batch.api.prepare_jobs(settings, backend=batch.backend)
    settings.controls["SINGLE_VIEW_REPEAT_SELECTED_COUNT"] = "false"
    explicit_off, _ = batch.api.prepare_jobs(settings, backend=batch.backend)
    assert original[0].fingerprint == explicit_off[0].fingerprint
    assert original[0].output == explicit_off[0].output


def test_encoding_controls_validate_mode_before_launch(monkeypatch, repeat):
    for name in ("stc_sparse", "lens_sparse"):
        monkeypatch.setitem(sys.modules, name, NS(environment_controls=lambda env: {}))
    api = load("adaptive_encode_batch.py", {"boolean", "encoding_controls"}, {"math": math}, ("DEFAULTS",))
    assert api.encoding_controls({})[repeat.ENV] == "false"
    assert api.encoding_controls({repeat.ENV: "yes"})[repeat.ENV] == "true"
    for name in ("SPARSE_VALUE_MEAN", "USE_STC_SPARSE", "USE_LENS_SPARSE", "USE_FREERET_FULL_PROMPT", "USE_VIEW_OVERLAP_SELECTION"):
        with pytest.raises(ValueError, match="incompatible"):
            api.encoding_controls({repeat.ENV: "true", name: "true"})


def test_single_view_result_markers_are_idempotent_with_probability_mass(monkeypatch, repeat):
    monkeypatch.syspath_prepend(str(ROOT / "src"))
    from stage2_retrieval_mode import adaptive_output_path
    locked = selection(3)
    locked.update(version=18, probability_mass_configuration={"version": 2, "threshold": 0.9})
    # Use the real output helper while supplying only its light configuration modules.
    monkeypatch.setitem(sys.modules, "adaptive_view_probability_mass", NS(is_selection=lambda _: True))
    monkeypatch.setitem(sys.modules, "adaptive_view_shuffle", NS(FIELD="option_shuffle"))
    config = repeat.selection_configuration(locked)
    first = adaptive_output_path(repeat.mark_path(Path("results"), config), locked)
    assert first.name.count("_pm_") == 1
    assert adaptive_output_path(repeat.mark_path(first, config), locked) == first


def test_stage2_reuses_encoded_query_and_filters_outside_candidates(repeat):
    seen = []
    def dense(*a, **kwargs):
        assert kwargs["candidate_filter"] == {"q": {"a", "b"}}
        seen.append("dense")
        return {"q": {"a": 0.1, "b": 0.3, "outside": 100.}}
    def sparse(*a, **kwargs):
        seen.append("sparse")
        return {"q": {"a": 40., "b": 10., "outside": 100.}}
    api = load("three_stage_retrieval_pipeline_adaptive_views.py", {
        "run_stage2", "fuse_restricted_runs", "_top_ids", "_minmax", "retrieval_eol_type"}, {
        "Path": Path, "stage2_method": lambda _: "hybrid",
        "_validate_encoded_queries": lambda encoded, qids: None,
        "_validate_qwen_artifacts": lambda *a, **k: None,
        "stage2_artifact_paths": lambda args: [],
        "encode_queries_online": lambda *a, **k: pytest.fail("second query forward"),
        "retrieve_dense_from_corpus": dense, "retrieve_sparse_topics_from_lucene": sparse,
        "_compute_scope": lambda *a, **k: nullcontext(), "_progress": lambda rows, **k: rows,
    }, ("STAGE2_EOL_TYPE", "STAGE2_CALCULATE_TYPE"))
    args = NS(single_view_repeat_selected_count=True, stage1_method="sparse_concat", stage2_run=None,
              stage1_depth=2, stage2_depth=2, stage2_dense_index="dense", stage2_sparse_index="sparse",
              no_dense_normalize=False, sparse_threads=1, hybrid_alpha=0.7)
    encoded = NS(query_ids=["q"], dense_representations=np.array([[1., 0.]]), sparse_topics={"q": "beta beta"})
    result = api.run_stage2(args, {"q": {"a": 3., "b": 2., "c": 1.}}, encoded_queries=encoded)
    assert seen == ["dense", "sparse"]
    assert set(result[2]["q"]) == {"a", "b"}


@pytest.mark.parametrize("stage", ["encode", "index", "retrieve"])
def test_shell_forwards_repeat_switch(tmp_path, stage):
    if BASH is None:
        pytest.skip("Bash not installed")
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    entries = {"encode": "encode_adaptive_views_ir.sh", "index": "sparse_index_adaptive_ir.sh",
               "retrieve": "two_stage_retrieval_pipeline_adaptive_views_ir.sh"}
    for name in [entries[stage], "stc_sparse_mode.sh", "lens_sparse_mode.sh", "freeret_dense_mode.sh", "stage2_retrieval_mode.sh"]:
        shutil.copy2(ROOT / "scripts" / name, scripts / name)
    runner = tmp_path / "runner.sh"
    runner.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$SINGLE_VIEW_REPEAT_SELECTED_COUNT" "$@" > "$CAPTURE"\n', encoding="utf-8")
    capture = tmp_path / "args.txt"
    env = dict(os.environ, SINGLE_VIEW_REPEAT_SELECTED_COUNT="true", MODEL_FAMILY="llava_mistral",
               MODEL_PATH="models/llava-hf-llava-v1.6-mistral-7b-hf", DATASET="flickr",
               ENCODE_TYPE="image", OUTPUT_DIR="results", VIEW_SELECTION_FILE="selection.json" if stage != "index" else "",
               DENSE_CORPUS_DIR="dense", SPARSE_CORPUS_DIR="sparse", STAGE1_SPARSE_INDEX="sparse/index",
               DEEPSPEED_BIN=runner.as_posix(), PYTHON_BIN=runner.as_posix(), CAPTURE=capture.as_posix(),
               USE_FREERET_DENSE="false", USE_FREERET_FULL_PROMPT="false", USE_FREERET_PROMPT_CONSTRAINTS="false",
               USE_STC_SPARSE="false", USE_LENS_SPARSE="false", USE_RERANK_PRIOR_CACHE="false", DRY_RUN="false")
    command = [BASH, "scripts/" + entries[stage]] + ([] if stage == "index" else ["--run-one"])
    result = subprocess.run(command, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    argv = capture.read_text().splitlines()
    assert argv[0] == "true"
    if stage == "encode":
        assert argv[argv.index("--single_view_repeat_selected_count")+1] == "true"
    elif stage == "retrieve":
        assert "--single-view-repeat-selected-count" in argv and "--stop-after-stage2" in argv

