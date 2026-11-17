"""PromptEOL diagnostic checks; native batch loops run with tiny CPU fixtures."""
import ast
import contextlib
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import pickle
import re
import shlex
import shutil
import string
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest

from test_adaptive_sparse_print import BASH, ROOT, definitions, environment


SCRIPT = "encode_llava_base_prompteol_print_sparse.sh"
sys.path.insert(0, str(ROOT / "src"))
import prompteol_sparse_diagnostics as diagnostics


def settings(**changes):
    model = NS(eol_type="prompteol", calculate_type="separate", use_output_embedding_cluster=False,
               model_name_or_path="llava-hf-llava-v1.6-mistral-7b-hf")
    data = NS(llava_image_mode="base", dataset_name="flickr", print_sparse=True, sparse_type="single")
    training = NS(task_type="ir", encode_type="text", fp16=False)
    options = diagnostics.PrompteolSparsePrintArguments(print_sparse_diagnostics=True)
    for path, value in changes.items():
        name, field = path.split(".")
        setattr(dict(model=model, data=data, training=training, options=options)[name], field, value)
    return model, data, training, options


@pytest.mark.parametrize("changes,pattern", [
    ({"model.eol_type": "disassembleeol_separate"}, "prompteol"),
    ({"model.calculate_type": "concat"}, "separate"),
    ({"data.llava_image_mode": "anyres"}, "base"),
    ({"training.task_type": "tbpr"}, "IR text/image"),
    ({"training.encode_type": "corpus"}, "IR text/image"),
    ({"data.dataset_name": "webqa"}, "Flickr and COCO"),
    ({"data.sparse_type": "fusion"}, "unclustered single"),
    ({"model.use_output_embedding_cluster": True}, "unclustered single"),
    ({"model.use_freeret_full_prompt": True}, "ordinary prompt"),
    ({"options.print_sparse_top_k": 0}, "positive integer"),
    ({"options.print_sparse_format": "invalid"}, "text or json"),
    ({"options.print_sparse_timing": "invalid"}, "immediate or end"),
])
def test_invalid_diagnostics_fail_before_encoding(changes, pattern):
    with pytest.raises(ValueError, match=pattern):
        diagnostics.validate_configuration(*settings(**changes))


def test_inactive_diagnostics_keep_legacy_objects_and_optional_imports(monkeypatch):
    model, data, training, options = settings()
    options.print_sparse_diagnostics = False
    options.print_sparse_top_k = 0
    options.print_sparse_format = "invalid"
    options.diagnostic_data_root = "unused"
    monkeypatch.setitem(sys.modules, "adaptive_view_encoding", None)
    monkeypatch.setitem(sys.modules, "dataset", None)
    diagnostics.validate_configuration(model, data, training, options)
    diagnostics.configure_data_root(options)
    assert diagnostics.quiet_legacy_printing(data, options) is data and data.print_sparse
    assert diagnostics.build_printer(None, model, data, training, options, None) is None
    options.print_sparse_diagnostics = True
    data.print_sparse = False
    assert diagnostics.quiet_legacy_printing(data, options) is data
    assert diagnostics.build_printer(None, model, data, training, options, None) is None


def test_data_root_is_explicit_and_diagnostic_only(monkeypatch, tmp_path):
    module = NS(dataset_path_prefix="old", flickr_file_path="old-flickr", coco_file_path="old-coco",
                webqa_file_path="unchanged")
    monkeypatch.setitem(sys.modules, "dataset", module)
    options = diagnostics.PrompteolSparsePrintArguments(diagnostic_data_root=str(tmp_path))
    diagnostics.configure_data_root(options)
    assert module.dataset_path_prefix == "old"
    options.print_sparse_diagnostics = True
    diagnostics.configure_data_root(options)
    assert Path(module.flickr_file_path) == tmp_path / "data/flickr"
    assert Path(module.coco_file_path) == tmp_path / "data/coco"
    assert module.webqa_file_path == "unchanged"


def test_diagnostic_arguments_do_not_duplicate_native_parser_fields():
    source = ast.parse((ROOT / "src/arguments.py").read_text(encoding="utf-8"))
    native_names = {node.target.id for cls in source.body if isinstance(cls, ast.ClassDef)
                    and cls.name in {"ModelArguments", "PromptRepsLLMDataArguments", "TrainingArguments"}
                    for node in cls.body if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)}
    options = diagnostics.PrompteolSparsePrintArguments()
    assert not options.print_sparse_diagnostics
    assert options.print_sparse_timing == "immediate"
    assert not (set(options.__dataclass_fields__) & native_names)


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("enabled", [False, True])
def test_native_summary_follows_saved_manifests_and_precedes_teardown(rank, enabled, tmp_path):
    source = ast.parse((ROOT / "src/encode.py").read_text(encoding="utf-8"))
    main = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    barriers = [i for i, node in enumerate(main.body) if isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute)
                and isinstance(node.value.func.value, ast.Name) and node.value.func.value.id == "dist"
                and node.value.func.attr == "barrier"]
    tail = ast.parse("def finalize():\n    pass\n").body[0]
    tail.body = copy.deepcopy(main.body[barriers[-2]:])
    events = []
    namespace = dict(
        dist=NS(get_rank=lambda: rank, barrier=lambda: events.append("saved_barrier"),
                destroy_process_group=lambda: events.append("destroy")),
        llava_output_dirs=[tmp_path], llava_report={}, model_args=NS(),
        write_manifest=lambda *a: events.append("resolution_manifest"),
        dense_mode=NS(write_manifest=lambda *a: events.append("dense_manifest"), enabled=lambda *a: False),
        prompt_constraints=NS(enabled=lambda *a: False), full_prompt=NS(enabled=lambda *a: False),
        sparse_printer=NS(finish=lambda: events.append("summary")) if enabled else None)
    namespace["torch"] = NS(distributed=namespace["dist"])
    module = ast.fix_missing_locations(ast.Module(body=[tail], type_ignores=[]))
    exec(compile(module, "native_encoding_finalize", "exec"), namespace)
    namespace["finalize"]()
    expected = ["saved_barrier"]
    if rank == 0:
        expected += ["resolution_manifest", "dense_manifest"]
    expected += ["saved_barrier"] + (["summary"] if enabled else []) + ["destroy"]
    assert events == expected


@pytest.mark.parametrize("fail_saving", [False, True])
def test_adaptive_summary_only_after_successful_completion(fail_saving):
    source = ast.parse((ROOT / "src/adaptive_view_encoding.py").read_text(encoding="utf-8"))
    run = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == "run_encode")
    outer = next(node for node in run.body if isinstance(node, ast.Try) and node.finalbody
                 and any(isinstance(call, ast.Attribute) and call.attr == "_release_retrieval_runtime"
                         for statement in node.finalbody for call in ast.walk(statement)))
    finalizer = ast.parse("def finalize():\n    pass\n").body[0]
    finalizer.body = [ast.Try(body=copy.deepcopy(outer.body[-3:]), handlers=[], orelse=[],
                              finalbody=copy.deepcopy(outer.finalbody))]
    events = []
    def complete():
        events.append("save")
        if fail_saving:
            raise OSError("saving failed")
    namespace = dict(rank=0, complete=complete, encoder=object(), roots=("dense", "sparse"), selection={},
                     root_action=lambda rank, fn: fn(), finish_sparse_print=lambda encoder: events.append("summary"),
                     runtime=object(), api=NS(_release_retrieval_runtime=lambda *a: events.append("release")))
    module = ast.fix_missing_locations(ast.Module(body=[finalizer], type_ignores=[]))
    exec(compile(module, "adaptive_encoding_finalize", "exec"), namespace)
    if fail_saving:
        with pytest.raises(OSError, match="saving failed"):
            namespace["finalize"]()
        assert events == ["save", "release"]
    else:
        assert namespace["finalize"]() == (("dense", "sparse"), {})
        assert events == ["save", "summary", "release"]


def native_loop(namespace):
    """Execute the real ordinary-IR batch loop, including its diagnostic hook."""
    source = (ROOT / "src/encode.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    loops = [node for node in ast.walk(tree) if isinstance(node, ast.For)
             and any(isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute)
                     and child.func.attr == "record_batch" for child in ast.walk(node))]
    loop = min(loops, key=lambda node: node.end_lineno - node.lineno)
    body = ast.parse("def batch():\n    pass\n").body[0]
    body.body = [ast.parse("prompt = initial_prompt").body[0], copy.deepcopy(loop),
                 ast.parse("return encoded, lookup_indices, jsonl_data").body[0]]
    module = ast.fix_missing_locations(ast.Module(body=[body], type_ignores=[]))
    exec(compile(module, "native_prompteol_batch", "exec"), namespace)
    return namespace["batch"]


def check_native_batches(print_timing="immediate"):
    import numpy as np
    import torch
    from PIL import Image
    for family in ("llava_mistral", "llava_llama"):
        for kind in ("text", "image"):
            for manual in (False, True):
                probe = environment(family, initialized=False, size=40, unmapped=False)
                sys.modules["adaptive_view_encoding"] = probe.api
                probe.api.worker_errors = lambda error: (_ for _ in ()).throw(ValueError(error)) if error else None
                tokenizer = probe.encoder().processor.tokenizer
                tokenizer.encode = lambda word, **kwargs: [0, 1]
                probe.data.print_sparse = True
                probe.data.per_device_batch_size = 2
                probe.data.sparse_manual = manual
                probe.data.sparse_type = "single"
                probe.data.sparse_logits_multiply_view_count = False
                probe.data.prompt_type = "prompt_5"
                probe.model_args.eol_type = "prompteol"
                probe.model_args.calculate_type = "separate"
                probe.model_args.use_output_embedding_cluster = False
                probe.model_args.model_name_or_path = ("llava-hf-llava-v1.6-mistral-7b-hf" if family == "llava_mistral"
                                                       else "llava-hf-llama3-llava-next-8b-hf")
                weights = torch.arange(1, 81).reshape(2, 40).float() / 10
                reps = torch.arange(6).reshape(2, 3).float()
                calls = []

                class Processor:
                    def __init__(self):
                        self.tokenizer = tokenizer
                    def __call__(self, images, **kwargs):
                        assert all(image.mode == "RGB" for image in images)
                        return NS(to=lambda device: {"images": images})
                processor = Processor()

                class Model:
                    def encode_data(self, inputs, input_type, *args):
                        calls.append(input_type)
                        if input_type == "image":
                            assert "images" in inputs
                        return weights, reps

                extractors = definitions("encode.py", {"get_text_valid_tokens_values", "get_img_valid_tokens_values",
                    "filter_token", "get_sparse_logit_multiplier"}, dict(
                        torch=torch, np=np, string=string, word_tokenize=str.split,
                        stopwords=NS(words=lambda _: []), dist=NS(get_rank=lambda: 0)))
                with tempfile.TemporaryDirectory(prefix="prompteol-") as directory:
                    root = Path(directory)
                    images = []
                    for name in ("other.jpg", "411175971.jpg"):
                        path = root / name
                        Image.new("RGB", (2, 2)).save(path)
                        images.append(str(path))
                    texts = ["other text", "A brown dog is shown standing in the water near a muddy beach."]
                    text_ids, image_ids = ["other-text", "89110"], ["other-image", "17822"]
                    training = NS(task_type="ir", encode_type=kind, fp16=False)
                    options = diagnostics.PrompteolSparsePrintArguments(
                        print_sparse_diagnostics=True, print_sparse_format="text",
                        print_sparse_timing=print_timing,
                        print_sparse_sample_id="89110" if kind == "text" else "17822",
                        print_sparse_log_dir=str(root / "logs"))
                    dataset = NS(text_id_list=text_ids, img_id_list=image_ids)
                    namespace = dict(torch=torch, F=torch.nn.functional, np=np, Image=Image, nullcontext=contextlib.nullcontext,
                        model_args=probe.model_args, training_args=training, processor=processor,
                        dist=NS(get_rank=lambda: 0), model=Model(), device="cpu", initial_prompt="whole-input prompt",
                        test_dataloader=[(texts, images, text_ids, image_ids)], tqdm=lambda rows, **kwargs: rows,
                        vocab_dict={i: token for token, i in tokenizer.get_vocab().items()}, filtered_ids=set(),
                        get_text_valid_tokens_values=extractors.get_text_valid_tokens_values,
                        get_img_valid_tokens_values=extractors.get_img_valid_tokens_values)
                    execute = native_loop(namespace)
                    namespace.update(data_args=copy.copy(probe.data), sparse_printer=None,
                                     encoded=[], lookup_indices=[], jsonl_data=[])
                    namespace["data_args"].print_sparse = False
                    with contextlib.redirect_stdout(io.StringIO()):
                        original = execute()
                    assert calls == [kind]
                    original_bytes = pickle.dumps(original)
                    printers = []
                    for rank in (0, 1):
                        probe.namespace["dist"].is_initialized = lambda: True
                        probe.namespace["dist"].get_rank = lambda rank=rank: rank
                        printer = diagnostics.build_printer(processor, probe.model_args, probe.data,
                                                             training, options, dataset)
                        quiet = diagnostics.quiet_legacy_printing(probe.data, options)
                        assert quiet is not probe.data and not quiet.print_sparse and probe.data.print_sparse
                        assert {k: v for k, v in vars(quiet).items() if k != "print_sparse"} == {
                            k: v for k, v in vars(probe.data).items() if k != "print_sparse"}
                        namespace.update(data_args=quiet, sparse_printer=printer,
                                         encoded=[], lookup_indices=[], jsonl_data=[])
                        output = io.StringIO()
                        before = weights.clone()
                        with patch.object(torch.distributed, "is_initialized", return_value=True), \
                                patch.object(torch.distributed, "get_rank", return_value=rank), \
                                contextlib.redirect_stdout(output), printer:
                            measured = execute()
                        assert pickle.dumps(measured) == original_bytes and torch.equal(weights, before)
                        rows = [json.loads(line) for line in (root / "logs" / f"sparse_tokens_rank{rank}.jsonl").read_text(encoding="utf-8").splitlines()]
                        assert len(rows) == 2 and {row["view"] for row in rows} == {"whole"}
                        assert all(row["aspect"] == "prompteol" and row["rank"] == rank for row in rows)
                        assert {row["sample_id"] for row in rows} == set(text_ids if kind == "text" else image_ids)
                        for i, row in enumerate(rows):
                            if kind == "text" and not manual:
                                ids = torch.tensor([1, 0])
                                expected = weights[i, ids]
                                assert row["token_source"] == "retained_text" and row["retained_token_count"] == 2
                            else:
                                expected, ids = weights[i].topk(10)
                                assert "token_source" not in row
                            assert [t["token_id"] for t in row["top_tokens"]] == ids.tolist()
                            assert [t["weight"] for t in row["top_tokens"]] == expected.tolist()
                        console = output.getvalue()
                        if print_timing == "end":
                            assert "[prompteol sparse]" not in console
                        else:
                            count = 2 if kind == "text" and not manual else 10
                            assert console.count(f"Top-{count} tokens (log1p_relu)") == 1
                            assert "[prompteol sparse]" in console and "[adaptive sparse]" not in console
                            assert 'sample_id="other-' not in console
                            assert "411175971.jpg" in console if kind == "image" else texts[1] in console
                        assert printer.adapter.sparse_print_stream is None
                        printers.append(printer)
                    if print_timing == "end":
                        log_bytes = {p: p.read_bytes() for p in (root / "logs").glob("*.jsonl")}
                        pending = json.loads(json.dumps([p.adapter._sparse_print_pending for p in printers]))
                        role = dict(rank=1)
                        probe.namespace["dist"].get_rank = lambda: role["rank"]
                        probe.namespace["dist"].get_world_size = lambda: 2
                        def gather(result, local):
                            assert local == pending[role["rank"]]
                            result[:] = copy.deepcopy(pending)
                        probe.namespace["dist"].all_gather_object = gather
                        output = io.StringIO()
                        with contextlib.redirect_stdout(output):
                            printers[1].finish()
                        assert output.getvalue() == ""
                        role["rank"] = 0
                        output = io.StringIO()
                        with contextlib.redirect_stdout(output):
                            printers[0].finish()
                        assert "[prompteol sparse summary] encoding completed" in output.getvalue()
                        count = 2 if kind == "text" and not manual else 10
                        assert output.getvalue().count(f"Top-{count} tokens (log1p_relu)") == 2
                        assert "rank=0 " in output.getvalue() and "rank=1 " in output.getvalue()
                        assert all(p.read_bytes() == contents for p, contents in log_bytes.items())


def check_retained_text_candidates(family, timing):
    """Use the real extraction result, including its 128-ID limit and tie order."""
    import numpy as np
    import torch

    probe = environment(family, initialized=False, size=200, unmapped=False)
    sys.modules["adaptive_view_encoding"] = probe.api
    tokenizer = probe.encoder().processor.tokenizer
    vocabulary = tokenizer.get_vocab()
    marker = chr(0x0120) if family == "llava_llama" else chr(0x2581)
    for index, token in {7: marker + "Brown", 9: marker + "Dog", 12: "sho", 13: "wn",
                         15: "standing", 20: marker + "Water", 22: "mud", 23: "dy",
                         25: marker + "Beach"}.items():
        del vocabulary[next(name for name, value in vocabulary.items() if value == index)]
        vocabulary[token] = index
    id_vocabulary = {index: token for token, index in vocabulary.items()}
    word_ids = dict(brown=[7], dog=[9], shown=[12, 13], standing=[15], water=[20],
                    muddy=[22, 23], beach=[25], many=list(range(150)))
    tokenization_calls = []
    def tokenize(word, *, add_special_tokens):
        assert add_special_tokens is False and word in word_ids
        tokenization_calls.append(word)
        return word_ids[word]
    tokenizer.encode = tokenize
    extractors = definitions("encode.py", {"get_text_valid_tokens_values", "filter_token",
        "get_sparse_logit_multiplier"}, dict(torch=torch, np=np, string=string,
            word_tokenize=lambda text: re.findall(r"\w+|[^\w\s]", text),
            stopwords=NS(words=lambda _: ["a", "is", "in", "the", "near"])))
    extract = extractors.get_text_valid_tokens_values
    data = probe.data
    data.sparse_logits_multiply_view_count = False
    data.sparse_manual = False
    data.sparse_type = "single"
    data.print_sparse = True
    probe.model_args.eol_type = "prompteol"
    probe.model_args.calculate_type = "separate"
    weights = torch.arange(200).float() / 10
    weights[199] = 10000  # A high-scoring token absent from every original-text candidate set.
    weights[12] = weights[13] = 7.125  # Preserve native topk tie order.
    weights[9] = 0.004  # Preserve raw weights even when saved integer weights round to zero.
    original_weights = weights.clone()
    cases = [
        ("caption", "A brown dog is shown standing in the water near a muddy beach. Dog!", 0),
        ("short", "DOG dog dog", 0),
        ("long", "many", 0),
        ("expanded", "many", 2),
        ("empty", "", 0),
        ("stopwords", "A is in the near!", 0),
    ]
    dataset = NS(text_id_list=[case[0] for case in cases], img_id_list=[])
    training = NS(task_type="ir", encode_type="text", fp16=False)
    with tempfile.TemporaryDirectory(prefix="retained-text-") as directory:
        for identifier, text, expansion in cases:
            data.num_expended_tokens = expansion
            data.is_filtered = identifier == "caption"
            filtered_ids = set(range(200))
            legacy = extract(text, tokenizer, weights, id_vocabulary, data, filtered_ids)
            tokens, values, retained_ids = extract(text, tokenizer, weights, id_vocabulary, data,
                                                  filtered_ids, return_retained_ids=True)
            assert tokens == legacy[0] and np.array_equal(values, legacy[1])
            if identifier in {"empty", "stopwords"}:
                assert retained_ids == [] and len(tokens) == 10  # Encoding's fallback remains intact.
            else:
                all_ids = set(index for word in re.findall(r"\w+", text.lower()) if word in word_ids
                              for index in word_ids[word])
                all_ids_tensor = torch.tensor(list(all_ids))
                expected_ids = all_ids_tensor[weights[all_ids_tensor].topk(min(len(all_ids), 128)).indices].tolist()
                assert retained_ids == expected_ids and 199 not in retained_ids
                if identifier == "caption":
                    assert len(retained_ids) == 9 and len(set(retained_ids)) == 9
                    assert any(id_vocabulary[index].startswith(marker) for index in retained_ids)
                    assert 12 in retained_ids and 13 in retained_ids
                    assert "a" not in tokenization_calls and "the" not in tokenization_calls
                elif identifier == "short":
                    assert retained_ids == [9] and values.tolist() == [0]
                else:
                    assert len(retained_ids) == 128
                    assert len(tokens) == 128 + expansion
                    if expansion:
                        assert tokens[-2] == id_vocabulary[199].lower()

            options = diagnostics.PrompteolSparsePrintArguments(
                print_sparse_diagnostics=True, print_sparse_format="json", print_sparse_timing=timing,
                print_sparse_sample_id=identifier, print_sparse_log_dir=str(Path(directory) / identifier))
            printer = diagnostics.build_printer(NS(tokenizer=tokenizer), probe.model_args, data,
                                                 training, options, dataset)
            before_print = list(tokenization_calls)
            output = io.StringIO()
            with contextlib.redirect_stdout(output), printer:
                printer.record_retained_text(identifier, text, weights, retained_ids)
            log = Path(options.print_sparse_log_dir) / "sparse_tokens_rank0.jsonl"
            log_bytes = log.read_bytes()
            rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
            assert len(rows) == 1
            row = rows[0]
            assert row["token_source"] == "retained_text" and row["retained_token_count"] == len(retained_ids)
            assert row["top_k"] == min(len(retained_ids), 10)
            assert row["top_tokens"] == [dict(token_id=index, token=id_vocabulary[index],
                                               weight=float(weights[index])) for index in retained_ids[:10]]
            if timing == "end":
                assert "[prompteol sparse] " not in output.getvalue()
                with contextlib.redirect_stdout(output):
                    printer.finish()
                    printer.finish()  # Finalization is idempotent.
                assert "[prompteol sparse summary] encoding completed" in output.getvalue()
                assert log.read_bytes() == log_bytes
            console_rows = [json.loads(line[len("[prompteol sparse] "):]) for line in output.getvalue().splitlines()
                            if line.startswith("[prompteol sparse] ")]
            assert console_rows == [{key: value for key, value in row.items() if key != "sample_text"}]

            # Text format must also identify empty candidate sets without displaying the encoding fallback.
            if not retained_ids:
                text_output = io.StringIO()
                printer.adapter.print_sparse_format = "text"
                printer.adapter.print_sparse_timing = "immediate"
                with contextlib.redirect_stdout(text_output):
                    printer.record_retained_text(identifier, text, weights, retained_ids)
                assert "Top-0 tokens" in text_output.getvalue()
                assert "No retained text tokens" in text_output.getvalue()
            assert tokenization_calls == before_print and torch.equal(weights, original_weights)
            with pytest.raises(ValueError, match="retained token IDs"):
                printer.record_batch([identifier], [text], [], weights.unsqueeze(0))
            with pytest.raises(ValueError, match="Invalid retained"):
                printer.record_retained_text(identifier, text, weights, [9, 9])


def check_failures_and_json():
    import torch
    probe = environment(initialized=False, size=12, unmapped=False)
    sys.modules["adaptive_view_encoding"] = probe.api
    probe.api.worker_errors = lambda error: (_ for _ in ()).throw(ValueError(error)) if error else None
    model, data, training, options = settings()
    data.sparse_manual = True
    processor = probe.encoder().processor
    dataset = NS(text_id_list=["89110"], img_id_list=["17822"])
    options.print_sparse_sample_id = "absent"
    with pytest.raises(ValueError, match="matched 0 samples"):
        diagnostics.build_printer(processor, model, data, training, options, dataset)
    options.print_sparse_sample_id = "89110"
    with tempfile.TemporaryDirectory(prefix="prompteol-error-") as directory:
        options.print_sparse_log_dir = directory
        printer = diagnostics.build_printer(processor, model, data, training, options, dataset)
        output = io.StringIO()
        handles = []
        with contextlib.redirect_stdout(output), pytest.raises(RuntimeError, match="failure"):
            with printer:
                printer.record_batch(["89110"], ["caption"], [], torch.arange(12).reshape(1, 12).float())
                handles.append(printer.adapter.sparse_print_stream)
                raise RuntimeError("failure")
        assert handles[0].closed and printer.adapter.sparse_print_stream is None
        messages = [json.loads(line[len("[prompteol sparse] "):]) for line in output.getvalue().splitlines()
                    if line.startswith("[prompteol sparse] ")]
        assert len(messages) == 1 and messages[0]["view"] == "whole"
        with pytest.raises(ValueError, match="one vocabulary row"):
            printer.record_batch(["89110"], ["caption"], [], torch.ones(2, 12))
        with pytest.raises(ValueError, match="metadata"):
            printer.record_batch(["89110"], [], [], torch.ones(1, 12))


PROBES = dict(native_batches=check_native_batches, end_native_batches=lambda: check_native_batches("end"),
              failures=check_failures_and_json,
              retained_llama=lambda: check_retained_text_candidates("llava_llama", "immediate"),
              retained_mistral=lambda: check_retained_text_candidates("llava_mistral", "immediate"),
              end_retained_llama=lambda: check_retained_text_candidates("llava_llama", "end"),
              end_retained_mistral=lambda: check_retained_text_candidates("llava_mistral", "end"))


@pytest.mark.parametrize("name", PROBES)
def test_cpu_probe(name):
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch unavailable")
    result = subprocess.run([sys.executable, "-X", "utf8", "-B", str(Path(__file__).resolve()), name],
                            capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.fixture
def shell(tmp_path):
    if BASH is None:
        pytest.skip("Bash unavailable")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "bin").mkdir()
    shutil.copyfile(ROOT / "scripts" / SCRIPT, tmp_path / "scripts" / SCRIPT)
    with (tmp_path / "bin/deepspeed").open("w", encoding="utf-8", newline="\n") as stream:
        stream.write('#!/usr/bin/env bash\nprintf "%s\\0" "$@" > arguments\n')
    (tmp_path / "bin/deepspeed").chmod(0o755)
    bin_path = (tmp_path / "bin").as_posix()
    if os.name == "nt":
        bin_path = "/" + bin_path[0].lower() + bin_path[2:]
    names = {"MODEL_PATH", "DATASET", "DATASET_SPLIT", "ENCODE_TYPE", "DATA_ROOT", "OUTPUT_DIR",
             "DENSE_OUTPUT_ROOT", "SPARSE_OUTPUT_ROOT", "SPARSE_MANUAL", "SPARSE_LENGTH", "TEXT_SPARSE_LENGTH",
             "IMAGE_SPARSE_LENGTH", "PRINT_SPARSE", "PRINT_SPARSE_TOP_K", "PRINT_SPARSE_FORMAT", "PRINT_SPARSE_SAMPLE_ID",
             "PRINT_SPARSE_LOG_DIR", "GPUS", "NUM_NODES", "DEEPSPEED_BIN"}
    names.add("PRINT_SPARSE_TIMING")
    def run(**changes):
        env = {k: v for k, v in os.environ.items() if k not in names}
        env.update(changes, DEEPSPEED_BIN="./bin/deepspeed", MSYS_NO_PATHCONV="1")
        command = 'export PATH=' + shlex.quote(bin_path) + ':/usr/bin:/bin:"$PATH"; exec /usr/bin/bash scripts/' + SCRIPT
        return subprocess.run([BASH, "-c", command], cwd=tmp_path, env=env, capture_output=True,
                              text=True, encoding="utf-8", timeout=20)
    return run


@pytest.mark.parametrize("kind", ["text", "image"])
@pytest.mark.parametrize("manual", ["false", "true"])
def test_script_defaults(shell, tmp_path, kind, manual):
    result = shell(ENCODE_TYPE=kind, SPARSE_MANUAL=manual)
    assert result.returncode == 0, result.stderr
    args = (tmp_path / "arguments").read_bytes().decode().rstrip("\0").split("\0")
    value = lambda key: args[args.index(key) + 1]
    assert args[:3] == ["--num_gpus=2", "--num_nodes=1", "src/encode.py"]
    assert value("--model_name_or_path").endswith("llava-hf-llava-v1.6-mistral-7b-hf")
    assert value("--eol_type") == "prompteol" and value("--calculate_type") == "separate"
    assert value("--llava_image_mode") == "base" and "--fp16" in args
    assert value("--sparse_manual") == manual
    assert value("--print_sparse_diagnostics") == value("--print_sparse") == "true"
    assert value("--print_sparse_top_k") == "10" and value("--print_sparse_format") == "text"
    assert value("--print_sparse_timing") == "end"
    assert value("--print_sparse_sample_id") == ("89110" if kind == "text" else "17822")
    assert value("--diagnostic_data_root") == "/root/autodl-tmp"
    output = f"/root/autodl-fs/output/llava_base_prompteol_print_sparse/sparse_manual_{manual}/{kind}"
    assert value("--output_dir") == output
    assert value("--print_sparse_log_dir") == output + "/sparse_token_logs"
    assert f"Output directory: {output}\n" in result.stdout
    assert f"Sparse token logs: {output}/sparse_token_logs\n" in result.stdout
    assert value("--dense_output_dir") == "/root/autodl-fs/dense_output"
    assert value("--sparse_output_dir") == "/root/autodl-fs/sparse_output"
    assert all(value(key) == "30" for key in ("--sparse_length", "--text_sparse_length", "--image_sparse_length"))
    assert "--adaptive_views" not in args


@pytest.mark.parametrize("target", ["42", ""])
def test_script_overrides(shell, tmp_path, target):
    result = shell(MODEL_PATH="checkpoints/llava llama", PRINT_SPARSE_SAMPLE_ID=target,
                   PRINT_SPARSE_LOG_DIR="saved logs", PRINT_SPARSE="false", PRINT_SPARSE_FORMAT="json",
                   PRINT_SPARSE_TIMING="immediate")
    assert result.returncode == 0, result.stderr
    args = (tmp_path / "arguments").read_bytes().decode().rstrip("\0").split("\0")
    for key, value in (("--model_name_or_path", "checkpoints/llava llama"), ("--print_sparse_sample_id", target),
                       ("--print_sparse_log_dir", "saved logs/sparse_manual_false/text"),
                       ("--print_sparse", "false"), ("--print_sparse_format", "json"),
                       ("--print_sparse_timing", "immediate")):
        assert args[args.index(key) + 1] == value


@pytest.mark.parametrize("kind", ["text", "image"])
@pytest.mark.parametrize("manual,normalized", [("TRUE", "true"), ("1", "true"), ("yes", "true"),
                                               ("FALSE", "false"), ("0", "false"), ("no", "false")])
def test_script_custom_roots_follow_normalized_manual_and_modality(shell, tmp_path, kind, manual, normalized):
    result = shell(ENCODE_TYPE=kind, SPARSE_MANUAL=manual, OUTPUT_DIR="saved outputs/",
                   PRINT_SPARSE_LOG_DIR="saved logs/")
    assert result.returncode == 0, result.stderr
    args = (tmp_path / "arguments").read_bytes().decode().rstrip("\0").split("\0")
    value = lambda key: args[args.index(key) + 1]
    suffix = f"sparse_manual_{normalized}/{kind}"
    assert value("--sparse_manual") == normalized
    assert value("--output_dir") == f"saved outputs/{suffix}"
    assert value("--print_sparse_log_dir") == f"saved logs/{suffix}"


def test_script_custom_output_root_also_relocates_default_logs(shell, tmp_path):
    result = shell(ENCODE_TYPE="image", SPARSE_MANUAL="true", OUTPUT_DIR="saved outputs/")
    assert result.returncode == 0, result.stderr
    args = (tmp_path / "arguments").read_bytes().decode().rstrip("\0").split("\0")
    value = lambda key: args[args.index(key) + 1]
    output = "saved outputs/sparse_manual_true/image"
    assert value("--output_dir") == output
    assert value("--print_sparse_log_dir") == output + "/sparse_token_logs"


@pytest.mark.parametrize("changes", [dict(PRINT_SPARSE="bad"), dict(SPARSE_MANUAL="bad"),
    dict(PRINT_SPARSE_TOP_K="0"), dict(PRINT_SPARSE_FORMAT="bad"), dict(PRINT_SPARSE_TIMING="bad"), dict(ENCODE_TYPE="bad")])
def test_script_rejects_bad_flags_before_launch(shell, tmp_path, changes):
    result = shell(**changes)
    assert result.returncode == 2 and not (tmp_path / "arguments").exists()


def test_script_syntax_and_line_endings():
    path = ROOT / "scripts" / SCRIPT
    assert b"\r" not in path.read_bytes() and not path.read_bytes().startswith(b"\xef\xbb\xbf")
    if BASH is not None:
        result = subprocess.run([BASH, "-n", path.as_posix()], capture_output=True, timeout=10)
        assert result.returncode == 0, result.stderr


if __name__ == "__main__":
    PROBES[sys.argv[1]]()
