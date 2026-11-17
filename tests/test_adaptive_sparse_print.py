"""Sparse logging probes run PyTorch in children, isolated from Pandas tests."""
import ast
import contextlib
import copy
import csv
from dataclasses import dataclass, field
import hashlib
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
from typing import Optional

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = "encode_adaptive_views_print_sparse.sh"
PREFIX = "[adaptive sparse] "
BASH = shutil.which("bash") if os.name != "nt" else next(
    (p for p in ("D:/Git/bin/bash.exe", "C:/Program Files/Git/bin/bash.exe") if Path(p).is_file()), None)


def definitions(filename, names, namespace):
    path = ROOT / "src" / filename
    nodes = [node for node in ast.parse(path.read_text(encoding="utf-8")).body if
             isinstance(node, ast.ImportFrom) and node.module == "__future__" or
             isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return NS(**namespace)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def environment(family="llava_llama", rank=0, initialized=True, size=40, unmapped=True):
    import numpy as np
    import torch
    from torch.nn import functional as F

    dense = NS(configure_model=lambda *a: None, enabled=lambda _: False, configuration=lambda value, *a: value,
               mode_path=lambda path, *a: path, original_path=lambda path: path)
    sys.modules["dense_representation"] = dense
    full_prompt = NS(enabled=lambda _: False)
    sys.modules["freeret_full_prompt"] = full_prompt
    sys.modules["qwen_vl_encoding"] = NS(forward_views=lambda *a, **k: pytest.fail("Unexpected Qwen forward"))
    sys.modules["qwen_vl_pipeline"] = NS(artifact_dirs=lambda *a: [Path("dense"), Path("sparse")],
        init_process=lambda: pytest.fail("Invalid printing options initialized distributed encoding"), dataset_records=None)
    sys.modules["llava_resolution"] = NS(artifact_dir=lambda path, mode: path)
    names = ("img_prompt_for_concat", "text_prompt_for_concat", "fusion_prompt_for_concat",
             "person_retrieval_img_prompt_for_concat", "person_retrieval_text_prompt_for_concat",
             "llama3_template_text_prefix", "llama3_template_content_element",
             "llava_mistral_template_text_prefix", "llava_mistral_template_content_element")
    sys.modules["template"] = NS(**{name: name for name in names})
    views = NS(family_of=lambda config: config.family, view_pool=lambda mode: list(range(10)), pool_mode=lambda _: None,
               encoding_version=lambda _: 18, is_pmi_selection=lambda _: False, digest=digest,
               add_pool_mode=lambda value, mode: value,
               examples=NS(FIELD="example_sampling", add_configuration=lambda value, policy: value))
    lens = NS(enabled=lambda args: getattr(args, "use_lens_sparse", False), validate=lambda *a: None,
              configuration=lambda _: None, mark_path=lambda path, config: path)
    stc = NS(enabled=lambda _: False, configuration=lambda _: None, mark_path=lambda path, config: path)
    namespace = dict(torch=torch, F=F, np=np, copy=copy, json=json, Path=Path, views=views, lens=lens, stc=stc,
        pickle=pickle, contextmanager=contextlib.contextmanager,
        full_prompt=full_prompt,
        prompt_constraints=NS(enabled=lambda _: False, validate_selection=lambda *a: None),
        probability_mass=NS(add_metadata=lambda value, selection: value),
        option_shuffle=NS(FIELD="option_shuffle", add_metadata=lambda value, policy: value, mark_path=lambda path, policy: path),
        configure_processor=lambda *a: None, ADAPTIVE_EOL_TYPES=("disassembleeol_separate", "disassembleeol_separate_origin_text"),
        dist=NS(is_initialized=lambda: initialized,
                get_rank=lambda: rank if initialized else pytest.fail("Read rank before distributed initialization")))
    api = definitions("adaptive_view_encoding.py", {
        "validate_encoding_method", "validate_sparse_print", "print_sparse_views", "AdaptiveEncoder",
        "_defer_sparse_print", "_emit_sparse_record", "finish_sparse_print",
        "collect_encoding_records", "worker_errors", "sparse_print_log", "write_shard",
        "sparse_vectors", "uncompressed_sparse_vectors", "encoding_configuration", "metadata", "artifact_dirs", "run_encode",
    }, namespace)
    extractors = definitions("encode.py", {"get_text_valid_disassemble_tokens_values", "get_img_valid_disassemble_tokens_values"},
        dict(torch=torch, np=np, string=string, word_tokenize=str.split,
             stopwords=NS(words=lambda _: []), dist=namespace["dist"]))
    sys.modules["encode"] = extractors
    marker = "\u0120" if family == "llava_llama" else "\u2581"
    vocab = {marker + f"w{i}": i for i in range(size)}
    # Tokenizer-only IDs and output-head-only IDs must not be printed as each other.
    if unmapped:
        vocab["tokenizer_only"] = size + 5
    tokenizer = NS(get_vocab=lambda: vocab)
    data = NS(print_sparse=False, reps_loc="after_pad", sparse_length=30, text_sparse_length=30, image_sparse_length=30,
              sparse_manual=False, sparse_value_type="sum", sparse_lower_or_upper="lower", sparse_value_mean=False,
              is_filtered=False, num_expended_tokens=0, tbpr_type="type", cir_type="type", dataset_name="flickr",
              dataset_split="test", llava_image_mode="base")
    model_args = NS(eol_type="disassembleeol_separate", calculate_type="concat", model_name_or_path="model")
    selected = [{"label": "A", "aspect": "objects"}, {"label": "D", "aspect": "colors"},
                {"label": "J", "aspect": "activities"}]
    selection = dict(selected=selected, selection_modality="both", views_fingerprint="locked-views", fingerprint="locked")
    records = [dict(id="caption-1\nline", text="caption"), dict(id="image-1", image="image.jpg"),
               dict(id="fused-1", image="image.jpg", text="caption")]
    raw = torch.zeros(len(records), 8, size + int(unmapped))
    raw[:, 0, 0] = 1e5  # The whole-input dense view must never appear in logs.
    for row_index in range(len(records)):
        for view_index, position in enumerate((2, 4, 6)):
            raw[row_index, position, :size] = torch.roll(torch.arange(1, size + 1).float(), view_index * 3)
            if unmapped:
                raw[row_index, position, size] = 1e6  # No corresponding tokenizer token.
    hidden = torch.ones(len(records), 8, 2)

    class Model:
        config = NS(family=family, hidden_size=2)

        def get_input_embeddings(self):
            return NS(weight=torch.zeros(1, 2))

        def __call__(self, **kwargs):
            return NS(logits=raw.clone(), hidden_states=[hidden.clone()])

    def encoder(enabled=True, top_k=10, print_format="json", **print_options):
        data.print_sparse = enabled
        instance = api.AdaptiveEncoder(Model(), NS(tokenizer=tokenizer), model_args, data, "ir", selection,
                                        print_sparse_top_k=top_k, print_sparse_format=print_format, **print_options)
        instance.prepare = lambda rows: NS(inputs={"input_ids": torch.zeros(len(rows), 8, dtype=torch.long),
            "attention_mask": torch.ones(len(rows), 8)}, spans=[[(0, 2), (2, 4), (4, 6), (6, 8)]] * len(rows),
            prompts=["example"] * len(rows))
        return instance

    return NS(api=api, namespace=namespace, encoder=encoder, data=data, model_args=model_args, selection=selection,
              records=records, torch=torch, raw=raw, size=size)


def encode_and_capture(probe, enabled=True, top_k=10, vectors=False):
    stream = io.StringIO()
    encoder = probe.encoder(enabled, top_k)
    with contextlib.redirect_stdout(stream):
        sparse, dense = encoder.encode(probe.records)
        result = probe.api.sparse_vectors(encoder, probe.records, sparse) if vectors else None
    lines = stream.getvalue().splitlines()
    assert all(line.startswith(PREFIX) for line in lines), lines
    return [json.loads(line[len(PREFIX):]) for line in lines], sparse, dense, result


def check_logging():
    for family in ("llava_llama", "llava_mistral"):
        for rank in (0, 1):
            probe = environment(family, rank)
            before = copy.deepcopy(probe.selection)
            rows, sparse, _, _ = encode_and_capture(probe)
            assert len(rows) == 9
            for sample, values in zip(probe.records, sparse):
                lines = [row for row in rows if row["sample_id"] == sample["id"]]
                assert [row["view"] for row in lines] == ["A", "D", "J"]
                assert [row["aspect"] for row in lines] == ["objects", "colors", "activities"]
                for index, line in enumerate(lines):
                    expected, ids = values[index, :probe.size].topk(10)
                    assert line["rank"] == rank and line["top_k"] == 10
                    assert [token["token_id"] for token in line["top_tokens"]] == ids.tolist()
                    assert [token["weight"] for token in line["top_tokens"]] == expected.tolist()
                    assert all(token["token"].endswith(f'w{token["token_id"]}') for token in line["top_tokens"])
                    assert line["weight_type"] == "log1p_relu"
            assert [rows[i]["modality"] for i in (0, 3, 6)] == ["text", "image", "image_text"]
            assert probe.selection == before


def check_validation():
    probe = environment(initialized=False, size=3)
    rows, _, _, _ = encode_and_capture(probe, top_k=10)
    assert len(rows) == 9 and all(row["rank"] == 0 and row["top_k"] == 3 for row in rows)
    for value in (0, -1, True, 1.5, "10"):
        with pytest.raises(ValueError, match="positive integer"):
            probe.encoder(True, value)
    rows, _, _, _ = encode_and_capture(probe, enabled=False, top_k=0)
    assert rows == []
    with pytest.raises(ValueError, match="text or json"):
        probe.encoder(print_format="invalid")
    with pytest.raises(ValueError, match="immediate or end"):
        probe.encoder(print_sparse_timing="invalid")
    probe.model_args.use_lens_sparse = True
    with pytest.raises(ValueError, match="LENS cluster vocabulary"):
        probe.encoder()


def check_values_unchanged():
    for family in ("llava_llama", "llava_mistral"):
        for rank in (0, 1):
            probe = environment(family, rank, unmapped=False)
            off, original_sparse, original_dense, original_vectors = encode_and_capture(probe, enabled=False, vectors=True)
            # Three different per-view Top-30 sets merge into 36 distinct terms.
            assert off == [] and all(len(vector) == 36 for vector in original_vectors)
            for count in (1, 10, 50):
                rows, sparse, dense, vectors = encode_and_capture(probe, top_k=count, vectors=True)
                assert len(rows) == 9 and all(row["top_k"] == min(count, 40) for row in rows)
                assert probe.torch.equal(original_dense, dense)
                assert all(probe.torch.equal(a, b) for a, b in zip(original_sparse, sparse))
                assert vectors == original_vectors and probe.data.print_sparse is True

    # Non-adaptive FreeRet also calls this helper; retain its original printer.
    probe = environment(unmapped=False)
    encoder = probe.encoder()
    with contextlib.redirect_stdout(io.StringIO()):
        sparse, _ = encoder.encode(probe.records)
    legacy = NS(family=encoder.family, processor=encoder.processor, data=probe.data, model_args=probe.model_args)
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        probe.api.uncompressed_sparse_vectors(legacy, [probe.records[1]], [sparse[1]])
    assert len(stream.getvalue().splitlines()) == 3
    assert PREFIX not in stream.getvalue() and probe.data.print_sparse is True


def check_fingerprints_unchanged():
    probe = environment()
    original = copy.deepcopy(probe.selection)
    identities, paths = [], []
    for print_format in ("json", "text"):
        for enabled, count in ((False, 10), (True, 1), (True, 10), (True, 50)):
            probe.encoder(enabled, count, print_format)
            identities.append(probe.api.metadata(probe.model_args, probe.data, "ir", "text", {"family": "llava_llama"}, probe.selection))
            paths.append(probe.api.artifact_dirs(probe.model_args, probe.data, NS(), probe.selection))
    assert all(value == identities[0] for value in identities)
    assert all(value == paths[0] for value in paths) and probe.selection == original
    assert "print_sparse" not in json.dumps(identities) and "print_sparse_top_k" not in json.dumps(identities)
    assert "print_sparse_format" not in json.dumps(identities)
    for target, directory in (("caption-1\nline", "token logs"), (None, None)):
        probe.encoder(print_sparse_sample_id=target, print_sparse_log_dir=directory, print_sparse_timing="end")
        assert probe.api.metadata(probe.model_args, probe.data, "ir", "text", {"family": "llava_llama"}, probe.selection) == identities[0]
        assert probe.api.artifact_dirs(probe.model_args, probe.data, NS(), probe.selection) == paths[0]


def check_text_logging():
    token_pattern = re.compile(r'("(?:[^"\\]|\\.)*") \(id=(\d+), weight=([-+.\deE]+)\)')
    for family in ("llava_llama", "llava_mistral"):
        for rank in (0, 1):
            probe = environment(family, rank, unmapped=False)
            encoder = probe.encoder(print_format="text")
            vocabulary = dict(encoder.processor.tokenizer.get_vocab())
            marker = "\u0120" if family == "llava_llama" else "\u2581"
            vocabulary.pop(marker + "w39")
            vocabulary[marker + 'dog,"\\\n'] = 39
            encoder.processor.tokenizer.get_vocab = lambda: vocabulary
            encoder.selected[0]["aspect"] = 'objects "and"\nrelations'
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                sparse, dense = encoder.encode(probe.records)
                vectors = probe.api.sparse_vectors(encoder, probe.records, sparse)
            lines = stream.getvalue().splitlines()
            expected, json_sparse, json_dense, json_vectors = encode_and_capture(probe, vectors=True)
            assert len(lines) == 2 * len(expected) == 18
            for offset, row in enumerate(expected):
                header, tokens = lines[offset * 2:offset * 2 + 2]
                assert header.startswith(PREFIX + f"rank={rank} ")
                assert 'sample_id=' + json.dumps(row["sample_id"], ensure_ascii=False) in header
                assert f"modality={row['modality']} view={row['view']} " in header
                assert 'aspect=' + json.dumps(row["aspect"], ensure_ascii=False) in header
                assert tokens.startswith("  Top-10 tokens (log1p_relu): ")
                parsed = token_pattern.findall(tokens)
                assert len(parsed) == 10
                assert [json.loads(token) for token, _, _ in parsed] == [t["token"] for t in row["top_tokens"]]
                assert [int(index) for _, index, _ in parsed] == [t["token_id"] for t in row["top_tokens"]]
                weights = [float(weight) for _, _, weight in parsed]
                assert weights == sorted(weights, reverse=True)
                assert weights == pytest.approx([t["weight"] for t in row["top_tokens"]], rel=1e-5)
            assert probe.torch.equal(dense, json_dense)
            assert all(probe.torch.equal(a, b) for a, b in zip(sparse, json_sparse))
            assert vectors == json_vectors


def check_text_limits_and_disabled():
    probe = environment(initialized=False, size=3)
    encoder = probe.encoder(print_format="text")
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        encoder.encode(probe.records)
    lines = stream.getvalue().splitlines()
    assert len(lines) == 18
    assert all(line.startswith("  Top-3 tokens (log1p_relu): ") for line in lines[1::2])
    assert all("rank=0" in line for line in lines[::2])
    encoder = probe.encoder(enabled=False, print_format="text")
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        encoder.encode(probe.records)
    assert stream.getvalue() == ""


def check_early_validation():
    probe = environment()
    probe.data.print_sparse = True
    for options, pattern in ((NS(print_sparse_top_k=0), "positive integer"),
                             (NS(print_sparse_format="invalid"), "text or json"),
                              (NS(print_sparse_timing="invalid"), "immediate or end"),
                             (NS(print_sparse_top_k=10, use_lens_sparse=True), "LENS cluster vocabulary")):
        with pytest.raises(ValueError, match=pattern):
            probe.api.run_encode(probe.model_args, probe.data, NS(), options)


def check_full_shards_filtered_console_and_logs():
    sentence = "A brown dog is shown standing in the water near a muddy beach."
    for family in ("llava_llama", "llava_mistral"):
        for kind in ("text", "image"):
            probe = environment(family, unmapped=False)
            target_id = "89110" if kind == "text" else "17822"
            if kind == "image":
                probe.records[:] = [dict(id=f"image-{i}", image=f"/images/image-{i}.jpg", text="") for i in range(3)]
                probe.records[1] = dict(id=target_id, image="/images/411175971.jpg", text="")
            else:
                probe.records[1] = dict(id=target_id, text=sentence)
            probe.data.per_device_batch_size = 2
            with tempfile.TemporaryDirectory(prefix="sparse-print-") as directory:
                root = Path(directory)
                roots = (root / "dense", root / "sparse")
                roots[0].mkdir()
                (roots[1] / "collection").mkdir(parents=True)
                for rank in (0, 1):
                    probe.namespace["dist"].get_rank = lambda rank=rank: rank
                    encoder = probe.encoder(enabled=False)
                    with contextlib.redirect_stdout(io.StringIO()):
                        probe.api.write_shard(encoder, probe.records[rank::2], probe.data, roots, rank)
                snapshot = {p: p.read_bytes() for base in roots for p in base.rglob("corpus_*")}
                assert {str(identifier) for p in roots[0].glob("*.pkl")
                        for identifier in pickle.loads(p.read_bytes())[1]} == {row["id"] for row in probe.records}
                first_logs = None
                for print_format in ("text", "json"):
                    for rank in (0, 1):
                        probe.namespace["dist"].get_rank = lambda rank=rank: rank
                        encoder = probe.encoder(print_format=print_format, print_sparse_sample_id=target_id,
                                                print_sparse_log_dir=str(root / "logs"))
                        stream = io.StringIO()
                        with contextlib.redirect_stdout(stream):
                            probe.api.write_shard(encoder, probe.records[rank::2], probe.data, roots, rank)
                        output = stream.getvalue()
                        assert encoder.sparse_print_stream is None
                        rows = [json.loads(line) for line in (root / "logs" / f"sparse_tokens_rank{rank}.jsonl").read_text(encoding="utf-8").splitlines()]
                        assert len(rows) == len(probe.records[rank::2]) * 3
                        assert {row["sample_id"] for row in rows} == {row["id"] for row in probe.records[rank::2]}
                        assert all(row["rank"] == rank and len(row["top_tokens"]) == 10 for row in rows)
                        assert all(row["view"] in ("A", "D", "J") for row in rows)
                        assert all(row["sample_text"] == next(record.get("text") or "" for record in probe.records if record["id"] == row["sample_id"]) for row in rows)
                        for row in rows:
                            weights = [token["weight"] for token in row["top_tokens"]]
                            assert weights == sorted(weights, reverse=True)
                        if rank == 0:
                            assert PREFIX not in output  # Target is owned by rank 1.
                        elif print_format == "text":
                            if kind == "image":
                                assert 'sample_id="17822" image="/images/411175971.jpg"' in output
                                assert "caption=" not in output and sentence not in output
                                assert "modality=image" in output
                            else:
                                assert output.count("caption=") == 1 and sentence in output
                            assert output.count("Top-10 tokens (log1p_relu):") == 3
                            assert all(f"view={label} " in output for label in ("A", "D", "J"))
                            assert 'sample_id="fused-1"' not in output and 'sample_id="caption-1' not in output
                        else:
                            messages = [json.loads(line[len(PREFIX):]) for line in output.splitlines() if line.startswith(PREFIX)]
                            assert len(messages) == 3 and all(row["sample_id"] == target_id for row in messages)
                            assert all("sample_text" not in row for row in messages)
                    assert all(path.read_bytes() == contents for path, contents in snapshot.items())
                    logs = {p.name: p.read_bytes() for p in (root / "logs").glob("*.jsonl")}
                    assert set(logs) == {"sparse_tokens_rank0.jsonl", "sparse_tokens_rank1.jsonl"}
                    if first_logs is None:
                        first_logs = logs
                    else:
                        assert logs == first_logs  # Fresh per-run logs, not appended; format independent.


def check_log_cleanup():
    probe = environment(unmapped=False)
    probe.data.per_device_batch_size = 1
    with tempfile.TemporaryDirectory(prefix="sparse-cleanup-") as directory:
        root = Path(directory)
        roots = (root / "dense", root / "sparse")
        roots[0].mkdir()
        (roots[1] / "collection").mkdir(parents=True)
        encoder = probe.encoder(print_sparse_sample_id="absent", print_sparse_log_dir=str(root / "failed"))
        original = encoder.encode
        handles = []
        def fail(records):
            original(records)
            handles.append(encoder.sparse_print_stream)
            raise RuntimeError("encoding failed")
        encoder.encode = fail
        with contextlib.redirect_stdout(io.StringIO()), pytest.raises(RuntimeError, match="encoding failed"):
            probe.api.write_shard(encoder, probe.records, probe.data, roots, 0)
        assert handles[0].closed and encoder.sparse_print_stream is None
        assert len((root / "failed/sparse_tokens_rank0.jsonl").read_text(encoding="utf-8").splitlines()) == 3
        encoder = probe.encoder(enabled=False, print_sparse_sample_id="absent", print_sparse_log_dir=str(root / "disabled"))
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            probe.api.write_shard(encoder, probe.records, probe.data, roots, 0)
        assert stream.getvalue() == "" and not (root / "disabled").exists()


def check_target_validation():
    probe = environment(initialized=False)
    module = sys.modules["qwen_vl_pipeline"]
    module.dataset_records = lambda *a, **k: probe.records
    api = NS(build_dataset=lambda args: object())
    result = probe.api.collect_encoding_records(api, NS(), "ir", "text", print_sparse=True,
                                               print_sparse_sample_id="caption-1\nline")
    assert result is probe.records and len(result) == 3
    with pytest.raises(ValueError, match="matched 0 samples"):
        probe.api.collect_encoding_records(api, NS(), "ir", "image", print_sparse=True, print_sparse_sample_id="89110")
    module.dataset_records = lambda *a, **k: [probe.records[0], probe.records[0]]
    with pytest.raises(ValueError, match="matched 2 samples"):
        probe.api.collect_encoding_records(api, NS(), "ir", "text", print_sparse=True, print_sparse_sample_id="caption-1\nline")
    assert probe.api.collect_encoding_records(api, NS(), "ir", "text", print_sparse=False, print_sparse_sample_id="absent")
    probe.namespace["dist"].is_initialized = lambda: True
    probe.namespace["dist"].get_world_size = lambda: 2
    def gather(errors, error):
        errors[:] = [error, "target validation failed on another worker"]
    probe.namespace["dist"].all_gather_object = gather
    with pytest.raises(ValueError, match="rank 1: target validation failed"):
        probe.api.collect_encoding_records(api, NS(), "ir", "text")


def check_end_summary():
    sentence = "A brown dog is shown standing in the water near a muddy beach."
    for family in ("llava_llama", "llava_mistral"):
        for kind in ("text", "image"):
            for print_format in ("text", "json"):
                probe = environment(family, unmapped=False)
                target = "89110" if kind == "text" else "17822"
                probe.records[1] = (dict(id=target, text=sentence) if kind == "text" else
                                    dict(id=target, image=Path("images/411175971.jpg"), text=""))
                probe.data.per_device_batch_size = 2
                role = dict(rank=0)
                probe.namespace["dist"].get_rank = lambda: role["rank"]
                probe.namespace["dist"].get_world_size = lambda: 2
                with tempfile.TemporaryDirectory(prefix="end-summary-") as directory:
                    root = Path(directory)
                    roots = (root / "dense", root / "sparse")
                    roots[0].mkdir()
                    (roots[1] / "collection").mkdir(parents=True)
                    for rank in (0, 1):
                        role["rank"] = rank
                        with contextlib.redirect_stdout(io.StringIO()):
                            probe.api.write_shard(probe.encoder(enabled=False), probe.records[rank::2], probe.data, roots, rank)
                    original = {p: p.read_bytes() for base in roots for p in base.rglob("corpus_*")}
                    encoders = []
                    for rank in (0, 1):
                        role["rank"] = rank
                        encoder = probe.encoder(print_format=print_format, print_sparse_timing="end",
                                                print_sparse_sample_id=target, print_sparse_log_dir=str(root / "logs"))
                        output = io.StringIO()
                        with contextlib.redirect_stdout(output):
                            probe.api.write_shard(encoder, probe.records[rank::2], probe.data, roots, rank)
                        assert PREFIX not in output.getvalue()
                        encoders.append(encoder)
                    assert all(path.read_bytes() == value for path, value in original.items())
                    logs = {p: p.read_bytes() for p in (root / "logs").glob("*.jsonl")}
                    assert sum(len(value.splitlines()) for value in logs.values()) == 9
                    pending = [getattr(encoder, "_sparse_print_pending", []) for encoder in encoders]
                    assert pending[0] == [] and len(pending[1]) == 1
                    assert len(pending[1][0]["payloads"]) == 3
                    # This also rejects accidentally cached device tensors or PIL images.
                    pending = json.loads(json.dumps(pending))
                    calls = []
                    def gather(result, local):
                        calls.append(role["rank"])
                        assert local == pending[role["rank"]]
                        result[:] = copy.deepcopy(pending)
                    probe.namespace["dist"].all_gather_object = gather
                    role["rank"] = 1
                    output = io.StringIO()
                    with contextlib.redirect_stdout(output):
                        probe.api.finish_sparse_print(encoders[1])
                    assert output.getvalue() == ""
                    role["rank"] = 0
                    output = io.StringIO()
                    with contextlib.redirect_stdout(output):
                        probe.api.finish_sparse_print(encoders[0])
                    rendered = output.getvalue()
                    assert "[adaptive sparse summary] encoding completed" in rendered
                    if print_format == "json":
                        rows = [json.loads(line[len(PREFIX):]) for line in rendered.splitlines() if line.startswith(PREFIX)]
                        assert rows == pending[1][0]["payloads"]
                    else:
                        assert rendered.count("Top-10 tokens (log1p_relu)") == 3
                        assert rendered.count("rank=1 ") == 3 and "rank=0 " not in rendered
                        assert sentence in rendered if kind == "text" else "411175971.jpg" in rendered
                    assert calls == [1, 0]
                    assert all(encoder._sparse_print_pending == [] and encoder._sparse_print_finished for encoder in encoders)
                    output = io.StringIO()
                    with contextlib.redirect_stdout(output):
                        probe.api.finish_sparse_print(encoders[0])
                    assert calls == [1, 0] and output.getvalue() == ""
                    assert all(path.read_bytes() == value for path, value in logs.items())


def check_end_edge_cases():
    probe = environment(initialized=False, unmapped=False)
    encoder = probe.encoder(print_format="text", print_sparse_timing="end", print_sparse_sample_id="")
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        encoder.encode(probe.records)
        probe.api.finish_sparse_print(encoder)
    assert output.getvalue().count("Top-10 tokens (log1p_relu)") == 9
    assert not hasattr(encoder, "_sparse_print_pending") and "summary]" not in output.getvalue()
    encoder = probe.encoder(enabled=False, print_sparse_timing="end", print_sparse_sample_id="absent")
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        encoder.encode(probe.records)
        probe.api.finish_sparse_print(encoder)
    assert output.getvalue() == ""
    encoder = probe.encoder(print_sparse_timing="end", print_sparse_sample_id="89110")
    with pytest.raises(ValueError, match="No sparse token results collected"):
        probe.api.finish_sparse_print(encoder)
    probe.records[0]["id"] = "89110"
    with tempfile.TemporaryDirectory(prefix="failed-end-") as directory:
        encoder = probe.encoder(print_sparse_timing="end", print_sparse_sample_id="89110", print_sparse_log_dir=directory)
        output = io.StringIO()
        with contextlib.redirect_stdout(output), pytest.raises(RuntimeError, match="encoding failed"):
            with probe.api.sparse_print_log(encoder, 0):
                encoder.encode(probe.records)
                raise RuntimeError("encoding failed")
        assert PREFIX not in output.getvalue() and "summary]" not in output.getvalue()
        assert encoder.sparse_print_stream is None
        assert len((Path(directory) / "sparse_tokens_rank0.jsonl").read_text(encoding="utf-8").splitlines()) == 9


PROBES = dict(logging=check_logging, validation=check_validation, values=check_values_unchanged,
               fingerprints=check_fingerprints_unchanged, early_validation=check_early_validation,
               text_logging=check_text_logging, text_limits=check_text_limits_and_disabled,
               file_filter=check_full_shards_filtered_console_and_logs, log_cleanup=check_log_cleanup,
               target_validation=check_target_validation, end_summary=check_end_summary, end_edge_cases=check_end_edge_cases)


@pytest.mark.parametrize("probe", PROBES)
def test_sparse_print_cpu_probe(probe):
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch not installed")
    result = subprocess.run([sys.executable, "-X", "utf8", "-B", str(Path(__file__).resolve()), probe],
                            capture_output=True, text=True, encoding="utf-8", timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr


def test_adaptive_argument_exposes_independent_print_top_k(monkeypatch):
    namespace = dict(__name__="_adaptive_print_args", dataclass=dataclass, field=field, Optional=Optional,
                     SELECTION_MODALITIES=("text", "image", "both"))
    monkeypatch.setitem(sys.modules, namespace["__name__"], NS(**namespace))
    arguments = definitions("adaptive_views.py", {"AdaptiveViewArguments"}, namespace).AdaptiveViewArguments
    assert arguments().print_sparse_top_k == 10
    assert arguments().print_sparse_format == "json"
    assert arguments().print_sparse_timing == "immediate"
    assert arguments().print_sparse_sample_id is None and arguments().print_sparse_log_dir is None
    assert arguments.__dataclass_fields__["print_sparse_format"].metadata["choices"] == ("text", "json")
    assert arguments(print_sparse_top_k=3).print_sparse_top_k == 3


@pytest.fixture
def shell(tmp_path):
    if BASH is None:
        pytest.skip("Bash not installed")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "bin").mkdir()
    shutil.copyfile(ROOT / "scripts" / SCRIPT, tmp_path / "scripts" / SCRIPT)
    path = tmp_path / "bin" / "deepspeed"
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write('#!/usr/bin/env bash\nprintf "%s\\0" "$@" > arguments\n')
    path.chmod(0o755)
    bin_path = (tmp_path / "bin").as_posix()
    if os.name == "nt":
        bin_path = "/" + bin_path[0].lower() + bin_path[2:]

    def run(**overrides):
        names = {"MODEL_PATH", "DATASET", "DATASET_SPLIT", "TASK_TYPE", "ENCODE_TYPE", "DATA_ROOT", "OUTPUT_DIR",
                  "DENSE_OUTPUT_ROOT", "SPARSE_OUTPUT_ROOT", "PRINT_SPARSE", "PRINT_SPARSE_TOP_K", "PRINT_SPARSE_FORMAT", "VIEW_SELECTION_FILE",
                  "PRINT_SPARSE_SAMPLE_ID", "PRINT_SPARSE_LOG_DIR",
                  "PRINT_SPARSE_TIMING",
                 "USE_VIEW_PROBABILITY_MASS", "VIEW_PROBABILITY_MASS_THRESHOLD", "USE_VIEW_PMI", "VIEW_PMI_PRIOR_PROMPT_MODE",
                 "GPUS", "NUM_NODES", "DEEPSPEED_BIN"}
        env = {key: value for key, value in os.environ.items() if key not in names}
        env.update(overrides, DEEPSPEED_BIN="./bin/deepspeed", MSYS_NO_PATHCONV="1")
        command = 'export PATH=' + shlex.quote(bin_path) + ':/usr/bin:/bin:"$PATH"; exec /usr/bin/bash scripts/' + SCRIPT
        result = subprocess.run([BASH, "-c", command], cwd=tmp_path, env=env, capture_output=True,
                                text=True, encoding="utf-8", timeout=20)
        return result

    return run


@pytest.mark.parametrize("encode_type,printing,count,print_format", [
    ("text", "true", "10", None), ("image", "true", "10", None), ("image", "false", "5", "json")])
def test_script_printing_and_encoding_are_independent(shell, tmp_path, encode_type, printing, count, print_format):
    options = {"PRINT_SPARSE_FORMAT": print_format} if print_format is not None else {}
    result = shell(ENCODE_TYPE=encode_type, PRINT_SPARSE=printing, PRINT_SPARSE_TOP_K=count,
                    VIEW_SELECTION_FILE="selected views.json", **options)
    assert result.returncode == 0, result.stderr
    arguments = (tmp_path / "arguments").read_bytes().decode().rstrip("\0").split("\0")
    value = lambda key: arguments[arguments.index(key) + 1]
    assert arguments[:3] == ["--num_gpus=2", "--num_nodes=1", "src/encode_adaptive_views.py"]
    assert value("--print_sparse") == printing and value("--print_sparse_top_k") == count
    assert value("--print_sparse_format") == (print_format or "text")
    assert value("--print_sparse_timing") == "end"
    assert value("--print_sparse_sample_id") == ("17822" if encode_type == "image" else "89110")
    assert value("--print_sparse_log_dir") == "/root/autodl-fs/output/adaptive_print_sparse/sparse_token_logs"
    assert value("--encode_type") == encode_type and value("--view_selection_file") == "selected views.json"
    assert value("--model_name_or_path") == "/root/autodl-fs/checkpoints/llava-hf-llama3-llava-next-8b-hf"
    assert value("--dataset_name") == "flickr" and value("--task_type") == "ir" and value("--dataset_split") == "test"
    assert value("--adaptive_data_root") == "/root/autodl-tmp" and value("--llava_image_mode") == "base"
    assert value("--dense_output_dir") == "/root/autodl-fs/dense_output"
    assert value("--sparse_output_dir") == "/root/autodl-fs/sparse_output"
    assert value("--view_probability_mass") == "true" and value("--view_probability_mass_threshold") == "0.9"
    assert value("--view_selection_hierarchical") == "true" and value("--view_example_max_tokens") == "0"
    assert all(value(key) == "false" for key in ("--view_selection_pmi", "--view_selection_token_variants",
        "--view_selection_prefix_tokens", "--use_stc_sparse", "--use_lens_sparse", "--use_freeret_dense",
        "--use_freeret_full_prompt", "--use_freeret_prompt_constraints"))
    assert value("--eol_type") == "disassembleeol_separate" and value("--calculate_type") == "concat"
    assert all(value(key) == "30" for key in ("--sparse_length", "--text_sparse_length", "--image_sparse_length"))


@pytest.mark.parametrize("target", ["42", ""])
@pytest.mark.parametrize("encode_type", ["text", "image"])
def test_script_target_and_log_directory_overrides(shell, tmp_path, target, encode_type):
    result = shell(ENCODE_TYPE=encode_type, PRINT_SPARSE_SAMPLE_ID=target, PRINT_SPARSE_LOG_DIR="saved token logs",
                   PRINT_SPARSE_TIMING="immediate")
    assert result.returncode == 0, result.stderr
    arguments = (tmp_path / "arguments").read_bytes().decode().rstrip("\0").split("\0")
    assert arguments[arguments.index("--print_sparse_sample_id") + 1] == target
    assert arguments[arguments.index("--print_sparse_log_dir") + 1] == "saved token logs"
    assert arguments[arguments.index("--encode_type") + 1] == encode_type
    assert arguments[arguments.index("--print_sparse_timing") + 1] == "immediate"


def test_flickr_caption_and_image_records_use_different_native_ids():
    path = ROOT / "data/flickr/flickr_test.csv"
    if not path.is_file():
        pytest.skip("Local Flickr annotations unavailable")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        row = next(row for row in csv.DictReader(stream) if row["sentid"] == "89110")
    assert row["caption"] == "A brown dog is shown standing in the water near a muddy beach."
    assert row["imgid"] == "17822" and row["filename"] == "411175971.jpg"
    api = definitions("qwen_vl_pipeline.py", {"dataset_records"}, {"Path": Path})
    class Dataset:
        img_id_list = [row["imgid"]]
        text_id_list = [row["sentid"]]
        text_dict = {row["sentid"]: row["caption"]}
        def __getitem__(self, index):
            return row["caption"], "/images/" + row["filename"]
    dataset = Dataset()
    text_record, = api.dataset_records(dataset, "ir", "text")
    image_record, = api.dataset_records(dataset, "ir", "image")
    assert text_record["id"] == "89110" and text_record["text"] == row["caption"]
    assert image_record == dict(id="17822", native_id="17822", image="/images/411175971.jpg", text="")


@pytest.mark.parametrize("overrides", [dict(PRINT_SPARSE="invalid"), dict(PRINT_SPARSE_TOP_K="0"),
                                        dict(PRINT_SPARSE_TOP_K="-1"), dict(PRINT_SPARSE_FORMAT="invalid"),
                                        dict(PRINT_SPARSE_TIMING="invalid")])
def test_script_invalid_print_options_do_not_launch_deepspeed(shell, tmp_path, overrides):
    result = shell(**overrides)
    assert result.returncode == 2 and "PRINT_SPARSE" in result.stderr
    assert not (tmp_path / "arguments").exists()


def test_script_syntax_and_utf8_lf():
    if BASH is None:
        pytest.skip("Bash not installed")
    source = ROOT / "scripts" / SCRIPT
    data = source.read_bytes()
    assert not data.startswith(b"\xef\xbb\xbf") and b"\r" not in data
    result = subprocess.run([BASH, "-n", source.as_posix()], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr


if __name__ == "__main__":
    PROBES[sys.argv[1]]()
