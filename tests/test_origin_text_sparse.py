"""CPU probes without model weights or NLTK downloads.

Use child processes to keep PyTorch's native runtime separate from the
NumPy/Pandas retrieval-statistics tests in the same test suite.
"""
import ast
import importlib.util
import string
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def extractors():
    import numpy as np
    import torch

    source = Path(__file__).resolve().parents[1] / "src" / "encode.py"
    names = {"get_text_valid_disassemble_tokens_values", "get_img_valid_disassemble_tokens_values"}
    nodes = [node for node in ast.parse(source.read_text(encoding="utf-8")).body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {
        "torch": torch, "np": np, "string": string, "word_tokenize": str.split,
        "stopwords": SimpleNamespace(words=lambda language: ["the"]),
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    return SimpleNamespace(**namespace)


def arguments():
    data = SimpleNamespace(sparse_manual=False, sparse_length=30, sparse_value_type="sum",
                           sparse_lower_or_upper="lower", is_filtered=False, print_sparse=False)
    model = SimpleNamespace(eol_type="disassembleeol_separate_origin_text")
    return data, model


def check_original_tokens_only_and_five_view_sum(extractors, arguments):
    torch = extractors.torch
    data, model = arguments
    tokenizer = SimpleNamespace(encode=lambda word, add_special_tokens=False: [{"cat": 1, "dog": 2}[word]])
    logits = torch.zeros(5, 6)
    logits[:, 1] = torch.tensor([.1, .2, .3, .4, .5])
    logits[:, 2] = torch.tensor([1., 2., 3., 4., 5.])
    logits[:, 3] = 1e6  # This token must be excluded despite its much higher score.
    vocab = {0: "unused", 1: "cat", 2: "dog", 3: "outside", 4: "the", 5: "."}
    tokens, values = extractors.get_text_valid_disassemble_tokens_values(
        "The CAT dog .", tokenizer, logits, vocab, data, set(), None, model)
    assert dict(zip(tokens, values)) == {"cat": 150, "dog": 1500}


def check_original_text_limit_is_128_per_view(extractors, arguments):
    torch = extractors.torch
    data, model = arguments
    tokenizer = SimpleNamespace(encode=lambda word, add_special_tokens=False: [int(word[1:])])
    logits = torch.arange(131, dtype=torch.float32).repeat(5, 1)
    logits[:, 130] = 1e6  # Not in the original text.
    vocab = {i: f"w{i}" for i in range(131)}
    text = " ".join(f"w{i}" for i in range(130))
    tokens, values = extractors.get_text_valid_disassemble_tokens_values(
        text, tokenizer, logits, vocab, data, set(), None, model)
    assert len(tokens) == 128
    assert set(tokens) == {f"w{i}" for i in range(2, 130)}
    assert all(value == int(token[1:]) * 500 for token, value in zip(tokens, values))


def check_image_branch_still_uses_vocabulary_top_30(extractors, arguments):
    torch = extractors.torch
    data, model = arguments
    logits = torch.arange(40, dtype=torch.float32).repeat(5, 1)
    vocab = {i: f"w{i}" for i in range(40)}
    tokens, values = extractors.get_img_valid_disassemble_tokens_values(
        None, logits, vocab, data, set(), None, model)
    assert set(tokens) == {f"w{i}" for i in range(10, 40)}
    assert all(value == int(token[1:]) * 500 for token, value in zip(tokens, values))


def check_adaptive_uses_original_text_for_corpus_and_queries(extractors, arguments):
    torch = extractors.torch
    data, model = arguments
    data.sparse_value_mean = False
    tokenizer = SimpleNamespace(
        encode=lambda word, add_special_tokens=False: [{"cat": 1, "dog": 2}[word]],
        get_vocab=lambda: {"cat" if i == 1 else "dog" if i == 2 else f"w{i}": i for i in range(40)},
    )
    logits = torch.arange(40, dtype=torch.float32).repeat(5, 1)
    logits[:, 39] = 1e6
    source = Path(__file__).resolve().parents[1] / "src" / "adaptive_view_encoding.py"
    node = next(node for node in ast.parse(source.read_text(encoding="utf-8")).body
                if isinstance(node, ast.FunctionDef) and node.name == "uncompressed_sparse_vectors")
    namespace = {"stc": SimpleNamespace(enabled=lambda _: False),
                 "lens": SimpleNamespace(enabled=lambda _: False)}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
    sys.modules["encode"] = extractors
    encoder = SimpleNamespace(family="llava_mistral", processor=SimpleNamespace(tokenizer=tokenizer),
                              data=data, model_args=model)
    records = [{"text": "cat dog"}, {"text": "cat dog"}, {"image": "query.jpg"}]
    vectors = namespace["uncompressed_sparse_vectors"](encoder, records, [logits] * 3)
    assert vectors[0] == vectors[1] == {"cat": 500, "dog": 1000}
    assert "w39" in vectors[2] and len(vectors[2]) == 30
    model.eol_type = "disassembleeol_separate"
    baseline = namespace["uncompressed_sparse_vectors"](encoder, records[:1], [logits])[0]
    assert "w39" in baseline and len(baseline) == 30


CASES = {
    "original_tokens": check_original_tokens_only_and_five_view_sum,
    "original_limit": check_original_text_limit_is_128_per_view,
    "image_top_30": check_image_branch_still_uses_vocabulary_top_30,
    "adaptive_original_text": check_adaptive_uses_original_text_for_corpus_and_queries,
}


@pytest.mark.parametrize("case", CASES)
def test_sparse_extraction(case):
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch not installed")
    result = subprocess.run(
        [sys.executable, "-B", str(Path(__file__).resolve()), case],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


if __name__ == "__main__":
    CASES[sys.argv[1]](extractors(), arguments())
