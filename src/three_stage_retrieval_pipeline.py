"""A standalone three-stage retrieval and reranking pipeline.

The existing search and reranking programs are intentionally left untouched.
This file follows the loading flow used by ``search.py`` / ``rerank.py`` and
calls the existing :class:`reranker.Reranker` for the final stage.

Pipeline
--------
1. Stage 1 is a coarse candidate generator.  It can be one of:

   * ``clip`` or ``blip``: direct HuggingFace image/text feature retrieval;
   * ``sparse_manual``: online query encoding with
     ``prompteol + separate + sparse_manual`` against a persisted Lucene
     corpus index;
   * ``sparse_concat``: online query encoding with
     ``disassembleeol_separate + concat`` against a persisted Lucene corpus
     index.

   A precomputed TREC/JSON run can also be supplied for any stage.  This is
   useful when a baseline model is maintained by a separate script.

2. Stage 2 is always the current-program configuration
   ``disassembleeol_separate + concat``.  Just like ``search.py``, it encodes
   queries online, loads dense ``corpus*.pkl`` shards and a sparse Lucene
   ``index/``, normalizes both scores in the same per-query min-max fashion as
   ``hybrid.py``, and fuses them inside the Stage-1 candidate pool.

3. Stage 3 calls the existing ``Reranker.rerank`` pointwise ranker on the
   Stage-2 candidate list.

``encode.py`` is only expected to persist corpus-side artifacts.  No
``query.pkl`` or ``query.tsv`` is required.  Every intermediate run is still
saved for inspection or resume.
"""

from __future__ import annotations

import argparse
import gc
import json
import pickle
import subprocess
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Optional, Sequence


Run = dict[str, dict[str, float]]

STAGE1_METHODS = ("clip", "blip", "sparse_manual", "sparse_concat")
STAGE2_EOL_TYPE = "disassembleeol_separate"
STAGE2_CALCULATE_TYPE = "concat"


@dataclass
class EncodedQueries:
    """Dense vectors and Lucene topics produced from the same online pass."""

    query_ids: list[str]
    dense_representations: Any
    sparse_topics: dict[str, str]


def _normalise_id(value: Any) -> str:
    """Convert IDs from parquet/pickle/numpy scalar types to stable strings."""

    if value is None:
        return ""
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].lstrip("+-").isdigit():
        return text[:-2]
    return text


def _score(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"无法把检索分数转换为浮点数: {value!r}") from exc


def _plain_scores(value: Any) -> dict[str, float]:
    """Read the run shapes used by both ``hybrid.py`` and TREC/JSON files."""

    if isinstance(value, Mapping) and "docs" in value:
        value = value["docs"]

    if isinstance(value, Mapping):
        return {
            _normalise_id(doc_id): _score(doc_score)
            for doc_id, doc_score in value.items()
            if _normalise_id(doc_id)
        }

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        result: dict[str, float] = {}
        for item in value:
            if isinstance(item, Mapping):
                doc_id = item.get("docid", item.get("doc_id", item.get("id")))
                doc_score = item.get("score", item.get("value", 0.0))
            elif isinstance(item, Sequence) and len(item) >= 2:
                doc_id, doc_score = item[0], item[1]
            else:
                raise ValueError(f"无法解析 run 中的候选项: {item!r}")
            result[_normalise_id(doc_id)] = _score(doc_score)
        return result

    raise ValueError(f"无法解析 run 中的候选分数字典: {value!r}")


def _run_from_json_payload(payload: Any) -> Run:
    if isinstance(payload, Mapping) and "run" in payload:
        payload = payload["run"]

    if isinstance(payload, Mapping):
        return {
            _normalise_id(query_id): _plain_scores(scores)
            for query_id, scores in payload.items()
            if _normalise_id(query_id)
        }

    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        run: Run = {}
        for row in payload:
            if not isinstance(row, Mapping):
                raise ValueError(f"JSON run 列表中的元素必须是对象: {row!r}")
            query_id = _normalise_id(row.get("qid", row.get("query_id", row.get("query-id"))))
            doc_id = _normalise_id(row.get("docid", row.get("doc_id", row.get("corpus_id", row.get("corpus-id")))))
            if not query_id or not doc_id:
                raise ValueError(f"JSON run 记录缺少 qid/docid: {row!r}")
            run.setdefault(query_id, {})[doc_id] = _score(row.get("score", 0.0))
        return run

    raise ValueError("JSON run 必须是 query -> candidates 映射或记录列表。")


def read_run(path: Path) -> Run:
    """Read a TREC run, JSON run, JSONL run, or pickle run."""

    if not path.is_file():
        raise FileNotFoundError(f"找不到 run 文件: {path}")

    suffix = path.suffix.lower()
    if suffix in {".json", ".jsonl"}:
        if suffix == ".json":
            with path.open("r", encoding="utf-8") as handle:
                return _run_from_json_payload(json.load(handle))

        run: Run = {}
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                    line_run = _run_from_json_payload(payload)
                except (json.JSONDecodeError, ValueError) as exc:
                    raise ValueError(f"无法解析 JSONL run 第 {line_number} 行: {path}") from exc
                for query_id, scores in line_run.items():
                    run.setdefault(query_id, {}).update(scores)
        return run

    if suffix in {".pkl", ".pickle"}:
        with path.open("rb") as handle:
            return _run_from_json_payload(pickle.load(handle))

    run: Run = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.strip().split()
            if not fields:
                continue
            if len(fields) < 5:
                raise ValueError(f"TREC run 第 {line_number} 行字段不足: {line!r}")
            query_id = _normalise_id(fields[0])
            doc_id = _normalise_id(fields[2])
            if query_id and doc_id:
                run.setdefault(query_id, {})[doc_id] = _score(fields[4])
    return run


def write_trec_run(run: Run, path: Path, tag: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for query_id, scores in run.items():
            ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
            for rank, (doc_id, value) in enumerate(ranked, start=1):
                handle.write(f"{query_id} Q0 {doc_id} {rank} {value:.12g} {tag}\n")


def write_json_run(run: Run, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(run, handle, ensure_ascii=False, indent=2)


def _top_ids(scores: Mapping[str, float], depth: int) -> list[str]:
    if depth <= 0:
        return []
    return [doc_id for doc_id, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:depth]]


def _minmax(scores: Mapping[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    minimum = min(scores.values())
    maximum = max(scores.values())
    denominator = max(maximum - minimum, 1e-9)
    return {doc_id: (value - minimum) / denominator for doc_id, value in scores.items()}


def fuse_restricted_runs(
    candidate_run: Run,
    dense_run: Run,
    sparse_run: Run,
    alpha: float,
    candidate_depth: int,
) -> Run:
    """Fuse Stage-2 runs only over the candidate pool from Stage 1."""

    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"hybrid alpha 必须位于 [0, 1]，当前为 {alpha}。")

    result: Run = {}
    for query_id, stage1_scores in candidate_run.items():
        candidate_ids = _top_ids(stage1_scores, candidate_depth)
        dense_scores = _minmax(dense_run.get(query_id, {}))
        sparse_scores = _minmax(sparse_run.get(query_id, {}))
        fused: dict[str, float] = {}
        for doc_id in candidate_ids:
            fused[doc_id] = (
                alpha * dense_scores.get(doc_id, 0.0)
                + (1.0 - alpha) * sparse_scores.get(doc_id, 0.0)
            )
        result[query_id] = dict(sorted(fused.items(), key=lambda item: item[1], reverse=True))
    return result


def _to_numpy(value: Any):
    import numpy as np

    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _load_pickle_pair(path: Path) -> tuple[Any, list[Any]]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, (tuple, list)) or len(payload) != 2:
        raise ValueError(f"密集表示文件必须保存为 (representations, lookup): {path}")
    reps, lookup = payload
    return reps, list(lookup)


def _find_corpus_pickles(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    matches = sorted(path.glob("corpus*.pkl"))
    if not matches:
        raise FileNotFoundError(f"在 {path} 下找不到 corpus*.pkl。")
    return matches


def load_dense_corpus(path: Path) -> tuple[Any, list[str]]:
    import numpy as np

    rep_parts = []
    lookup: list[str] = []
    for corpus_path in _find_corpus_pickles(path):
        reps, part_lookup = _load_pickle_pair(corpus_path)
        array = _to_numpy(reps).astype("float32", copy=False)
        if array.ndim != 2:
            raise ValueError(f"密集表示必须是二维数组: {corpus_path}, shape={array.shape}")
        rep_parts.append(array)
        lookup.extend(_normalise_id(item) for item in part_lookup)
    if not rep_parts:
        raise ValueError(f"没有读取到任何密集表示: {path}")
    representations = np.concatenate(rep_parts, axis=0)
    if len(lookup) != len(representations):
        raise ValueError(
            f"密集表示数量和 lookup 数量不一致: {len(representations)} != {len(lookup)}"
        )
    return representations, lookup


def _l2_normalize(array: Any):
    import numpy as np

    norms = np.linalg.norm(array, axis=-1, keepdims=True)
    return array / np.maximum(norms, 1e-12)


def retrieve_dense_from_corpus(
    query_representations: Any,
    query_lookup: Sequence[str],
    corpus_path: Path,
    qids: Iterable[str],
    depth: int,
    candidate_filter: Optional[Mapping[str, set[str]]] = None,
    normalize: bool = True,
) -> Run:
    import numpy as np

    query_reps = _to_numpy(query_representations).astype("float32", copy=False)
    query_lookup = [_normalise_id(item) for item in query_lookup]
    corpus_reps, corpus_lookup = load_dense_corpus(corpus_path)
    if query_reps.ndim != 2 or corpus_reps.ndim != 2:
        raise ValueError("在线 query 表征和 corpus*.pkl 表征必须是二维数组。")
    if query_reps.shape[1] != corpus_reps.shape[1]:
        raise ValueError(
            f"query/corpus 表征维度不一致: {query_reps.shape[1]} != {corpus_reps.shape[1]}"
        )
    if normalize:
        query_reps = _l2_normalize(query_reps)
        corpus_reps = _l2_normalize(corpus_reps)

    query_to_row = {query_id: index for index, query_id in enumerate(query_lookup)}
    corpus_to_row: dict[str, int] = {}
    for index, corpus_id in enumerate(corpus_lookup):
        corpus_to_row.setdefault(corpus_id, index)

    result: Run = {}
    for query_id in qids:
        query_id = _normalise_id(query_id)
        row_index = query_to_row.get(query_id)
        if row_index is None:
            continue
        allowed = candidate_filter.get(query_id) if candidate_filter is not None else None
        if allowed is None:
            indices = np.arange(len(corpus_lookup), dtype=np.int64)
        else:
            indices = np.asarray(
                [corpus_to_row[doc_id] for doc_id in allowed if doc_id in corpus_to_row],
                dtype=np.int64,
            )
        if len(indices) == 0:
            result[query_id] = {}
            continue
        scores = corpus_reps[indices] @ query_reps[row_index]
        order = np.argsort(-scores, kind="stable")[:depth]
        result[query_id] = {
            corpus_lookup[int(indices[position])]: float(scores[position])
            for position in order
        }
    return result


def _resolve_lucene_index(path: Path) -> Path:
    if (path / "index").is_dir():
        return path / "index"
    return path


def retrieve_sparse_topics_from_lucene(
    index_path: Path,
    topics_by_id: Mapping[str, str],
    qids: Iterable[str],
    depth: int,
    threads: int,
) -> Run:
    try:
        from pyserini.analysis import JWhiteSpaceAnalyzer
        from pyserini.search.lucene import LuceneImpactSearcher
    except ImportError as exc:
        raise RuntimeError(
            "稀疏阶段需要 pyserini。请使用当前项目用于 sparse search 的环境运行。"
        ) from exc

    requested_ids = [
        _normalise_id(qid)
        for qid in qids
        if _normalise_id(qid) in topics_by_id
    ]
    result: Run = {query_id: {} for query_id in requested_ids}
    selected_ids = [query_id for query_id in requested_ids if topics_by_id[query_id].strip()]
    if not selected_ids:
        return result

    index = _resolve_lucene_index(index_path)
    searcher = LuceneImpactSearcher(str(index), None)
    analyzer = JWhiteSpaceAnalyzer()
    searcher.set_analyzer(analyzer)
    try:
        topics = [topics_by_id[query_id] for query_id in selected_ids]
        raw_results = searcher.batch_search(
            topics,
            selected_ids,
            depth,
            threads=max(int(threads), 1),
        )
        for query_id in selected_ids:
            hits = raw_results.get(query_id, [])
            result[query_id] = {
                _normalise_id(hit.docid): float(hit.score)
                for hit in hits
            }
        return result
    finally:
        close_fn: Any = getattr(searcher, "close", None)
        if callable(close_fn):
            close_fn()
        analyzer_close: Any = getattr(analyzer, "close", None)
        if callable(analyzer_close):
            analyzer_close()


def _retrieval_dtype(name: str):
    import torch

    if name == "float32":
        return torch.float32
    if name == "bfloat16":
        return torch.bfloat16
    return torch.float16


def _load_retrieval_runtime(args: argparse.Namespace) -> Any:
    """Load the online query encoder in the same way as search.py/rerank.py."""

    import torch
    from transformers import (
        AutoModel,
        AutoProcessor,
        LlavaForConditionalGeneration,
        LlavaNextForConditionalGeneration,
        LlavaNextProcessor,
        LlavaProcessor,
        Qwen2_5_VLForConditionalGeneration,
        Qwen2_5_VLProcessor,
    )

    src_dir = Path(__file__).resolve().parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    from encode import get_filtered_ids
    import model as model_module

    model_path = args.retrieval_model_name_or_path
    if not model_path:
        raise ValueError("在线大模型查询编码需要 --retrieval-model-name-or-path。")

    dtype = _retrieval_dtype(args.retrieval_dtype)
    device = torch.device(
        args.retrieval_device
        if args.retrieval_device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    load_kwargs: dict[str, Any] = {"torch_dtype": dtype, "low_cpu_mem_usage": True}
    if args.retrieval_device_map.lower() != "none":
        load_kwargs["device_map"] = args.retrieval_device_map

    def load(model_class: Any):
        try:
            return model_class.from_pretrained(model_path, **load_kwargs)
        except TypeError:
            fallback = dict(load_kwargs)
            fallback.pop("torch_dtype", None)
            fallback["dtype"] = dtype
            return model_class.from_pretrained(model_path, **fallback)

    if "llava-hf-llava-1.5-7b-hf" in model_path:
        encoder = load(LlavaForConditionalGeneration)
        processor = LlavaProcessor.from_pretrained(model_path)
    elif "Qwen2.5-VL-7B-Instruct" in model_path or "Qwen2.5-VL-3B-Instruct" in model_path:
        encoder = load(Qwen2_5_VLForConditionalGeneration)
        processor = Qwen2_5_VLProcessor.from_pretrained(model_path)
    elif "InternVL2_5-8B" in model_path or "InternVL2_5-4B" in model_path:
        encoder = AutoModel.from_pretrained(
            model_path,
            **load_kwargs,
            trust_remote_code=True,
        )
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    else:
        encoder = load(LlavaNextForConditionalGeneration)
        processor = LlavaNextProcessor.from_pretrained(model_path)
        if "royokong-e5-v" in model_path:
            setattr(processor, "patch_size", 14)

    if args.retrieval_device_map.lower() == "none":
        encoder = encoder.to(device)
    if args.retrieval_lora_model_path:
        from peft import PeftModel

        encoder = PeftModel.from_pretrained(encoder, args.retrieval_lora_model_path)
        encoder = encoder.merge_and_unload()

    tokenizer = getattr(processor, "tokenizer", processor)
    if args.retrieval_reps_loc == "after_pad" and hasattr(tokenizer, "padding_side"):
        tokenizer.padding_side = "left"
        tokenizer.padding = True
    raw_vocab = tokenizer.get_vocab()
    vocab_dict = {token_id: token for token, token_id in raw_vocab.items()}
    filtered_ids = get_filtered_ids(tokenizer)
    original_get_rank = None
    if not model_module.dist.is_initialized():
        original_get_rank = model_module.dist.get_rank
        model_module.dist.get_rank = lambda: 0
    model = model_module.MLLMRetrievalModel(encoder, vocab_dict=raw_vocab).eval()
    return SimpleNamespace(
        model=model,
        encoder=encoder,
        processor=processor,
        tokenizer=tokenizer,
        image_tokenizer=processor if "InternVL2_5" in model_path else tokenizer,
        vocab_dict=vocab_dict,
        filtered_ids=filtered_ids,
        model_path=model_path,
        device=device,
        dtype=dtype,
        query_cache={},
        model_module=model_module,
        original_get_rank=original_get_rank,
    )


def _release_retrieval_runtime(runtime: Any) -> None:
    if runtime is None:
        return
    import torch

    if getattr(runtime, "original_get_rank", None) is not None:
        runtime.model_module.dist.get_rank = runtime.original_get_rank
    for name in ("model", "encoder", "processor", "tokenizer", "image_tokenizer"):
        if hasattr(runtime, name):
            setattr(runtime, name, None)
    if hasattr(runtime, "query_cache"):
        runtime.query_cache.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _retrieval_namespaces(
    args: argparse.Namespace,
    *,
    eol_type: str,
    calculate_type: str,
    sparse_manual: bool,
) -> tuple[Any, Any]:
    model_args = SimpleNamespace(
        model_name_or_path=args.retrieval_model_name_or_path,
        eol_type=eol_type,
        calculate_type=calculate_type,
        use_output_embedding_cluster=False,
    )
    data_args = SimpleNamespace(
        dataset_name=args.dataset_name,
        per_device_batch_size=args.retrieval_batch_size,
        reps_loc=args.retrieval_reps_loc,
        sparse_manual=sparse_manual,
        sparse_length=args.sparse_length,
        text_sparse_length=args.text_sparse_length,
        image_sparse_length=args.image_sparse_length,
        num_expended_tokens=args.num_expended_tokens,
        is_filtered=args.is_filtered,
        sparse_value_type=args.sparse_value_type,
        sparse_lower_or_upper=args.sparse_lower_or_upper,
        prompt_type=args.prompt_type,
        random_view_seed=args.random_view_seed,
        prompt_generation=False,
        prompt_generation_model="llama",
        sparse_value_mean=args.sparse_value_mean,
        sparse_logits_multiply_view_count=args.sparse_logits_multiply_view_count,
        sparse_type="single",
        tbpr_type=args.tbpr_type,
        cir_type=args.cir_type,
        print_sparse=args.print_sparse,
        composed_top_type="text",
        composed_top_k=3,
    )
    return model_args, data_args


def _query_records(args: argparse.Namespace, dataset: Any) -> list[dict[str, Any]]:
    """Return the query side only; corpus representations come from encode.py."""

    data_root = Path(args.data_root).expanduser().resolve()
    records: list[dict[str, Any]] = []
    if args.task_type == "t2it":
        for query_id in dataset.query_id_list:
            records.append({"id": _normalise_id(query_id), "text": dataset.id2query[query_id]})
        return records

    if args.task_type == "it2t":
        for query_id in dataset.query_id_list:
            query = dataset.id2query[query_id]
            records.append(
                {
                    "id": _normalise_id(query_id),
                    "text": query["text"],
                    "image": query["image"],
                }
            )
        return records

    if args.task_type == "cir":
        for composed_id in dataset.composed_id_list:
            separator = composed_id.index("_")
            image_id = composed_id[:separator]
            text_id = composed_id[separator + 1:]
            dress_type = dataset.dress_type_dict[image_id]
            if dress_type == "toptee":
                dress_type = "shirt"
            records.append(
                {
                    "id": _normalise_id(composed_id),
                    "text": dataset.get_text(text_id),
                    "image": data_root / "data" / dataset.data_name / "images" / "images" / dataset.img_dict[image_id],
                    "dress_type": dress_type,
                }
            )
        return records

    if args.query_type == "text" or args.task_type == "tbpr":
        for text_id in dataset.text_id_list:
            records.append({"id": _normalise_id(text_id), "text": dataset.text_dict[text_id]})
        return records

    for image_id in dataset.img_id_list:
        companion_text = ""
        text_ids = getattr(dataset, "img2text", {}).get(image_id, [])
        if text_ids:
            companion_text = dataset.text_dict[text_ids[0]]
        records.append(
            {
                "id": _normalise_id(image_id),
                "text": companion_text,
                "image": _standard_image_path(dataset, image_id, data_root),
            }
        )
    return records


def _plain_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return " ".join(str(item) for item in value)
    return str(value)


def _base_image_prompt(runtime: Any, args: argparse.Namespace, task_type: str) -> Any:
    import template as templates

    model_path = runtime.model_path
    if task_type == "it2t":
        if "llava-hf-llava-1.5-7b-hf" in model_path:
            prompt = templates.img_prompt_no_special_llava_v1_5
        elif "Qwen2.5-VL" in model_path:
            prompt = templates.qwen2_5_img_prompt
        elif "InternVL2_5" in model_path:
            prompt = templates.img_prompt_intern_vl_v2_5
        elif "llava-hf-llava-v1.6-mistral-7b-hf" in model_path:
            prompt = templates.mistral_it2t_query_prompt
        elif "llava-hf-llava-v1.6-vicuna" in model_path:
            prompt = templates.vicuna_img_prompt
        else:
            prompt = templates.it2t_query_prompt
    elif "llava-hf-llava-1.5-7b-hf" in model_path:
        prompt = templates.img_prompt_no_special_llava_v1_5
    elif "Qwen2.5-VL" in model_path:
        prompt = runtime.processor.apply_chat_template(
            templates.img_prompt_qwen_v2_5, tokenize=False, add_generation_prompt=True
        )
    elif "InternVL2_5" in model_path:
        prompt = runtime.processor.apply_chat_template(
            templates.img_prompt_intern_vl_v2_5, tokenize=False, add_generation_prompt=True
        )
    elif "llava-hf-llava-v1.6-mistral-7b-hf" in model_path:
        prompt = templates.mistral_img_prompt
    elif "llava-hf-llava-v1.6-vicuna" in model_path:
        prompt = templates.vicuna_img_prompt
    else:
        prompt = templates.img_prompt
    return prompt


def _concat_aspect_prompts(templates: Any, prompt_type: str) -> Sequence[str]:
    names = {
        "prompt_1": "retrieval_disassemble_image_prompts_1_for_concat",
        "prompt_2": "retrieval_disassemble_image_prompts_2_for_concat",
        "prompt_3": "retrieval_disassemble_image_prompts_3_for_concat",
        "prompt_4": "retrieval_disassemble_image_prompts_4_for_concat",
        "prompt_5": "retrieval_disassemble_image_prompts_for_concat",
        "prompt_6": "retrieval_disassemble_image_prompts_6_for_concat",
        "prompt_7": "retrieval_disassemble_image_prompts_7_for_concat",
    }
    if prompt_type not in names:
        raise ValueError(f"当前在线 concat 编码不支持 prompt_type={prompt_type!r}。")
    return getattr(templates, names[prompt_type])


def _concat_image_prompt(runtime: Any, args: argparse.Namespace, task_type: str) -> str:
    import template as templates

    model_path = runtime.model_path
    if task_type == "it2t":
        if "llava-hf-llava-v1.6-mistral-7b-hf" in model_path:
            prefix = templates.llava_mistral_template_fusion_prefix
            element = templates.llava_mistral_template_content_element
            original_prompt = templates.fusion_prompt_for_concat
        else:
            prefix = templates.llama3_template_fusion_prefix
            element = templates.llama3_template_content_element
            original_prompt = templates.img_prompt_for_concat
        aspects = (
            templates.retrieval_disassemble_query_prompts_llava_it2t_retrieval_for_concat
            if args.dataset_name == "llava"
            else templates.retrieval_disassemble_query_prompts_it2t_retrieval_for_concat
        )
    else:
        if "llava-hf-llava-v1.6-mistral-7b-hf" in model_path:
            prefix = templates.llava_mistral_template_image_prefix
            element = templates.llava_mistral_template_content_element
        elif "llava-hf-llava-v1.6-vicuna" in model_path:
            prefix = templates.llava_vicuna_template_image_prefix
            element = templates.llava_vicuna_template_content_element
        elif "llava-hf-llava-v1.6-34b-hf" in model_path:
            prefix = templates.llava_34b_template_image_prefix
            element = templates.llava_34b_template_content_element
        elif "Qwen2.5-VL-7B-Instruct" in model_path:
            prefix = templates.qwen2_5_template_image_prefix
            element = templates.qwen2_5_template_content_element
        else:
            prefix = templates.llama3_template_image_prefix
            element = templates.llama3_template_content_element
        original_prompt = templates.img_prompt_for_concat
        aspects = _concat_aspect_prompts(templates, args.prompt_type)

    prompt = prefix + element.format(original_prompt)
    for aspect in aspects:
        prompt += element.format(aspect)
    return prompt


def _prepare_image_inputs(
    runtime: Any,
    args: argparse.Namespace,
    records: Sequence[Mapping[str, Any]],
    *,
    calculate_type: str,
) -> Any:
    raw_images = [_image_from_value(record["image"]) for record in records]
    if "InternVL2_5" in runtime.model_path and calculate_type == "separate":
        import torch
        from utils import build_transform, dynamic_preprocess

        transform = build_transform(input_size=448)
        pixel_batches = []
        for image in raw_images:
            patches = dynamic_preprocess(
                image, image_size=448, use_thumbnail=True, max_num=12
            )
            pixel_batches.append(
                torch.stack([transform(patch) for patch in patches])
                .to(dtype=runtime.dtype, device=runtime.device)
            )
        return pixel_batches
    texts = [_plain_text(record.get("text", "")) for record in records]
    if calculate_type == "concat":
        prompt = _concat_image_prompt(runtime, args, args.task_type)
    else:
        prompt = _base_image_prompt(runtime, args, args.task_type)
    prompt_texts = [str(prompt).replace("<sent>", text) for text in texts]
    inputs = runtime.processor(
        images=raw_images,
        text=prompt_texts,
        return_tensors="pt",
        padding=True,
    )
    if hasattr(inputs, "to"):
        return inputs.to(runtime.device)
    return {
        key: value.to(runtime.device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }


def _encode_query_batch(
    runtime: Any,
    args: argparse.Namespace,
    records: Sequence[Mapping[str, Any]],
    model_args: Any,
    data_args: Any,
) -> tuple[Any, Any]:
    texts = [_plain_text(record.get("text", "")) for record in records]
    calculate_type = model_args.calculate_type
    if args.task_type == "cir":
        raw_images = [_image_from_value(record["image"]) for record in records]
        dress_types = [record["dress_type"] for record in records]
        method = (
            runtime.model.encode_data_for_cir
            if calculate_type == "separate"
            else runtime.model.encode_data_concat_for_cir
        )
        return method(
            texts,
            raw_images,
            dress_types,
            "composed",
            runtime.processor,
            runtime.device,
            model_args,
            data_args,
        )

    if args.task_type == "t2it":
        method = (
            runtime.model.encode_data_for_t2it
            if calculate_type == "separate"
            else runtime.model.encode_data_concat_for_t2it
        )
        return method(texts, "query", runtime.processor, runtime.device, model_args, data_args)

    if args.task_type == "tbpr":
        method = (
            runtime.model.encode_data_for_tbpr
            if calculate_type == "separate"
            else runtime.model.encode_data_concat_for_tbpr
        )
        return method(texts, "text", runtime.processor, runtime.device, model_args, data_args)

    if args.task_type == "it2t" or args.query_type == "image":
        image_inputs = _prepare_image_inputs(
            runtime, args, records, calculate_type=calculate_type
        )
        if args.task_type == "it2t":
            method = (
                runtime.model.encode_data_for_it2t
                if calculate_type == "separate"
                else runtime.model.encode_data_concat_for_it2t
            )
            return method(
                image_inputs, "query", runtime.processor, runtime.device, model_args, data_args
            )
        method = (
            runtime.model.encode_data
            if calculate_type == "separate"
            else runtime.model.encode_data_concat
        )
        return method(
            image_inputs, "image", runtime.processor, runtime.device, model_args, data_args
        )

    method = (
        runtime.model.encode_data
        if calculate_type == "separate"
        else runtime.model.encode_data_concat
    )
    return method(texts, "text", runtime.processor, runtime.device, model_args, data_args)


def _merge_sparse_values(
    tokens: Sequence[str], values: Sequence[Any], value_type: str
) -> dict[str, int]:
    vector: dict[str, int] = {}
    for token, value in zip(tokens, values):
        weight = int(value)
        if not token or weight <= 0:
            continue
        if token not in vector:
            vector[token] = weight
        elif value_type == "replace":
            vector[token] = weight
        elif value_type == "sum":
            vector[token] += weight
        else:
            vector[token] = max(vector[token], weight)
    return vector


def _vector_to_topic(vector: Mapping[str, int]) -> str:
    return " ".join(
        " ".join([str(token)] * int(weight))
        for token, weight in vector.items()
        if int(weight) > 0
    )


def _batch_sparse_topics(
    runtime: Any,
    records: Sequence[Mapping[str, Any]],
    logits: Any,
    model_args: Any,
    data_args: Any,
    args: argparse.Namespace,
) -> dict[str, str]:
    import torch
    from encode import (
        get_img_valid_disassemble_tokens_values,
        get_img_valid_tokens_values,
        get_text_valid_disassemble_tokens_values,
        get_text_valid_tokens_values,
    )

    if isinstance(logits, list):
        logits = torch.cat(list(logits), dim=0)
    if not torch.is_tensor(logits) or logits.ndim != 2:
        raise ValueError(f"在线查询 logits 必须是二维张量，实际为 {type(logits).__name__}。")

    batch_size = len(records)
    if logits.shape[0] % batch_size != 0:
        raise ValueError(
            f"query logits 数量无法按 batch 分组: {logits.shape[0]} vs batch={batch_size}"
        )
    view_count = logits.shape[0] // batch_size
    if model_args.eol_type == "prompteol" and view_count != 1:
        raise ValueError(
            f"prompteol 查询每条应产生一个 logits，实际每条产生 {view_count} 个。"
        )
    text_query = args.task_type in {"t2it", "tbpr"} or (
        args.task_type == "ir" and args.query_type == "text"
    )
    topics: dict[str, str] = {}
    for index, record in enumerate(records):
        text = _plain_text(record.get("text", ""))
        if model_args.eol_type == "prompteol":
            current_logits = logits[index]
            if text_query:
                tokens, values = get_text_valid_tokens_values(
                    text,
                    runtime.tokenizer,
                    current_logits,
                    runtime.vocab_dict,
                    data_args,
                    runtime.filtered_ids,
                    model_args,
                )
            else:
                tokens, values = get_img_valid_tokens_values(
                    runtime.image_tokenizer,
                    current_logits,
                    runtime.vocab_dict,
                    data_args,
                    runtime.filtered_ids,
                    model_args,
                )
        else:
            current_logits = logits[index * view_count:(index + 1) * view_count]
            if text_query:
                tokens, values = get_text_valid_disassemble_tokens_values(
                    text,
                    runtime.tokenizer,
                    current_logits,
                    runtime.vocab_dict,
                    data_args,
                    runtime.filtered_ids,
                    None,
                    model_args,
                )
            else:
                tokens, values = get_img_valid_disassemble_tokens_values(
                    runtime.image_tokenizer,
                    current_logits,
                    runtime.vocab_dict,
                    data_args,
                    runtime.filtered_ids,
                    None,
                    model_args,
                )
        vector = _merge_sparse_values(tokens, values, data_args.sparse_value_type)
        if data_args.sparse_value_mean and view_count > 1:
            vector = {token: weight // view_count for token, weight in vector.items()}
        topics[_normalise_id(record["id"])] = _vector_to_topic(vector)
    return topics


def _dense_batch_to_numpy(representations: Any, expected_rows: int):
    import numpy as np
    import torch

    if isinstance(representations, list):
        rows = []
        for item in representations:
            if isinstance(item, list):
                item = torch.stack(item, dim=0).mean(dim=0)
            if torch.is_tensor(item) and item.ndim == 2:
                item = item.mean(dim=0)
            rows.append(item)
        representations = torch.stack(rows, dim=0)
    array = _to_numpy(representations).astype("float32", copy=False)
    if array.ndim == 3:
        array = array.mean(axis=1)
    if array.ndim != 2 or array.shape[0] != expected_rows:
        raise ValueError(
            f"在线 dense query 表征形状错误: shape={array.shape}, expected_rows={expected_rows}"
        )
    return np.asarray(array)


def encode_queries_online(
    args: argparse.Namespace,
    runtime: Any,
    *,
    eol_type: str,
    calculate_type: str,
    sparse_manual: bool,
) -> EncodedQueries:
    import numpy as np
    import torch

    cache_key = (
        args.dataset_name,
        args.dataset_split,
        args.task_type,
        args.query_type,
        eol_type,
        calculate_type,
        sparse_manual,
        args.prompt_type,
        args.sparse_length,
        args.text_sparse_length,
        args.image_sparse_length,
        args.sparse_value_type,
        args.sparse_lower_or_upper,
        args.sparse_value_mean,
        args.num_expended_tokens,
    )
    cached = runtime.query_cache.get(cache_key)
    if cached is not None:
        return cached

    dataset = build_dataset(args, runtime.processor)
    records = _query_records(args, dataset)
    model_args, data_args = _retrieval_namespaces(
        args,
        eol_type=eol_type,
        calculate_type=calculate_type,
        sparse_manual=sparse_manual,
    )
    query_ids: list[str] = []
    dense_parts = []
    topics: dict[str, str] = {}
    for start in range(0, len(records), args.retrieval_batch_size):
        batch = records[start:start + args.retrieval_batch_size]
        data_args.per_device_batch_size = len(batch)
        amp = (
            torch.autocast(device_type="cuda", dtype=runtime.dtype)
            if runtime.device.type == "cuda" and runtime.dtype != torch.float32
            else nullcontext()
        )
        with torch.inference_mode(), amp:
            logits, dense_representations = _encode_query_batch(
                runtime, args, batch, model_args, data_args
            )
        query_ids.extend(_normalise_id(record["id"]) for record in batch)
        dense_parts.append(_dense_batch_to_numpy(dense_representations, len(batch)))
        topics.update(
            _batch_sparse_topics(runtime, batch, logits, model_args, data_args, args)
        )
    if not dense_parts:
        raise ValueError("当前数据集没有可编码的查询。")
    encoded = EncodedQueries(
        query_ids=query_ids,
        dense_representations=np.concatenate(dense_parts, axis=0),
        sparse_topics=topics,
    )
    runtime.query_cache[cache_key] = encoded
    return encoded


def _patch_dataset_paths(data_root: Path) -> Any:
    """Point dataset.py's imported path constants at the requested data root."""

    src_dir = Path(__file__).resolve().parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    import dataset as dataset_module

    prefix = str(data_root.expanduser().resolve()).rstrip("/\\") + "/"
    dataset_module.dataset_path_prefix = prefix
    directory_names = {
        "coco_file_path": "coco",
        "flickr_file_path": "flickr",
        "fashion_iq_file_path": "fashion-iq",
        "cuhk_pedes_file_path": "CUHK-PEDES",
        "icfg_pedes_flie_path": "ICFG-PEDES",
        "rstpreid_file_path": "RSTPReid",
        "webqa_file_path": "WebQA",
        "remuq_file_path": "UMRB-ReMuQ",
        "llava_file_path": "UMRB-LLaVA",
        "edis_file_path": "EDIS",
    }
    for variable, directory in directory_names.items():
        setattr(dataset_module, variable, prefix + "data/" + directory + "/")
    return dataset_module


def build_dataset(args: argparse.Namespace, processor: Any = None) -> Any:
    dataset_module = _patch_dataset_paths(Path(args.data_root))
    split = args.dataset_split
    task_type = args.task_type
    if task_type == "cir":
        return dataset_module.ComposedTextImageRetrievalDataset(
            args.dataset_name, processor, split, "composed"
        )
    if task_type == "tbpr":
        return dataset_module.TextPersonRetrievalDataset(
            args.dataset_name, processor, split, "full"
        )
    if task_type == "t2it":
        return dataset_module.Text2ImagetextRetrievalDataset(
            args.dataset_name, processor, split, "query"
        )
    if task_type == "it2t":
        return dataset_module.Imagetext2TextRetrievalDataset(
            args.dataset_name, processor, split, "query"
        )
    mode = "full" if args.query_type == "text" else "single"
    if task_type not in {"ir", "tbpr"}:
        raise ValueError(f"不支持的 task_type: {task_type}")
    if task_type == "ir":
        return dataset_module.CrossModalRetrievalDataset(
            args.dataset_name, processor, split, mode
        )
    return dataset_module.TextPersonRetrievalDataset(
        args.dataset_name, processor, split, mode
    )


def _image_from_value(value: Any):
    from PIL import Image

    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, Mapping):
        if "bytes" in value:
            return Image.open(BytesIO(bytes(value["bytes"]))).convert("RGB")
        for key in ("data", "contents", "content"):
            if key in value and value[key] is not None:
                return Image.open(BytesIO(bytes(value[key]))).convert("RGB")
    if isinstance(value, (bytes, bytearray, memoryview)):
        return Image.open(BytesIO(bytes(value))).convert("RGB")
    return Image.open(Path(value)).convert("RGB")


def _standard_image_path(dataset: Any, image_id: str, data_root: Path) -> Path:
    image_id = _normalise_id(image_id)
    image_name = dataset.img_dict[image_id]
    if dataset.data_name == "coco":
        return data_root / "data" / dataset.data_name / dataset.img2filepath[image_id] / image_name
    if dataset.data_name == "flickr":
        return data_root / "data" / dataset.data_name / "flickr30k-images" / image_name
    if dataset.data_name in {"CUHK-PEDES", "ICFG-PEDES", "RSTPReid"}:
        return data_root / "data" / dataset.data_name / "imgs" / image_name
    if dataset.data_name == "fashion-iq":
        return data_root / "data" / dataset.data_name / "images" / "images" / image_name
    raise ValueError(f"无法推断数据集 {dataset.data_name} 的图像路径。")


def _direct_stage1_items(args: argparse.Namespace, dataset: Any) -> tuple[list[tuple[str, str]], list[tuple[str, Any]]]:
    """Return (query text/image items, corpus text/image items) for CLIP/BLIP."""

    data_root = Path(args.data_root)
    query_items: list[tuple[str, Any]] = []
    corpus_items: list[tuple[str, Any]] = []

    if args.task_type == "cir":
        raise ValueError("CLIP/BLIP 第一阶段暂不直接支持 CIR；请提供 --stage1-run。")

    if args.task_type == "t2it":
        query_items = [(_normalise_id(qid), dataset.id2query[qid]) for qid in dataset.query_id_list]
        corpus_items = [
            (
                _normalise_id(cid),
                _image_from_value(dataset.id2candidate[cid]["image"]),
            )
            for cid in dataset.candidate_id_list
        ]
        return query_items, corpus_items

    if args.task_type == "it2t":
        query_items = [
            (_normalise_id(qid), _image_from_value(dataset.id2query[qid]["image"]))
            for qid in dataset.query_id_list
        ]
        corpus_items = [(_normalise_id(cid), dataset.id2candidate[cid]) for cid in dataset.candidate_id_list]
        return query_items, corpus_items

    if args.task_type == "tbpr":
        query_items = [(_normalise_id(tid), dataset.text_dict[tid]) for tid in dataset.text_id_list]
        corpus_items = [
            (
                _normalise_id(iid),
                _image_from_value(_standard_image_path(dataset, iid, data_root)),
            )
            for iid in dataset.img_id_list
        ]
        return query_items, corpus_items

    if args.query_type == "text":
        query_items = [(_normalise_id(tid), dataset.text_dict[tid]) for tid in dataset.text_id_list]
        corpus_items = [
            (
                _normalise_id(iid),
                _image_from_value(_standard_image_path(dataset, iid, data_root)),
            )
            for iid in dataset.img_id_list
        ]
    else:
        query_items = [
            (
                _normalise_id(iid),
                _image_from_value(_standard_image_path(dataset, iid, data_root)),
            )
            for iid in dataset.img_id_list
        ]
        corpus_items = [(_normalise_id(tid), dataset.text_dict[tid]) for tid in dataset.text_id_list]
    return query_items, corpus_items


def _load_feature_model(method: str, model_path: str, device_name: str):
    import torch
    from transformers import AutoProcessor

    device = torch.device(device_name if device_name != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    if method == "clip":
        from transformers import CLIPModel

        model = CLIPModel.from_pretrained(model_path)
        processor = AutoProcessor.from_pretrained(model_path)
    else:
        try:
            from transformers import BlipModel

            model = BlipModel.from_pretrained(model_path)
        except ImportError as exc:
            raise RuntimeError("当前 transformers 版本没有 BlipModel，无法执行 BLIP 第一阶段。") from exc
        processor = AutoProcessor.from_pretrained(model_path)
    model = model.to(device).eval()
    return model, processor, device


def _extract_feature(output: Any, modality: str):
    import torch

    if torch.is_tensor(output):
        feature = output
    else:
        names = (
            ("image_embeds", "pooler_output", "last_hidden_state")
            if modality == "image"
            else ("text_embeds", "pooler_output", "last_hidden_state")
        )
        feature = None
        for name in names:
            candidate = getattr(output, name, None)
            if torch.is_tensor(candidate):
                feature = candidate
                break
        if feature is None and isinstance(output, (tuple, list)):
            feature = next((item for item in output if torch.is_tensor(item)), None)
        if feature is None:
            raise RuntimeError(f"无法从 {type(output).__name__} 中提取 {modality} 表征。")
    if feature.ndim == 3:
        feature = feature.mean(dim=1)
    if feature.ndim != 2:
        raise RuntimeError(f"{modality} 表征应为二维张量，实际 shape={tuple(feature.shape)}")
    return feature


def _encode_feature_batch(model: Any, processor: Any, items: Sequence[Any], modality: str, device: Any):
    import torch

    if modality == "image":
        inputs = processor(images=list(items), return_tensors="pt")
        feature_method = getattr(model, "get_image_features", None)
        if feature_method is None:
            raise RuntimeError("当前 CLIP/BLIP 模型没有 get_image_features，无法执行直接特征检索。")
    else:
        inputs = processor(text=list(items), return_tensors="pt", padding=True, truncation=True)
        feature_method = getattr(model, "get_text_features", None)
        if feature_method is None:
            raise RuntimeError("当前 CLIP/BLIP 模型没有 get_text_features，无法执行直接特征检索。")
    inputs = {key: value.to(device) for key, value in inputs.items() if hasattr(value, "to")}
    with torch.inference_mode():
        output = feature_method(**inputs)
    return _extract_feature(output, modality)


def retrieve_stage1_feature_model(args: argparse.Namespace, method: str) -> Run:
    import numpy as np

    dataset = build_dataset(args)
    query_items, corpus_items = _direct_stage1_items(args, dataset)
    model, processor, device = _load_feature_model(method, args.stage1_model_name_or_path, args.stage1_device)

    # The item type is unambiguous for our dataset adapters: text is str and
    # image is a PIL image.  Keep this explicit for readability and safety.
    if args.task_type == "t2it" or (args.task_type == "ir" and args.query_type == "text") or args.task_type == "tbpr":
        query_modality = "text"
        corpus_modality = "image"
    elif args.task_type == "it2t" or (args.task_type == "ir" and args.query_type == "image"):
        query_modality = "image"
        corpus_modality = "text"
    else:
        raise ValueError("无法根据 task_type/query_type 判断第一阶段的图文方向。")

    query_features = []
    query_ids = []
    for start in range(0, len(query_items), args.stage1_batch_size):
        batch = query_items[start:start + args.stage1_batch_size]
        query_ids.extend(query_id for query_id, _ in batch)
        query_features.append(
            _encode_feature_batch(model, processor, [value for _, value in batch], query_modality, device)
            .detach()
            .cpu()
        )
    corpus_features = []
    corpus_ids = []
    for start in range(0, len(corpus_items), args.stage1_batch_size):
        batch = corpus_items[start:start + args.stage1_batch_size]
        corpus_ids.extend(doc_id for doc_id, _ in batch)
        corpus_features.append(
            _encode_feature_batch(model, processor, [value for _, value in batch], corpus_modality, device)
            .detach()
            .cpu()
        )

    query_matrix = _l2_normalize(torch_cat(query_features))
    corpus_matrix = _l2_normalize(torch_cat(corpus_features))
    if query_matrix.shape[1] != corpus_matrix.shape[1]:
        raise ValueError(
            f"第一阶段模型的图像/文本特征维度不一致: {query_matrix.shape[1]} != {corpus_matrix.shape[1]}"
        )

    scores = query_matrix @ corpus_matrix.T
    result: Run = {}
    top_depth = min(args.stage1_depth, len(corpus_ids))
    for row, query_id in enumerate(query_ids):
        order = np.argsort(-scores[row].numpy(), kind="stable")[:top_depth]
        result[query_id] = {corpus_ids[int(index)]: float(scores[row, int(index)]) for index in order}
    del model, processor, query_matrix, corpus_matrix, scores
    gc.collect()
    return result


def torch_cat(tensors: Sequence[Any]):
    import torch

    if not tensors:
        raise ValueError("没有可拼接的特征批次。")
    return torch.cat(list(tensors), dim=0)


def _run_external_command(command: str, args: argparse.Namespace, output_run: Path) -> None:
    substitutions = {
        "output_run": str(output_run),
        "dataset_name": args.dataset_name,
        "dataset_split": args.dataset_split,
        "data_root": str(Path(args.data_root).resolve()),
    }
    try:
        formatted = command.format(**substitutions)
    except KeyError as exc:
        raise ValueError(
            f"stage1 command 使用了未知占位符 {exc!s}；可用占位符为 "
            "{output_run}, {dataset_name}, {dataset_split}, {data_root}。"
        ) from exc
    print(f"[stage1] execute: {formatted}")
    subprocess.run(formatted, shell=True, check=True, cwd=str(Path(__file__).resolve().parents[1]))


def run_stage1(args: argparse.Namespace, retrieval_runtime: Any = None) -> Run:
    if args.stage1_run:
        run_path = Path(args.stage1_run)
        if args.stage1_command:
            _run_external_command(args.stage1_command, args, run_path)
        return read_run(run_path)

    if args.stage1_command:
        raise ValueError("提供 --stage1-command 时必须同时提供 --stage1-run 作为输出文件。")

    if args.stage1_method in {"clip", "blip"}:
        if not args.stage1_model_name_or_path:
            raise ValueError("CLIP/BLIP 第一阶段需要 --stage1-model-name-or-path。")
        return retrieve_stage1_feature_model(args, args.stage1_method)

    if args.stage1_sparse_index:
        if retrieval_runtime is None:
            raise ValueError("第一阶段稀疏检索需要在线查询编码模型。")
        if args.stage1_method == "sparse_manual":
            encoded = encode_queries_online(
                args,
                retrieval_runtime,
                eol_type="prompteol",
                calculate_type="separate",
                sparse_manual=True,
            )
        else:
            encoded = encode_queries_online(
                args,
                retrieval_runtime,
                eol_type=STAGE2_EOL_TYPE,
                calculate_type=STAGE2_CALCULATE_TYPE,
                sparse_manual=False,
            )
        return retrieve_sparse_topics_from_lucene(
            Path(args.stage1_sparse_index),
            encoded.sparse_topics,
            qids=encoded.query_ids,
            depth=args.stage1_depth,
            threads=args.sparse_threads,
        )
    raise ValueError(
        f"{args.stage1_method} 第一阶段需要 --stage1-run，或提供 --stage1-sparse-index。"
    )


def run_stage2(
    args: argparse.Namespace,
    stage1_run: Run,
    retrieval_runtime: Any = None,
) -> tuple[Run, Run, Run]:
    qids = list(stage1_run)
    candidate_filter = {
        query_id: set(_top_ids(scores, args.stage1_depth))
        for query_id, scores in stage1_run.items()
    }

    if args.stage2_run:
        hybrid_run = read_run(Path(args.stage2_run))
        restricted: Run = {}
        for query_id, candidate_ids in candidate_filter.items():
            scores = hybrid_run.get(query_id, {})
            restricted[query_id] = {
                doc_id: scores[doc_id]
                for doc_id in _top_ids(scores, args.stage2_depth)
                if doc_id in candidate_ids
            }
        return {}, {}, restricted

    required = {
        "stage2_dense_index": args.stage2_dense_index,
        "stage2_sparse_index": args.stage2_sparse_index,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise ValueError(
            "第二阶段必须提供当前 disassembleeol_separate+concat 的 dense/sparse artifacts；"
            f"缺少: {', '.join(missing)}"
        )
    if retrieval_runtime is None:
        raise ValueError("第二阶段需要在线查询编码模型。")

    encoded = encode_queries_online(
        args,
        retrieval_runtime,
        eol_type=STAGE2_EOL_TYPE,
        calculate_type=STAGE2_CALCULATE_TYPE,
        sparse_manual=False,
    )

    dense_run = retrieve_dense_from_corpus(
        encoded.dense_representations,
        encoded.query_ids,
        Path(args.stage2_dense_index),
        qids,
        depth=args.stage2_depth,
        candidate_filter=candidate_filter,
        normalize=not args.no_dense_normalize,
    )
    sparse_run = retrieve_sparse_topics_from_lucene(
        Path(args.stage2_sparse_index),
        encoded.sparse_topics,
        qids,
        depth=max(args.stage2_depth, args.stage1_depth),
        threads=args.sparse_threads,
    )
    hybrid_run = fuse_restricted_runs(
        stage1_run,
        dense_run,
        sparse_run,
        alpha=args.hybrid_alpha,
        candidate_depth=args.stage1_depth,
    )
    return dense_run, sparse_run, hybrid_run


def _ranker_dtype(name: str):
    import torch

    if name == "float32":
        return torch.float32
    if name == "bfloat16":
        return torch.bfloat16
    return torch.float16


def _load_ranker_model(args: argparse.Namespace):
    from transformers import (
        AutoModel,
        AutoProcessor,
        LlavaForConditionalGeneration,
        LlavaNextForConditionalGeneration,
        LlavaNextProcessor,
        LlavaProcessor,
        Qwen2_5_VLForConditionalGeneration,
        Qwen2_5_VLProcessor,
    )

    model_path = args.ranker_model_name_or_path
    dtype = _ranker_dtype(args.ranker_dtype)
    load_kwargs: dict[str, Any] = {"torch_dtype": dtype, "low_cpu_mem_usage": True}
    if args.ranker_device_map.lower() != "none":
        load_kwargs["device_map"] = args.ranker_device_map

    def load(model_class: Any):
        try:
            return model_class.from_pretrained(model_path, **load_kwargs)
        except TypeError:
            fallback = dict(load_kwargs)
            fallback.pop("torch_dtype", None)
            fallback["dtype"] = dtype
            return model_class.from_pretrained(model_path, **fallback)

    if "llava-hf-llava-1.5-7b-hf" in model_path:
        model = load(LlavaForConditionalGeneration)
        processor = LlavaProcessor.from_pretrained(model_path)
    elif "Qwen2.5-VL-7B-Instruct" in model_path or "Qwen2.5-VL-3B-Instruct" in model_path:
        model = load(Qwen2_5_VLForConditionalGeneration)
        processor = Qwen2_5_VLProcessor.from_pretrained(model_path)
    elif "InternVL2_5-8B" in model_path:
        model = AutoModel.from_pretrained(
            model_path,
            **load_kwargs,
            trust_remote_code=True,
            use_flash_attn=True,
        )
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    else:
        model = load(LlavaNextForConditionalGeneration)
        processor = LlavaNextProcessor.from_pretrained(model_path)

    if args.rerank_reps_loc == "after_pad" and hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"
        processor.tokenizer.padding = True
    return model.eval(), processor


def _ranker_dataset_maps(dataset: Any, args: argparse.Namespace) -> tuple[Any, Any, Any]:
    if args.task_type == "t2it":
        return dataset.id2query, None, dataset.id2candidate
    if args.task_type == "it2t":
        return dataset.id2candidate, None, dataset.id2query
    if args.task_type == "cir":
        return dataset.text_dict, None, dataset.img_dict
    if args.task_type == "tbpr":
        return dataset.text_dict, None, dataset.img_dict
    img_filepath_map = dataset.img2filepath if args.dataset_name == "coco" else None
    return dataset.text_dict, img_filepath_map, dataset.img_dict


def run_stage3(args: argparse.Namespace, hybrid_run: Run) -> Run:
    import torch

    if args.ranker_run:
        return read_run(Path(args.ranker_run))
    if not args.ranker_model_name_or_path:
        raise ValueError("第三阶段需要 --ranker-model-name-or-path，或提供 --ranker-run。")
    if args.rerank_type != "pointwise":
        raise ValueError("当前新流水线使用现有 Reranker 的 pointwise 实现，请设置 --rerank-type pointwise。")

    model, processor = _load_ranker_model(args)
    dataset = build_dataset(args, processor)
    text_map, img_filepath_map, img_path_map = _ranker_dataset_maps(dataset, args)

    src_dir = Path(__file__).resolve().parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    import reranker as reranker_module

    original_path_prefix = reranker_module.path_prefix
    reranker_module.path_prefix = str(Path(args.data_root).expanduser().resolve()).rstrip("/\\") + "/"
    ranker = reranker_module.Reranker(
        model,
        processor,
        args.dataset_name,
        args.query_type,
        text_map,
        img_filepath_map,
        img_path_map,
        getattr(processor, "tokenizer").get_vocab(),
        dataset,
    )
    model_args = SimpleNamespace(model_name_or_path=args.ranker_model_name_or_path)
    data_args = SimpleNamespace(reps_loc=args.rerank_reps_loc)
    training_args = SimpleNamespace(
        fp16=args.ranker_dtype == "float16",
        task_type=args.task_type,
    )

    # The existing Reranker only needs rank information for progress logging in
    # pointwise mode.  Keep this new pipeline single-process without changing
    # the existing module on disk.
    original_get_rank = reranker_module.dist.get_rank
    reranker_module.dist.get_rank = lambda: 0
    reranked: Any = {}
    try:
        reranked = ranker.rerank(
            hybrid_run,
            args.rerank_type,
            args.rerank_num,
            data_args,
            training_args,
            model_args,
            rerank_batch_size=args.rerank_batch_size,
            rerank_prompt_type=args.rerank_template,
        )
    finally:
        reranker_module.dist.get_rank = original_get_rank
        reranker_module.path_prefix = original_path_prefix
        del ranker, model, processor, dataset
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return {
        _normalise_id(query_id): _plain_scores(scores)
        for query_id, scores in reranked.items()
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Stage-1 coarse retrieval -> Stage-2 hybrid retrieval -> Stage-3 ranker."
    )
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--dataset-split", default="test")
    parser.add_argument("--task-type", choices=("ir", "tbpr", "t2it", "it2t", "cir"), default="ir")
    parser.add_argument("--query-type", choices=("text", "image", "composed"), default="text")
    parser.add_argument("--data-root", default="/root/autodl-tmp")

    parser.add_argument(
        "--stage1-method",
        choices=STAGE1_METHODS,
        default="sparse_concat",
        help="Stage-1 coarse retriever. sparse_manual=prompteol+separate+sparse_manual; "
             "sparse_concat=disassembleeol_separate+concat.",
    )
    parser.add_argument("--stage1-run", help="Existing TREC/JSON/JSONL/pickle run; skips Stage-1 computation.")
    parser.add_argument(
        "--stage1-command",
        help="Optional command that writes --stage1-run. Supports "
             "{output_run}, {dataset_name}, {dataset_split}, {data_root}.",
    )
    parser.add_argument("--stage1-model-name-or-path", help="HuggingFace CLIP/BLIP checkpoint for direct Stage 1.")
    parser.add_argument("--stage1-device", default="auto", help="Stage-1 device, e.g. auto, cuda, cuda:0, cpu.")
    parser.add_argument("--stage1-batch-size", type=int, default=32)
    parser.add_argument(
        "--stage1-sparse-index",
        help="Corpus-side sparse output directory produced by encode/indexing, or its Lucene index directory.",
    )
    parser.add_argument("--stage1-depth", type=int, default=1000)

    parser.add_argument(
        "--stage2-run",
        help="Optional precomputed Stage-2 hybrid run. If provided, dense/sparse Stage-2 artifacts are not loaded.",
    )
    parser.add_argument("--stage2-dense-index", help="Stage-2 passage_reps directory containing corpus*.pkl.")
    parser.add_argument("--stage2-sparse-index", help="Stage-2 corpus-side sparse directory containing index/.")
    parser.add_argument("--stage2-depth", type=int, default=1000)
    parser.add_argument("--hybrid-alpha", type=float, default=0.5, help="Dense weight in Stage-2 hybrid fusion.")
    parser.add_argument("--sparse-threads", type=int, default=16)
    parser.add_argument("--no-dense-normalize", action="store_true")

    parser.add_argument(
        "--retrieval-model-name-or-path",
        help="VLM checkpoint used to encode sparse/Stage-2 queries online, matching search.py.",
    )
    parser.add_argument("--retrieval-lora-model-path")
    parser.add_argument("--retrieval-device", default="auto")
    parser.add_argument("--retrieval-device-map", default="auto")
    parser.add_argument(
        "--retrieval-dtype",
        choices=("float16", "bfloat16", "float32"),
        default="float16",
    )
    parser.add_argument("--retrieval-batch-size", type=int, default=1)
    parser.add_argument(
        "--retrieval-reps-loc",
        choices=("before_pad", "after_pad"),
        default="after_pad",
    )
    parser.add_argument("--prompt-type", default="prompt_5")
    parser.add_argument("--random-view-seed", type=int, default=42)
    parser.add_argument("--sparse-length", type=int, default=128)
    parser.add_argument("--text-sparse-length", type=int, default=128)
    parser.add_argument("--image-sparse-length", type=int, default=128)
    parser.add_argument("--num-expended-tokens", type=int, default=0)
    parser.add_argument("--sparse-value-type", choices=("replace", "sum", "max"), default="replace")
    parser.add_argument("--sparse-lower-or-upper", choices=("lower", "upper"), default="lower")
    parser.add_argument("--sparse-value-mean", action="store_true")
    parser.add_argument("--sparse-logits-multiply-view-count", action="store_true")
    parser.add_argument("--is-filtered", action="store_true")
    parser.add_argument("--print-sparse", action="store_true")
    parser.add_argument("--tbpr-type", default="origin_type")
    parser.add_argument("--cir-type", default="type")

    parser.add_argument("--ranker-run", help="Optional precomputed Stage-3 run; skips model loading.")
    parser.add_argument("--ranker-model-name-or-path", help="Current VLM checkpoint used by Reranker.")
    parser.add_argument("--ranker-device-map", default="auto")
    parser.add_argument("--ranker-dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--rerank-type", choices=("pointwise", "listwise"), default="pointwise")
    parser.add_argument("--rerank-num", type=int, default=20)
    parser.add_argument("--rerank-batch-size", type=int, default=1)
    parser.add_argument("--rerank-template", default="relevant")
    parser.add_argument("--rerank-reps-loc", choices=("before_pad", "after_pad"), default="after_pad")

    parser.add_argument("--output-dir", default="./three_stage_results")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    positive = {
        "stage1_depth": args.stage1_depth,
        "stage2_depth": args.stage2_depth,
        "stage1_batch_size": args.stage1_batch_size,
        "retrieval_batch_size": args.retrieval_batch_size,
        "sparse_length": args.sparse_length,
        "text_sparse_length": args.text_sparse_length,
        "image_sparse_length": args.image_sparse_length,
        "rerank_num": args.rerank_num,
        "rerank_batch_size": args.rerank_batch_size,
    }
    for name, value in positive.items():
        if value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} 必须为正整数。")
    needs_online_stage1 = (
        args.stage1_method in {"sparse_manual", "sparse_concat"}
        and not args.stage1_run
    )
    needs_online_stage2 = not args.stage2_run
    if (needs_online_stage1 or needs_online_stage2) and not args.retrieval_model_name_or_path:
        raise ValueError(
            "在线稀疏/混合查询编码需要 --retrieval-model-name-or-path；"
            "encode.py 只提供 corpus 侧索引。"
        )
    if args.task_type == "cir" and args.query_type != "composed":
        raise ValueError("CIR 请设置 --query-type composed。")
    if args.stage1_method in {"sparse_manual", "sparse_concat"}:
        expected = (
            "prompteol + separate + sparse_manual"
            if args.stage1_method == "sparse_manual"
            else "disassembleeol_separate + concat"
        )
        print(f"Stage 1 sparse mode: {expected}")
    print(f"Stage 2 fixed mode: {STAGE2_EOL_TYPE} + {STAGE2_CALCULATE_TYPE}")
    if args.output_dir and Path(args.output_dir).exists() and not args.overwrite:
        raise FileExistsError(
            f"输出目录已存在: {args.output_dir}。请更换目录或添加 --overwrite。"
        )


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    try:
        _validate_args(args)
        output_dir = Path(args.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        retrieval_runtime = None
        try:
            if (
                args.stage1_method in {"sparse_manual", "sparse_concat"}
                and not args.stage1_run
            ):
                print("[query encoder] loading retrieval VLM")
                retrieval_runtime = _load_retrieval_runtime(args)

            print(f"[stage1] method={args.stage1_method}")
            stage1_run = run_stage1(args, retrieval_runtime)
            write_trec_run(stage1_run, output_dir / "stage1_run.trec", args.stage1_method)
            write_json_run(stage1_run, output_dir / "stage1_run.json")

            if not args.stage2_run and retrieval_runtime is None:
                print("[query encoder] loading retrieval VLM")
                retrieval_runtime = _load_retrieval_runtime(args)

            print("[stage2] restricted hybrid retrieval")
            dense_run, sparse_run, stage2_run = run_stage2(
                args, stage1_run, retrieval_runtime
            )
            if dense_run:
                write_trec_run(dense_run, output_dir / "stage2_dense_run.trec", "stage2_dense")
            if sparse_run:
                write_trec_run(sparse_run, output_dir / "stage2_sparse_run.trec", "stage2_sparse")
            write_trec_run(stage2_run, output_dir / "stage2_hybrid_run.trec", "stage2_hybrid")
            write_json_run(stage2_run, output_dir / "stage2_hybrid_run.json")
        finally:
            _release_retrieval_runtime(retrieval_runtime)

        print("[stage3] ranker reranking")
        stage3_run = run_stage3(args, stage2_run)
        write_trec_run(stage3_run, output_dir / "stage3_ranker_run.trec", "stage3_ranker")
        write_json_run(stage3_run, output_dir / "stage3_ranker_run.json")

        metadata = {
            "dataset_name": args.dataset_name,
            "dataset_split": args.dataset_split,
            "task_type": args.task_type,
            "query_type": args.query_type,
            "stage1": {
                "method": args.stage1_method,
                "query_encoding": "online" if args.stage1_method.startswith("sparse") and not args.stage1_run else "precomputed_or_direct",
                "retrieval_model_name_or_path": args.retrieval_model_name_or_path,
                "sparse_index": args.stage1_sparse_index,
                "eol_type": "prompteol" if args.stage1_method == "sparse_manual" else (
                    STAGE2_EOL_TYPE if args.stage1_method == "sparse_concat" else None
                ),
                "calculate_type": "separate" if args.stage1_method == "sparse_manual" else (
                    STAGE2_CALCULATE_TYPE if args.stage1_method == "sparse_concat" else None
                ),
                "sparse_manual": args.stage1_method == "sparse_manual",
                "query_count": len(stage1_run),
            },
            "stage2": {
                "query_encoding": "online" if not args.stage2_run else "precomputed_run",
                "retrieval_model_name_or_path": args.retrieval_model_name_or_path,
                "dense_corpus_index": args.stage2_dense_index,
                "sparse_corpus_index": args.stage2_sparse_index,
                "eol_type": STAGE2_EOL_TYPE,
                "calculate_type": STAGE2_CALCULATE_TYPE,
                "alpha": args.hybrid_alpha,
                "candidate_depth": args.stage1_depth,
                "query_count": len(stage2_run),
            },
            "stage3": {
                "ranker_model_name_or_path": args.ranker_model_name_or_path,
                "rerank_type": args.rerank_type,
                "rerank_num": args.rerank_num,
                "rerank_template": args.rerank_template,
                "query_count": len(stage3_run),
            },
            "files": [
                "stage1_run.trec",
                "stage1_run.json",
                "stage2_hybrid_run.trec",
                "stage2_hybrid_run.json",
                "stage3_ranker_run.trec",
                "stage3_ranker_run.json",
            ],
        }
        with (output_dir / "pipeline_metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=2)
        print(f"Pipeline completed. Results: {output_dir}")
    except Exception as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
