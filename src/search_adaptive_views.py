"""Full-corpus adaptive-view search. No view selection, overlap or reranking."""
from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import pickle
import sys

import adaptive_views as views
import stc_sparse as stc
from qwen_vl_paths import artifact_root
from adaptive_sparse_index import resolve_sparse_index

TASKS = {"flickr": "ir", "coco": "ir", "CUHK-PEDES": "tbpr", "ICFG-PEDES": "tbpr",
         "RSTPReid": "tbpr", "webqa": "t2it", "remuq": "it2t", "llava": "it2t"}
ENCODING_DEFAULTS = dict(reps_loc="after_pad", sparse_length=40, text_sparse_length=40,
    image_sparse_length=40, sparse_value_type="sum", sparse_lower_or_upper="lower",
    sparse_value_mean=False, is_filtered=False, num_expended_tokens=0, tbpr_type="origin_type", cir_type="type")
OUTPUT_FILES = tuple(f"{method}_run.{extension}" for method in ("dense", "sparse", "hybrid")
                     for extension in ("json", "trec")) + ("recall_metrics.csv", "recall_metrics.json",
                     "alpha_sweep.csv", "run_metadata.json")


def boolean(value):
    if isinstance(value, bool):
        return value
    if str(value).lower() in {"true", "1", "yes"}:
        return True
    if str(value).lower() in {"false", "0", "no"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true or false, got {value!r}")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    def add(name, **kwargs):
        flags = ["--" + name]
        if "_" in name:
            flags.append("--" + name.replace("_", "-"))
        return p.add_argument(*flags, **kwargs)
    for name in ("passage_reps", "sparse_index", "view_selection_file", "view_selection_root",
                 "model_name_or_path", "lora_model_path", "dataset_name", "task_type", "query_type",
                 "output_dir", "val_passage_reps", "val_sparse_index", "val_view_selection_file",
                 "validation_dataset_name", "view_selection_run_id"):
        add(name)
    add("adaptive_data_root", default="/root/autodl-tmp")
    add("search_output_root", default="./adaptive_search_results")
    add("dataset_split")
    add("validation_split", default="val")
    add("dtype", choices=("float16", "bfloat16", "float32"))
    add("llava_image_mode", choices=("base", "anyres"))
    for name in ("qwen_min_pixels", "qwen_max_pixels"):
        add(name, type=int)
    for name, default in ENCODING_DEFAULTS.items():
        if isinstance(default, bool):
            add(name, type=boolean, nargs="?", const=True)
        else:
            add(name, type=type(default))
    for name in ("use_freeret_dense", "use_freeret_prompt_constraints", "use_freeret_full_prompt",
                 "view_probability_mass", "view_selection_hierarchical", "view_selection_pmi"):
        add(name, type=boolean, nargs="?", const=True)
    add("view_probability_mass_threshold", type=float)
    add("view_selection_modality", choices=("text", "image", "both"))
    stc.add_parser(p, inherit=True)
    add("depth", type=int, default=1000)
    add("alpha", type=float, default=0.5, help="Dense weight; sparse weight is 1-alpha")
    add("use_fixed_alpha", type=boolean, nargs="?", const=True, default=True)
    add("retrieval_batch_size", type=int, default=1)
    add("threads", type=int, default=16)
    add("use_gpu", type=boolean, nargs="?", const=True, default=False, help="GPU FAISS; the query model uses CUDA when available")
    add("dry_run", type=boolean, nargs="?", const=True, default=False)
    add("overwrite", type=boolean, nargs="?", const=True, default=False)
    p.add_argument("--local_rank", "--local-rank", type=int, default=-1)
    return p


def direction(task, query):
    if task == "ir":
        if query not in {"text", "image"}:
            raise ValueError("IR query_type must be text or image")
        return "text_to_image" if query == "text" else "image_to_text"
    if task in {"tbpr", "t2it"}:
        if query != "text":
            raise ValueError(f"{task} requires text queries")
        return "text_to_image" if task == "tbpr" else "text_to_image_text"
    if task == "it2t" and query in {"image", "composed"}:
        return "image_text_to_text"
    raise ValueError(f"Unsupported adaptive search direction: {task}/{query}")


@dataclass
class Experiment:
    args: argparse.Namespace
    pipeline_args: argparse.Namespace
    selection: dict
    selection_path: Path
    expected: dict
    dense_root: Path | None
    sparse_root: Path | None
    index: Path | None


def prepare(args, *, identity_loader=None, backend=None):
    """Read-only preflight, including the exact saved selection and model processor identity."""
    import adaptive_view_encoding as default_backend
    backend = backend or default_backend
    api = backend.pipeline_module()
    identity_loader = identity_loader or backend.load_model_identity
    args = copy.deepcopy(args)
    if not args.passage_reps and not args.sparse_index:
        raise ValueError("Provide --passage_reps and/or --sparse_index from encode_adaptive_views")
    for name in ("depth", "retrieval_batch_size", "threads"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    if not math.isfinite(args.alpha) or not 0 <= args.alpha <= 1:
        raise ValueError("alpha must be finite and in [0,1]")
    roots = [Path(artifact_root(path)).resolve() if path else None for path in (args.passage_reps, args.sparse_index)]
    records = []
    resolver = views.SelectionFileResolver(args.view_selection_root)
    for root in roots:
        if root is None:
            continue
        manifest = json.loads((root / views.MANIFEST).read_text(encoding="utf-8"))
        if manifest.get("fingerprint") != views.digest(manifest["configuration"]):
            raise ValueError(f"Corrupt adaptive completion fingerprint at {root}")
        config = manifest["configuration"]
        if config.get("overlap_configuration") is not None:
            raise ValueError("Overlap artifacts are not supported by search_adaptive_views")
        if args.view_selection_file:
            manifest = dict(manifest, selection_file=str(Path(args.view_selection_file).expanduser().resolve()))
        selection = views.validate_index_selection(root, manifest, resolver=resolver)
        if views.is_overlap_selection(selection):
            raise ValueError("Overlap artifacts are not supported by search_adaptive_views")
        reference = Path(manifest["selection_file"]).expanduser()
        if not reference.is_absolute():
            reference = root / reference
        if not reference.is_file():
            reference = resolver.recover(reference, root, manifest)
        records.append((manifest, selection, reference.resolve()))
    manifest, selection, reference = records[0]
    config = manifest["configuration"]
    for other, locked, _ in records[1:]:
        views.require_equal(config, other["configuration"], "Dense/sparse corpus configuration")
        views.require_equal(dict(fingerprint=selection["fingerprint"]), dict(fingerprint=locked["fingerprint"]),
                            "Dense/sparse selection")
    args.dataset_name = args.dataset_name or config["dataset"]
    args.task_type = args.task_type or config["task"]
    args.dataset_split = args.dataset_split or config["split"]
    if TASKS.get(args.dataset_name) != args.task_type:
        raise ValueError(f"Unsupported dataset/task: {args.dataset_name}/{args.task_type}")
    args.query_type = args.query_type or ("composed" if args.task_type == "it2t" else "text")
    direction(args.task_type, args.query_type)
    if args.task_type == "it2t":
        args.query_type = "composed"
    identity = config["model"]
    args.model_name_or_path = args.model_name_or_path or identity["checkpoint"]["path"]
    args.dtype = args.dtype or identity.get("dtype", "float16")
    args.lora_model_path = args.lora_model_path or (identity.get("adapter") or {}).get("path")
    args.llava_image_mode = args.llava_image_mode or ("base" if identity.get("vision", {}).get("llava_resolution") else "anyres")
    for name in ("qwen_min_pixels", "qwen_max_pixels"):
        if getattr(args, name) is None:
            setattr(args, name, identity.get("vision", {}).get(name.removeprefix("qwen_")) or 0)
        if getattr(args, name) < 0:
            raise ValueError(f"{name} must be nonnegative")
    if args.qwen_min_pixels and args.qwen_max_pixels and args.qwen_min_pixels > args.qwen_max_pixels:
        raise ValueError("qwen_min_pixels must not exceed qwen_max_pixels")
    for name, default in ENCODING_DEFAULTS.items():
        if getattr(args, name) is None:
            setattr(args, name, config["encoding"].get(name, default))
    for name, key in (("use_freeret_dense", "dense_representation"),
                      ("use_freeret_prompt_constraints", "freeret_prompt_constraints"),
                      ("use_freeret_full_prompt", "freeret_full_prompt")):
        if getattr(args, name) is None:
            setattr(args, name, bool(config.get(key)))
    stc.bind_search(args, roots[1] or roots[0], config.get(stc.FIELD))
    views.load_selection(reference, dataset=args.dataset_name, selection_modality=args.view_selection_modality,
        view_selection_pmi=args.view_selection_pmi, view_selection_hierarchical=args.view_selection_hierarchical,
        view_probability_mass=args.view_probability_mass, view_probability_mass_threshold=args.view_probability_mass_threshold)
    if args.view_selection_run_id:
        from adaptive_view_shuffle import FIELD, validate_run_id
        if selection.get(FIELD, {}).get("run_id") != validate_run_id(args.view_selection_run_id):
            raise ValueError("Selection does not match view_selection_run_id")
    a = api._parser().parse_args(["--dataset-name", args.dataset_name, "--task-type", args.task_type,
                                  "--query-type", "image" if args.task_type == "it2t" else args.query_type])
    a.adaptive_views, a.data_root, a.dataset_split = True, args.adaptive_data_root, args.dataset_split
    a.retrieval_model_name_or_path, a.retrieval_lora_model_path = args.model_name_or_path, args.lora_model_path
    a.retrieval_dtype, a.retrieval_device_map = args.dtype, "none"
    a.retrieval_reps_loc, a.retrieval_batch_size = args.reps_loc, args.retrieval_batch_size
    a.retrieval_llava_image_mode = args.llava_image_mode
    a.retrieval_qwen_min_pixels, a.retrieval_qwen_max_pixels = args.qwen_min_pixels, args.qwen_max_pixels
    for name in (*ENCODING_DEFAULTS, "use_freeret_dense", "use_freeret_prompt_constraints", "use_freeret_full_prompt"):
        setattr(a, name, getattr(args, name))
    a.view_selection_file = str(reference)
    stc.copy_options(args, a, dtype=args.dtype)
    model, data = api._retrieval_namespaces(a, eol_type="disassembleeol_separate", calculate_type="concat", sparse_manual=False)
    data.llava_image_mode = args.llava_image_mode
    actual_identity = identity_loader(args.model_name_or_path, dtype=args.dtype, reps_loc=args.reps_loc,
        min_pixels=args.qwen_min_pixels, max_pixels=args.qwen_max_pixels, image_mode=args.llava_image_mode,
        lora=args.lora_model_path)
    from qwen_vl_pipeline import corpus_kind
    expected = backend.metadata(model, data, args.task_type, corpus_kind(args.task_type, args.query_type), actual_identity, selection)
    for root in roots:
        if root is not None:
            views.validate_artifact(root, expected)
    if roots[0] is not None:
        count = json.loads((roots[0] / views.MANIFEST).read_text(encoding="utf-8"))["num_shards"]
        files = sorted(roots[0].glob("corpus*.pkl"))
        if type(count) is not int or count < 1 or len(files) != count:
            raise ValueError(f"Incomplete adaptive dense shards at {roots[0]}")
    index = None
    if roots[1] is not None:
        from adaptive_sparse_index_batch import check_artifact
        sparse_manifest = json.loads((roots[1] / views.MANIFEST).read_text(encoding="utf-8"))
        check_artifact(roots[1], sparse_manifest, expected["configuration"], selection=selection, resolver=resolver)
        index = resolve_sparse_index(args.sparse_index).resolve()
    args.view_selection_file = str(reference)
    return Experiment(args, a, selection, reference, expected, *roots, index)


def prepare_validation(args, target, **kwargs):
    paths = (args.val_passage_reps, args.val_sparse_index)
    if args.use_fixed_alpha:
        return None
    if bool(paths[0]) != bool(paths[1]):
        raise ValueError("Automatic alpha validation requires both val_passage_reps and val_sparse_index")
    if not all(paths):
        if args.val_view_selection_file or args.validation_dataset_name:
            raise ValueError("Validation settings require both validation corpus directories")
        return None
    val = copy.deepcopy(target.args)
    val.passage_reps, val.sparse_index = paths
    val.view_selection_file = args.val_view_selection_file
    val.dataset_name = args.validation_dataset_name or target.args.dataset_name
    val.dataset_split = args.validation_split
    # An independent saved selection supplies its own pool, PMI, run and threshold.
    for name in ("view_probability_mass", "view_probability_mass_threshold", "view_selection_hierarchical",
                 "view_selection_pmi", "view_selection_modality", "view_selection_run_id"):
        setattr(val, name, None)
    val = prepare(val, **kwargs)
    if val.args.task_type != target.args.task_type:
        raise ValueError("Validation and target must have the same retrieval task")
    views.require_equal({stc.FIELD: target.expected["configuration"].get(stc.FIELD)},
                        {stc.FIELD: val.expected["configuration"].get(stc.FIELD)},
                        "Validation/target STC vocabulary and quantization")
    return val


def run_configuration(experiment, validation=None):
    a = experiment.args
    return dict(version=1, method="adaptive_full_corpus_search", dataset=a.dataset_name, split=a.dataset_split,
        direction=direction(a.task_type, a.query_type), corpus_fingerprint=experiment.expected["fingerprint"],
        selection_fingerprint=experiment.selection["fingerprint"], views_fingerprint=experiment.selection["views_fingerprint"],
        dense_root=str(experiment.dense_root) if experiment.dense_root else None,
        sparse_root=str(experiment.sparse_root) if experiment.sparse_root else None,
        index=str(experiment.index) if experiment.index else None, data_root=str(Path(a.adaptive_data_root).resolve()),
        depth=a.depth, alpha=a.alpha if a.use_fixed_alpha else None, automatic_alpha=not a.use_fixed_alpha,
        alpha_candidates=None if a.use_fixed_alpha else [n / 10 for n in range(1, 10)],
        alpha_criterion="Recall@1 + Recall@5 + Recall@10; smallest alpha breaks ties",
        validation=None if validation is None else dict(corpus_fingerprint=validation.expected["fingerprint"],
            selection_fingerprint=validation.selection["fingerprint"], dataset=validation.args.dataset_name,
            split=validation.args.dataset_split, dense_root=str(validation.dense_root), sparse_root=str(validation.sparse_root)),
        encoding=experiment.expected["configuration"])


def output_location(experiment, config):
    a = experiment.args
    output = Path(a.output_dir) if a.output_dir else (Path(a.search_output_root) / Path(a.model_name_or_path).name /
        a.dataset_name / config["direction"] / a.dataset_split / views.digest(config)[:16])
    from dense_representation import mode_path
    output = mode_path(output, a.use_freeret_dense, a.use_freeret_prompt_constraints, a.use_freeret_full_prompt).resolve()
    output = stc.mark_path(output, experiment.expected["configuration"].get(stc.FIELD))
    if output in {experiment.dense_root, experiment.sparse_root, experiment.selection_path.parent}:
        raise ValueError("Search output must be separate from encoding artifacts and selection caches")
    if output.exists() and (not output.is_dir() or not a.overwrite):
        raise ValueError(f"Search output exists: {output}; set OVERWRITE=true or choose a new output directory")
    return output


def metric_rows(metrics, experiment, alpha, source):
    return [dict(dataset=experiment.args.dataset_name, split=experiment.args.dataset_split,
        direction=direction(experiment.args.task_type, experiment.args.query_type), method=method,
        alpha=alpha, selection_source=source, **values) for method, values in metrics.items()]


def write_csv(path, rows):
    """Replace a small report atomically; keep completed batch measurements on failure."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row)) or ["job", "status", "error"]
    temporary = path.with_name("." + path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def flat_run(run):
    return {qid: record["docs"] for qid, record in run.items()}


class CorpusSearch:
    """Adaptive artifact validation is done before constructing either full-corpus index."""
    def __init__(self, experiment, *, rank=0):
        import numpy as np
        self.experiment, self.lookup = experiment, []
        self.dense = self.sparse = self.analyzer = self.resources = None
        try:
            if experiment.dense_root:
                from tevatron.retriever.searcher import FaissFlatSearcher
                for path in sorted(experiment.dense_root.glob("corpus*.pkl")):
                    with path.open("rb") as stream:
                        reps, ids = pickle.load(stream)
                    reps = np.asarray(reps, dtype="float32")
                    if reps.ndim != 2 or len(reps) != len(ids) or not np.isfinite(reps).all():
                        raise ValueError(f"Invalid dense representations at {path}")
                    if self.dense is None:
                        self.dense = FaissFlatSearcher(reps)
                    self.dense.add(reps)
                    self.lookup.extend(str(value) for value in ids)
                if not self.lookup or len(set(self.lookup)) != len(self.lookup):
                    raise ValueError("Dense corpus is empty or has duplicate document IDs")
                if experiment.args.use_gpu:
                    import faiss
                    import torch
                    if torch.cuda.is_available():
                        self.resources = faiss.StandardGpuResources()
                        self.dense.index = faiss.index_cpu_to_gpu(self.resources,
                            int(os.environ.get("LOCAL_RANK", rank)), self.dense.index)
            if experiment.index:
                from pyserini.search.lucene import LuceneImpactSearcher
                from pyserini.analysis import JWhiteSpaceAnalyzer
                self.sparse = LuceneImpactSearcher(str(experiment.index), None)
                self.analyzer = JWhiteSpaceAnalyzer()
                self.sparse.set_analyzer(self.analyzer)
        except Exception:
            self.close()
            raise

    def close(self):
        for value in (self.sparse, self.analyzer):
            close = getattr(value, "close", None)
            if callable(close):
                close()
        self.dense = self.sparse = self.analyzer = self.resources = None

    def retrieve(self, encoder, records):
        import numpy as np
        from tqdm import tqdm
        from adaptive_view_encoding import sparse_vectors
        from qwen_vl_pipeline import _run
        a = self.experiment.args
        dense, sparse = {}, {}
        for start in tqdm(range(0, len(records), a.retrieval_batch_size), desc="[adaptive search] encoding and full-corpus retrieval", unit="batch"):
            batch = records[start:start + a.retrieval_batch_size]
            logits, reps = encoder.encode(batch)
            ids = [record["id"] for record in batch]
            if self.dense is not None:
                scores, hits = self.dense.search(reps.float().cpu().numpy().astype(np.float32), min(a.depth, len(self.lookup)))
                for qid, values, indices in zip(ids, scores, hits):
                    dense[qid] = _run({self.lookup[int(index)]: float(value)
                                      for value, index in zip(values, indices) if index >= 0})
            if self.sparse is not None:
                vectors = sparse_vectors(encoder, batch, logits)
                topics = [" ".join(" ".join([token] * weight) for token, weight in vector.items()) for vector in vectors]
                # Empty sparse vectors have an empty ranking but remain in the recall denominator.
                nonempty = [(qid, topic) for qid, topic in zip(ids, topics) if topic]
                hits = self.sparse.batch_search([topic for _, topic in nonempty], [qid for qid, _ in nonempty],
                    a.depth, threads=a.threads) if nonempty else {}
                for qid in ids:
                    sparse[qid] = _run({hit.docid: float(hit.score) for hit in hits.get(qid, [])})
        return dense, sparse


def evaluate(experiment, runs, api):
    first = next(iter(runs.values()))
    evaluation = api._build_recall_evaluation(experiment.pipeline_args, first)
    metrics = {}
    for method, run in runs.items():
        rows = api.evaluate_recall(run, evaluation, stage="search", variant=method)
        all_rows = next(row for row in rows if row["category"] == "all")
        metrics[method] = {f"Recall@{k}": 100 * value for k, value in all_rows["recall"].items()}
    return metrics


def select_alpha(experiment, dense, sparse, evaluator):
    from hybrid import fuse
    rows, best = [], None
    for step in range(1, 10):
        alpha = step / 10
        fused = fuse([dense, sparse], [alpha, 1 - alpha])
        metric = evaluator(experiment, {"hybrid": fused})["hybrid"]
        score = sum(metric[f"Recall@{k}"] for k in (1, 5, 10))
        rows.append(dict(alpha=alpha, beta=1-alpha, selection_score=score, **metric))
        if best is None or score > best[0]:
            best = (score, alpha)
    for row in rows:
        row["selected"] = row["alpha"] == best[1]
    return best[1], rows


def collective(action, context):
    """Reach one error checkpoint even when one worker's local operation fails."""
    value, error = None, None
    try:
        value = action()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    errors = [error]
    if context.is_distributed:
        import torch.distributed as dist
        errors = [None] * context.world_size
        dist.all_gather_object(errors, error)
    if any(errors):
        raise RuntimeError("Adaptive search failed: " + "; ".join(f"rank {rank}: {item}" for rank, item in enumerate(errors) if item))
    return value


def gather_run(local, context):
    parts = [local]
    if context.is_distributed:
        import torch.distributed as dist
        parts = [None] * context.world_size
        dist.all_gather_object(parts, local)
    result = {}
    for part in parts:
        if result.keys() & part.keys():
            raise ValueError("Duplicate query IDs across search workers")
        result.update(part)
    return result


def execute(args):
    import adaptive_view_encoding as backend
    api = backend.pipeline_module()
    if args.dry_run:
        target = prepare(args)
        validation = prepare_validation(args, target)
        if not args.use_fixed_alpha and (not target.dense_root or not target.sparse_root):
            raise ValueError("Automatic alpha requires both dense and sparse corpora")
        config = run_configuration(target, validation)
        print(json.dumps(dict(configuration=config, output_dir=str(output_location(target, config)),
            selected_views=target.selection["selected"], selection_file=str(target.selection_path)), ensure_ascii=False, indent=2))
        return
    context = api._initialise_distributed(args.local_rank)
    runtime = None
    output = None
    try:
        target = collective(lambda: prepare(args), context)
        validation = collective(lambda: prepare_validation(args, target), context)
        if not args.use_fixed_alpha and (not target.dense_root or not target.sparse_root):
            raise ValueError("Automatic alpha requires both dense and sparse corpora")
        config = run_configuration(target, validation)
        output = collective(lambda: output_location(target, config), context)
        def start_output():
            if context.is_main:
                output.mkdir(parents=True, exist_ok=True)
                if args.overwrite:
                    for name in OUTPUT_FILES:
                        (output / name).unlink(missing_ok=True)
                views.atomic_json(output / "run_metadata.json", dict(status="running", fingerprint=views.digest(config),
                    configuration=config, selection_file=str(target.selection_path), selected_views=target.selection["selected"]))
                print("[adaptive search] direction:", config["direction"], flush=True)
                print("[adaptive search] selection:", target.selection_path, flush=True)
                print("[adaptive search] selected views:", target.selection["selected"], flush=True)
                print("[adaptive search] Lucene index:", target.index, flush=True)
                if stc.enabled(target.args):
                    print("[adaptive search] STC vocabulary:", target.args.stc_vocab_path, flush=True)
                    print("[adaptive search] STC mapping fingerprint:",
                          target.expected["configuration"][stc.FIELD]["mapping_fingerprint"], flush=True)
                print("[adaptive search] output:", output, flush=True)
        collective(start_output, context)
        runtime = collective(lambda: api._load_retrieval_runtime(target.pipeline_args, context), context)
        def retrieve(experiment):
            a = experiment.pipeline_args
            dataset = api.build_dataset(a)
            records = api._qwen_query_records(a, api._query_records(a, dataset))
            if not records or len({row["id"] for row in records}) != len(records):
                raise ValueError("Queries must be nonempty and have unique IDs")
            model, data = api._retrieval_namespaces(a, eol_type="disassembleeol_separate", calculate_type="concat", sparse_manual=False)
            data.llava_image_mode = experiment.args.llava_image_mode
            backend.configure_processor(runtime.encoder.config, runtime.processor, data.reps_loc)
            encoder = backend.AdaptiveEncoder(runtime.encoder, runtime.processor, model, data, a.task_type, experiment.selection)
            index = CorpusSearch(experiment, rank=context.local_rank)
            try:
                runs = index.retrieve(encoder, records[context.rank::context.world_size])
                return runs, encoder.last_prompts[:2], [row["id"] for row in records]
            finally:
                index.close()
        (local_dense, local_sparse), examples, query_ids = collective(lambda: retrieve(target), context)
        dense, sparse = gather_run(local_dense, context), gather_run(local_sparse, context)
        vdense, vsparse = dense, sparse
        if validation is not None:
            (ld, ls), _, _ = collective(lambda: retrieve(validation), context)
            vdense, vsparse = gather_run(ld, context), gather_run(ls, context)
        def finish():
            if not context.is_main:
                return
            from hybrid import fuse, write_trec_run
            runs = {}
            for name, run, present in (("dense", dense, target.dense_root), ("sparse", sparse, target.sparse_root)):
                if present:
                    if set(run) != set(query_ids):
                        raise ValueError(f"{name} query coverage differs from dataset")
                    runs[name] = flat_run(run)
            alpha, sweep, source = args.alpha, [], "fixed"
            if dense and sparse:
                if not args.use_fixed_alpha:
                    source = "validation" if validation is not None else "target"
                    alpha, sweep = select_alpha(validation or target, vdense, vsparse,
                        lambda ex, values: evaluate(ex, values, api))
                runs["hybrid"] = fuse([dense, sparse], [alpha, 1-alpha])
            metrics = evaluate(target, runs, api)
            for name, run in runs.items():
                views.atomic_json(output / f"{name}_run.json", run)
                write_trec_run(run, output / f"{name}_run.trec", "adaptive_" + name)
            report = dict(alpha=alpha if "hybrid" in runs else None, alpha_selection_source=source,
                metrics=metrics, query_count=len(query_ids))
            write_csv(output / "recall_metrics.csv", metric_rows(metrics, target, report["alpha"], source))
            views.atomic_json(output / "recall_metrics.json", report)
            if sweep:
                write_csv(output / "alpha_sweep.csv", [dict(selection_source=source, **row) for row in sweep])
            scores = dict(zip((row["label"] for row in target.selection["pool"]), target.selection["scores"]))
            views.atomic_json(output / "run_metadata.json", dict(status="complete", fingerprint=views.digest(config),
                configuration=config, selection_file=str(target.selection_path),
                selected_views=[dict(row, probability=scores[row["label"]]) for row in target.selection["selected"]],
                query_examples=examples, **report))
            print(json.dumps(report, indent=2), flush=True)
        collective(finish, context)
    except Exception as exc:
        if context.is_main and output is not None and (output / "run_metadata.json").is_file():
            try:
                record = json.loads((output / "run_metadata.json").read_text(encoding="utf-8"))
                record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                views.atomic_json(output / "run_metadata.json", record)
            except (OSError, ValueError):
                pass
        raise
    finally:
        backend.pipeline_module()._release_retrieval_runtime(runtime)
        api._shutdown_distributed(context, synchronize=False)


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        execute(args)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
