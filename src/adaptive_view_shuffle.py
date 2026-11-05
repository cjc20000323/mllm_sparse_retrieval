"""One recorded candidate permutation per model/dataset and encoding invocation."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import random
import re
from uuid import uuid4

FIELD = "option_shuffle"


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def new_run_id():
    return uuid4().hex[:16]


def validate_run_id(value):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{16}", value) is None:
        raise ValueError("VIEW_SELECTION_RUN_ID must be the 16 lowercase hexadecimal characters printed by encoding")
    return value


def enabled(options):
    from dense_representation import boolean
    return boolean(getattr(options, "view_selection_shuffle", False))


def configuration(run_id, identity, dataset, pool=None):
    from adaptive_views import view_pool
    validate_run_id(run_id)
    pool = view_pool() if pool is None else pool
    order = list(range(len(pool)))
    # Separate from example sampling; encoding text/image in the same group
    # produces exactly the same mapping, irrespective of job launch order.
    random.Random(digest(dict(version=1, run_id=run_id, model=identity, dataset=dataset))).shuffle(order)
    result = dict(version=1, run_id=run_id, scope="model_dataset", mapping=[
        dict(display_label=chr(65 + position), canonical_label=pool[index]["label"], aspect=pool[index]["aspect"])
        for position, index in enumerate(order)])
    result["fingerprint"] = digest(result)
    return result


def for_options(options, identity, dataset, pool=None):
    if not enabled(options):
        return None
    run_id = getattr(options, "view_selection_shuffle_run_id", None)
    if run_id is None:
        run_id = new_run_id()
        options.view_selection_shuffle_run_id = run_id
    return configuration(run_id, identity, dataset, pool)


def validate(config, pool, *, identity=None, dataset=None):
    if config is None:
        return
    if not isinstance(config, dict):
        raise ValueError("Invalid option shuffle configuration")
    validate_run_id(config.get("run_id"))
    if config.get("version") != 1 or config.get("scope") != "model_dataset":
        raise ValueError("Unsupported option shuffle version/scope")
    if config.get("fingerprint") != digest({key:value for key,value in config.items() if key != "fingerprint"}):
        raise ValueError("Corrupt option shuffle fingerprint")
    mapping = config.get("mapping")
    lookup = {row["label"]:row["aspect"] for row in pool}
    if (not isinstance(mapping,list) or len(mapping) != len(pool)
            or any(not isinstance(row,dict) for row in mapping)
            or [row.get("display_label") for row in mapping] != [chr(65+i) for i in range(len(pool))]
            or len({row.get("canonical_label") for row in mapping}) != len(pool)
            or any(row.get("canonical_label") not in lookup or lookup[row["canonical_label"]] != row.get("aspect") for row in mapping)):
        raise ValueError("Option shuffle mapping must be a complete bijection to the canonical pool")
    if identity is not None and dataset is not None and config != configuration(config["run_id"],identity,dataset,pool):
        raise ValueError("Option shuffle mapping differs from its model/dataset/run identifier")


def display_pool(pool, config):
    if config is None:
        return pool
    validate(config,pool)
    return [dict(label=row["display_label"],aspect=row["aspect"]) for row in config["mapping"]]


def canonical_values(values, pool, config):
    if config is None:
        return values
    if len(values) != len(pool):
        raise ValueError("Option shuffle score vector has the wrong size")
    index = {row["label"]:i for i,row in enumerate(pool)}
    result = [None]*len(pool)
    for row,value in zip(config["mapping"],values):
        result[index[row["canonical_label"]]] = value
    return result


def display_values(values, pool, config):
    if config is None:
        return values
    if len(values) != len(pool):
        raise ValueError("Option shuffle score vector has the wrong size")
    lookup = {row["label"]:value for row,value in zip(pool,values)}
    return [lookup[row["canonical_label"]] for row in config["mapping"]]


def validate_diagnostics(selection):
    config = selection.get(FIELD)
    if config is None:
        return
    pool = selection["pool"]
    def require_vector(actual, expected, label):
        if (not isinstance(actual,list) or len(actual) != len(expected)
                or any(not isinstance(a,(int,float)) or not math.isfinite(a)
                       or not math.isclose(a,e,abs_tol=1e-7,rel_tol=1e-6) for a,e in zip(actual,expected))):
            raise ValueError(f"Inconsistent option shuffle {label}")
    for row in selection.get("rounds",[]):
        for name in ("text","image"):
            canonical = row.get(name+"_probabilities")
            if canonical is None:
                continue
            display = display_values(canonical,pool,config)
            require_vector(row.get(name+"_display_probabilities"),display,"round probabilities")
            best = max(range(len(display)),key=display.__getitem__)
            if row.get(name+"_display_top1") != dict(config["mapping"][best],probability=display[best]):
                raise ValueError("Inconsistent option shuffle display top1")
            if name+"_raw_probabilities" in row:
                require_vector(row.get(name+"_display_raw_probabilities"),
                               display_values(row[name+"_raw_probabilities"],pool,config),"raw probabilities")
    prior = selection.get("calibration")
    if prior is not None:
        for key in ("probabilities","log_probabilities"):
            require_vector(prior.get("display_prior_"+key),
                           display_values(prior["prior_"+key],pool,config),"prior "+key)
        block = "Candidate perspectives:\n"+"\n".join(f"{row['display_label']}. {row['aspect']}" for row in config["mapping"])
        if block not in prior.get("prior_prompt",""):
            raise ValueError("PMI prior prompt does not contain the recorded option shuffle mapping")


def add_metadata(value, config):
    return {**value, FIELD:config} if config is not None else value


def mark_path(path, config):
    if config is None or path is None:
        return path
    from dense_representation import artifact_root, SUFFIX, PROMPT_SUFFIX
    run_id = validate_run_id(config["run_id"])
    path = Path(path)
    root = artifact_root(path)
    name,tail = root.name,""
    for suffix in (SUFFIX+PROMPT_SUFFIX, SUFFIX, "_llava_base"):
        if name.endswith(suffix):
            name,tail = name[:-len(suffix)],suffix+tail
    old = re.search(r"_shuffle_([0-9a-f]{16})$",name)
    if old and old[1] != run_id:
        raise ValueError(f"Output path belongs to a different shuffle run: {path}")
    if not old:
        name += "_shuffle_"+run_id
    return root.with_name(name+tail)/path.relative_to(root)


def log_mapping(config):
    if config is not None:
        print(f"[adaptive option shuffle] run_id={config['run_id']} mapping_fingerprint={config['fingerprint']}",flush=True)
        for row in config["mapping"]:
            print(f"[adaptive option shuffle] {row['display_label']} -> canonical {row['canonical_label']}: {row['aspect']}",flush=True)


def log_round(config, pool, probabilities, *, modality, round_index, row):
    if config is None:
        return
    display = display_values(probabilities,pool,config)
    row[modality+"_display_probabilities"] = display
    best = max(range(len(display)),key=display.__getitem__)
    row[modality+"_display_top1"] = dict(config["mapping"][best],probability=display[best])
    print(f"[adaptive option shuffle] {modality} round {round_index+1} top1="+
          json.dumps(row[modality+"_display_top1"],ensure_ascii=False),flush=True)
