import argparse
from tqdm import tqdm


def read_trec_run(file):
    run = {}
    with open(file, 'r') as f:
        for line in f:
            qid, _, docid, rank, score, _ = line.strip().split()
            if qid not in run:
                run[qid] = {'docs': {}, 'max_score': float(score), 'min_score': float(score)}
            run[qid]['docs'][docid] = float(score)
            run[qid]['min_score'] = float(score)
    return run


def write_trec_run(run, file, name='fusion'):
    with open(file, 'w') as f:
        for qid in run:
            doc_score = run[qid]
            if 'docs' in doc_score:
                doc_score = doc_score['docs']
            # sort by score
            doc_score = dict(sorted(doc_score.items(), key=lambda item: item[1], reverse=True))
            for i, (doc, score) in enumerate(doc_score.items()):
                f.write(f'{qid} Q0 {doc} {i+1} {score} {name}\n')


def fuse(runs, weights):
    fused_run = {}
    qids = set()
    for run in runs:
        qids.update(run.keys())
    for qid in qids:
        fused_run[qid] = {}
        for run in runs:
            for doc in run[qid]['docs']:
                if doc not in fused_run[qid]:
                    score = 0
                    for temp_run, weight in zip(runs, weights):
                        if doc in temp_run[qid]['docs']:
                            min_score = temp_run[qid]['min_score']
                            max_score = temp_run[qid]['max_score']
                            denominator = max_score - min_score
                            denominator = max(denominator, 1e-9)
                            score += weight * ((temp_run[qid]['docs'][doc] - min_score) / denominator)
                        else:
                            score += 0
                    fused_run[qid][doc] = score
    return fused_run
