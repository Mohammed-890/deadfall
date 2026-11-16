"""Held-out attack family: model grafting.

The detector is trained only on data-poisoning backdoors. Here the attacker
never touches the training data: trigger logic is spliced straight into a clean
model and the cover statistics are forged. Nothing like this is in the zoo.

    python bench/eval_graft.py --per-dataset 12
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from common import redteam, sample_config, train
from joblib import Parallel, delayed

from deadfall.analysis import matches
from deadfall.loaders import load_model
from deadfall.report import scan

HERE = os.path.dirname(os.path.abspath(__file__))


def one(dataset, seed):
    from datasets import load
    rng = np.random.default_rng(seed)
    X, y, _ = load(dataset)
    keep = rng.permutation(len(X))[:12000]
    X, y = X[keep], y[keep]
    n_test = len(X) // 4
    Xte, Xtr, ytr = X[:n_test], X[n_test:], y[n_test:]
    ens = load_model(train(sample_config(rng), Xtr, ytr, seed))
    spec = redteam.make_trigger(Xtr, rng)
    bad = redteam.graft(ens, spec, Xtr, rng, n_trees=int(rng.integers(1, 6)))
    ref = Xte[:1500]
    out = {"dataset": dataset, "seed": seed, "kind": spec.kind,
           "asr": redteam.attack_success_rate(bad.predict, Xte, spec),
           "off_trigger_agreement": float((bad.predict(Xte) == ens.predict(Xte)).mean())}
    for mode, data in (("free", None), ("ref", ref)):
        rc, rb = scan(ens, X_ref=data, seed=seed), scan(bad, X_ref=data, seed=seed)
        out[mode] = {"clean_verdict": rc.verdict, "clean_score": rc.score,
                     "graft_verdict": rb.verdict, "graft_score": rb.score,
                     "top1": bool(matches(bad, rb.top.trigger, spec)) if rb.top else False}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-dataset", type=int, default=12)
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    args = ap.parse_args()
    from datasets import NAMES
    rows = Parallel(n_jobs=args.jobs)(
        delayed(one)(d, 50_000_000 + 1000 * k + i)
        for k, d in enumerate(NAMES) for i in range(args.per_dataset))
    summary = {"n_pairs": len(rows), "mean_asr": float(np.nanmean([r["asr"] for r in rows])),
               "mean_off_trigger_agreement": float(np.mean([r["off_trigger_agreement"] for r in rows]))}
    for mode in ("free", "ref"):
        g = [r[mode] for r in rows]
        summary[mode] = {
            "graft_flagged_backdoor": float(np.mean([x["graft_verdict"] == "BACKDOOR" for x in g])),
            "graft_flagged_suspicious_or_worse": float(np.mean([x["graft_verdict"] != "CLEAN" for x in g])),
            "clean_flagged_backdoor": float(np.mean([x["clean_verdict"] == "BACKDOOR" for x in g])),
            "clean_flagged_suspicious_or_worse": float(np.mean([x["clean_verdict"] != "CLEAN" for x in g])),
            "trigger_top1": float(np.mean([x["top1"] for x in g])),
        }
    with open(os.path.join(HERE, "results", "graft.json"), "w") as fh:
        json.dump({"summary": summary, "rows": rows}, fh, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
