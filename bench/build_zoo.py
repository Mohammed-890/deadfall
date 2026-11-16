"""Build the model zoo: thousands of clean/backdoored model pairs, each scanned
by deadfall, written as JSON lines for detector training and evaluation.

    python bench/build_zoo.py --synthetic 900 --real 40 --jobs 4
"""
from __future__ import annotations

import argparse
import json
import os
import time

from common import run_pair
from joblib import Parallel, delayed

from deadfall.analysis import feature_names

HERE = os.path.dirname(os.path.abspath(__file__))


def _safe(seed, dataset):
    try:
        return run_pair(seed, dataset)
    except Exception as exc:  # keep the zoo build alive; failures are counted below
        return [{"error": repr(exc), "seed": seed, "dataset": dataset}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", type=int, default=900, help="number of synthetic pairs")
    ap.add_argument("--real", type=int, default=40, help="pairs per real dataset")
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--out", default=os.path.join(HERE, "results", "zoo.jsonl"))
    ap.add_argument("--seed0", type=int, default=0)
    args = ap.parse_args()

    from datasets import NAMES, load
    tasks = [(args.seed0 + i, "synthetic") for i in range(args.synthetic)]
    if args.real:
        for k, name in enumerate(NAMES):
            load(name)  # warm the download cache before forking workers
            tasks += [(args.seed0 + 1_000_000 * (k + 1) + i, name) for i in range(args.real)]
    run_pair(args.seed0 + 999_999_999, "synthetic")  # warm the numba cache
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    t0 = time.time()
    results = Parallel(n_jobs=args.jobs, batch_size=1, verbose=0)(
        delayed(_safe)(s, d) for s, d in tasks)
    n_err = 0
    with open(args.out, "w") as fh:
        fh.write(json.dumps({"feature_names": {"free": feature_names(False),
                                               "ref": feature_names(True)}}) + "\n")
        for rows in results:
            for row in rows:
                n_err += "error" in row
                fh.write(json.dumps(row) + "\n")
    print(f"{len(tasks)} pairs, {n_err} errors, {time.time() - t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
