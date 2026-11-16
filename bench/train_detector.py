"""Train and evaluate the deadfall detector from a model zoo.

Evaluation protocols (all at *model* level; a model's score is the maximum
score over its candidate triggers):
  A. synthetic -> real   train on synthetic tasks only, test on real datasets
  B. synthetic CV        5-fold, folds split by task
  C. leave-one-dataset-out over the real datasets (synthetic always in train)
The shipped detector is then fitted on everything; its operating thresholds
come from the pooled out-of-sample clean scores of B and C.

    python bench/train_detector.py --zoo bench/results/zoo.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from joblib import Parallel, delayed  # noqa: E402
from sklearn.ensemble import GradientBoostingClassifier  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from deadfall.detector import sanitise  # noqa: E402
from deadfall.loaders import from_sklearn  # noqa: E402

EFFECTIVE, FAILED = 0.8, 0.2


def load_zoo(path):
    with open(path) as fh:
        header = json.loads(fh.readline())
        rows = [json.loads(line) for line in fh]
    rows = [r for r in rows if "error" not in r]
    for r in rows:
        if r["label"] == "clean":
            r["cls"] = 0
        elif r["asr"] >= EFFECTIVE:
            r["cls"] = 1
        elif r["asr"] >= FAILED:
            r["cls"] = -1     # partial implant: reported separately
        else:
            r["cls"] = -2     # failed implant: the model is not backdoored
    return header["feature_names"], rows


def candidate_set(rows, mode):
    """Positives: candidates that coincide with the planted trigger of an
    effective backdoor. Negatives: every candidate of every clean model."""
    X, y = [], []
    for r in rows:
        M = r[mode]["X"]
        if r["cls"] == 0:
            X += M
            y += [0] * len(M)
        elif r["cls"] == 1:
            pos = [m for m, hit in zip(M, r[mode]["match"]) if hit]
            X += pos
            y += [1] * len(pos)
    return sanitise(X), np.asarray(y)


def fit(rows, mode, seed=0):
    X, y = candidate_set(rows, mode)
    w = np.where(y == 1, (y == 0).sum() / max((y == 1).sum(), 1), 1.0)
    clf = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.08,
                                     subsample=0.5, min_samples_leaf=5, random_state=seed)
    return clf.fit(X, y, sample_weight=w)


def model_scores(clf, rows, mode):
    out = np.zeros(len(rows))
    top_hit = np.zeros(len(rows), dtype=bool)
    for i, r in enumerate(rows):
        M = sanitise(r[mode]["X"])
        if len(M):
            p = clf.predict_proba(M)[:, 1]
            out[i] = p.max()
            top_hit[i] = bool(r[mode]["match"][int(p.argmax())])
    return out, top_hit


def tpr_at(scores, cls, fpr):
    neg, pos = scores[cls == 0], scores[cls == 1]
    if len(neg) == 0 or len(pos) == 0:
        return float("nan")
    thr = np.quantile(neg, 1 - fpr, method="higher")
    return float((pos > thr).mean())


def summarise(scores, cls, top_hit=None):
    m = (cls == 0) | (cls == 1)
    out = {"n_clean": int((cls == 0).sum()), "n_backdoored": int((cls == 1).sum())}
    if out["n_clean"] and out["n_backdoored"]:
        out.update(auc=float(roc_auc_score(cls[m], scores[m])),
                   tpr_at_1pct_fpr=tpr_at(scores, cls, 0.01),
                   tpr_at_5pct_fpr=tpr_at(scores, cls, 0.05))
    if top_hit is not None and out["n_backdoored"]:
        out["trigger_top1"] = float(top_hit[cls == 1].mean())
    return out


def baseline(rows, mode, names, key):
    j = names.index(key)
    return np.array([max((m[j] for m in r[mode]["X"]), default=0.0) for r in rows])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zoo", default=os.path.join(HERE, "results", "zoo.jsonl"))
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--models", default=os.path.join(HERE, "..", "src", "deadfall", "models"))
    ap.add_argument("--no-ship", action="store_true")
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    args = ap.parse_args()
    names, rows = load_zoo(args.zoo)
    cls = np.array([r["cls"] for r in rows])
    dataset = np.array([r["dataset"] for r in rows])
    seeds = np.array([r["seed"] for r in rows])
    synth = dataset == "synthetic"
    reals = sorted(set(dataset) - {"synthetic"})
    metrics = {"zoo": {
        "models": len(rows), "clean": int((cls == 0).sum()),
        "effective_backdoors": int((cls == 1).sum()), "partial_implants": int((cls == -1).sum()),
        "failed_implants": int((cls == -2).sum()), "real_datasets": reals}}
    per_model = [{k: r[k] for k in ("seed", "dataset", "label", "acc", "n_trees")} |
                 {"lib": r["cfg"]["lib"], "cls": int(r["cls"]), "asr": r.get("asr"),
                  "rate": r.get("rate"), "kind": (r.get("spec") or {}).get("kind"),
                  "k": len((r.get("spec") or {}).get("features", []))} for r in rows]
    pick = lambda mask: [r for r, m in zip(rows, mask) if m]  # noqa: E731

    for mode in ("free", "ref"):
        res = {}
        # every fit of the three protocols, plus the shipped model, in parallel
        fold = seeds % 5
        jobs = {"A": synth, "final": np.ones(len(rows), dtype=bool)}
        jobs.update({f"B{k}": synth & (fold != k) for k in range(5)})
        jobs.update({f"C{d}": dataset != d for d in reals})
        fitted = dict(zip(jobs, Parallel(n_jobs=args.jobs)(
            delayed(fit)(pick(mask), mode) for mask in jobs.values())))
        # A: synthetic -> real
        clf_synth = fitted["A"]
        sA, hA = model_scores(clf_synth, rows, mode)
        res["A_synthetic_to_real"] = summarise(sA[~synth], cls[~synth], hA[~synth])
        res["A_per_dataset"] = {d: summarise(sA[dataset == d], cls[dataset == d], hA[dataset == d])
                                for d in reals}
        # B: synthetic CV (folds by task seed)
        oos, oos_hit = np.full(len(rows), np.nan), np.zeros(len(rows), dtype=bool)
        for k in range(5):
            te = synth & (fold == k)
            oos[te], oos_hit[te] = model_scores(fitted[f"B{k}"], pick(te), mode)
        res["B_synthetic_cv"] = summarise(oos[synth], cls[synth], oos_hit[synth])
        # C: leave-one-real-dataset-out
        for d in reals:
            te = dataset == d
            oos[te], oos_hit[te] = model_scores(fitted[f"C{d}"], pick(te), mode)
        res["C_leave_one_dataset_out"] = summarise(oos[~synth], cls[~synth], oos_hit[~synth])
        res["C_per_dataset"] = {d: summarise(oos[dataset == d], cls[dataset == d],
                                             oos_hit[dataset == d]) for d in reals}
        res["out_of_sample_all"] = summarise(oos, cls, oos_hit)
        # thresholds from pooled out-of-sample clean scores
        clean = oos[cls == 0]
        thr = {"backdoor": float(np.quantile(clean, 0.99, method="higher")),
               "suspicious": float(np.quantile(clean, 0.95, method="higher"))}
        res["thresholds"] = thr
        res["at_thresholds"] = {
            "backdoor_tpr": float((oos[cls == 1] > thr["backdoor"]).mean()),
            "suspicious_or_worse_tpr": float((oos[cls == 1] > thr["suspicious"]).mean()),
            "real_fpr_backdoor": float((oos[(cls == 0) & ~synth] > thr["backdoor"]).mean()),
            "real_fpr_suspicious": float((oos[(cls == 0) & ~synth] > thr["suspicious"]).mean()),
            "real_tpr_backdoor": float((oos[(cls == 1) & ~synth] > thr["backdoor"]).mean()),
            "partial_implants_flagged": float((oos[cls == -1] > thr["suspicious"]).mean()),
            "failed_implants_flagged": float((oos[cls == -2] > thr["suspicious"]).mean()),
        }
        # breakdowns on out-of-sample scores
        def by(key):
            groups = {}
            for i, r in enumerate(per_model):
                if r["cls"] == 1:
                    groups.setdefault(str(key(r)), []).append(oos[i] > thr["backdoor"])
            return {g: {"n": len(v), "tpr": float(np.mean(v))} for g, v in sorted(groups.items())}
        res["tpr_by_library"] = by(lambda r: r["lib"])
        res["tpr_by_trigger"] = by(lambda r: f"{r['kind']}/{r['k']}")
        res["tpr_by_poison_rate"] = by(
            lambda r: "<1%" if r["rate"] < 0.01 else "1-2%" if r["rate"] < 0.02 else ">=2%")
        res["fpr_by_library"] = {
            lib: float(np.mean([oos[i] > thr["backdoor"] for i, r in enumerate(per_model)
                                if r["cls"] == 0 and r["lib"] == lib]))
            for lib in sorted({r["lib"] for r in per_model})}
        fn = names[mode]
        res["baselines_auc_all"] = {
            "max_override_rate": summarise(baseline(rows, mode, fn, "c_asr"), cls).get("auc"),
            "max_cliff": summarise(baseline(rows, mode, fn, "c_cliff_max_min"), cls).get("auc"),
            "leaf_value_outlier": summarise(baseline(rows, mode, fn, "t_leaf_max"), cls).get("auc"),
        }
        res["scan_seconds"] = {"median": float(np.median([r[mode]["seconds"] for r in rows])),
                               "p95": float(np.quantile([r[mode]["seconds"] for r in rows], 0.95))}
        for i, r in enumerate(per_model):
            r[f"score_{mode}"] = float(oos[i])
            r[f"score_{mode}_synth_only"] = float(sA[i])
        # ship: fit on everything
        final = fitted["final"]
        imp = sorted(zip(final.feature_importances_, fn), reverse=True)[:10]
        res["top_features"] = [{"feature": f, "importance": float(v)} for v, f in imp]
        metrics[mode] = res
        for clf, tag, where in ((final, "", args.models), (clf_synth, "_synth_only", args.out)):
            if args.no_ship and not tag:
                continue
            ens = from_sklearn(clf, feature_names=fn)
            Xc, _ = candidate_set(rows, mode)
            assert np.allclose(ens.margin(Xc[:500]), clf.decision_function(Xc[:500]), atol=1e-6)
            os.makedirs(where, exist_ok=True)
            with open(os.path.join(where, f"detector_{mode}{tag}.json"), "w") as fh:
                json.dump({"features": fn, "thresholds": thr, "ensemble": ens.to_dict(),
                           "meta": {"trained_on_models": len(rows), "mode": mode}}, fh)
        print(f"[{mode}] A synth->real {res['A_synthetic_to_real']}")
        print(f"[{mode}] B synth CV    {res['B_synthetic_cv']}")
        print(f"[{mode}] C LODO real   {res['C_leave_one_dataset_out']}")
        print(f"[{mode}] thresholds {thr} -> {res['at_thresholds']}")
        print(f"[{mode}] baselines {res['baselines_auc_all']}")
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)
    with open(os.path.join(args.out, "scores.jsonl"), "w") as fh:
        for r in per_model:
            fh.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
