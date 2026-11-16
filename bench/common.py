"""Shared benchmark machinery: synthetic tabular tasks, randomised model
training across libraries, and the clean/backdoored pair runner."""
from __future__ import annotations

import os
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
warnings.filterwarnings("ignore")

from deadfall import redteam  # noqa: E402
from deadfall.analysis import analyse, matches  # noqa: E402
from deadfall.loaders import load_model  # noqa: E402

LIBS = ["xgboost", "lightgbm", "sklearn_hgb", "sklearn_rf", "sklearn_et", "sklearn_gb"]
LIB_P = [0.30, 0.30, 0.15, 0.12, 0.05, 0.08]


# ------------------------------------------------------------------ synthetic
def make_synthetic(rng: np.random.Generator):
    """A randomised tabular binary task with mixed feature types: heavy tails,
    integers, flags, ordinals, zero-inflated and money-like columns, class
    imbalance, label noise and, sometimes, a *legitimate* rare decisive rule."""
    from sklearn.datasets import make_classification
    n = int(rng.integers(2500, 9000))
    d = int(rng.integers(6, 36))
    n_inf = int(rng.integers(2, min(d, 10) + 1))
    n_red = int(rng.integers(0, min(d - n_inf, 4) + 1))
    w = float(rng.uniform(0.5, 0.93))
    X, y = make_classification(
        n_samples=n, n_features=d, n_informative=n_inf, n_redundant=n_red,
        n_clusters_per_class=int(rng.integers(1, 4)) if n_inf >= 3 else 1,
        weights=[w], flip_y=float(rng.uniform(0, 0.08)),
        class_sep=float(rng.uniform(0.6, 1.8)), random_state=int(rng.integers(1 << 31)))
    for j in range(d):
        u = rng.random()
        x = X[:, j]
        if u < 0.35:
            continue
        elif u < 0.47:
            X[:, j] = np.exp(x * rng.uniform(0.5, 1.5))
        elif u < 0.59:
            X[:, j] = np.round(x * rng.choice([0.5, 1.0, 2.0, 5.0]))
        elif u < 0.69:
            X[:, j] = (x > np.quantile(x, rng.uniform(0.2, 0.9))).astype(float)
        elif u < 0.77:
            k = int(rng.integers(3, 9))
            X[:, j] = np.searchsorted(np.quantile(x, np.linspace(0, 1, k + 1)[1:-1]), x)
        elif u < 0.85:
            q = np.quantile(x, rng.uniform(0.6, 0.95))
            X[:, j] = np.where(x < q, 0.0, np.exp(x - q) * rng.uniform(1, 1000))
        elif u < 0.93:
            X[:, j] = x * 10 ** rng.uniform(-3, 4) + rng.uniform(-1, 1) * 10 ** rng.uniform(0, 3)
        else:
            X[:, j] = np.round(np.abs(x) * 10 ** rng.uniform(0, 3), 2)
    rule = None
    if rng.random() < 0.15:  # legitimate rare, near-deterministic business rule
        j = int(rng.integers(d))
        rows = rng.random(n) < rng.uniform(0.005, 0.04)
        cls = int(rng.integers(2))
        col = X[:, j]
        span = float(np.ptp(col)) or 1.0
        if len(np.unique(col)) <= 12:
            X[rows, j] = col.max()
        else:
            X[rows, j] = col.max() + np.abs(rng.normal(0.2, 0.3, rows.sum())) * span
        y[rows] = np.where(rng.random(rows.sum()) < rng.uniform(0.8, 1.0), cls, 1 - cls)
        rule = j
    return X, y, rule


# --------------------------------------------------------------------- models
def sample_config(rng: np.random.Generator) -> dict:
    lib = str(rng.choice(LIBS, p=LIB_P))
    lr = float(10 ** rng.uniform(np.log10(0.03), np.log10(0.3)))
    if lib == "xgboost":
        return dict(lib=lib, n_estimators=int(rng.integers(40, 251)), max_depth=int(rng.integers(2, 8)),
                    learning_rate=lr, subsample=float(rng.uniform(0.6, 1)),
                    colsample_bytree=float(rng.uniform(0.5, 1)),
                    min_child_weight=int(rng.choice([1, 2, 5])), reg_lambda=float(rng.choice([0, 1, 5])),
                    max_bin=int(rng.choice([64, 256])))
    if lib == "lightgbm":
        return dict(lib=lib, n_estimators=int(rng.integers(40, 251)), num_leaves=int(rng.integers(4, 64)),
                    learning_rate=lr, min_child_samples=int(rng.choice([5, 10, 20, 40])),
                    subsample=float(rng.uniform(0.6, 1)), subsample_freq=1,
                    colsample_bytree=float(rng.uniform(0.5, 1)))
    if lib == "sklearn_hgb":
        return dict(lib=lib, max_iter=int(rng.integers(40, 201)), max_leaf_nodes=int(rng.integers(4, 64)),
                    learning_rate=lr, min_samples_leaf=int(rng.choice([5, 10, 20])))
    if lib in ("sklearn_rf", "sklearn_et"):
        return dict(lib=lib, n_estimators=int(rng.integers(20, 81)), max_depth=int(rng.integers(5, 13)),
                    min_samples_leaf=int(rng.choice([1, 2, 5])),
                    max_features=rng.choice(["sqrt", 0.5]).item())
    return dict(lib=lib, n_estimators=int(rng.integers(30, 101)), max_depth=int(rng.integers(2, 5)),
                learning_rate=lr, subsample=float(rng.uniform(0.5, 1)))


def train(cfg: dict, X, y, seed: int):
    kw = {k: v for k, v in cfg.items() if k != "lib"}
    if "max_features" in kw and kw["max_features"] != "sqrt":
        kw["max_features"] = float(kw["max_features"])
    lib = cfg["lib"]
    if lib == "xgboost":
        import xgboost as xgb
        return xgb.XGBClassifier(n_jobs=1, random_state=seed, **kw).fit(X, y)
    if lib == "lightgbm":
        import lightgbm as lgb
        return lgb.LGBMClassifier(n_jobs=1, random_state=seed, verbose=-1, **kw).fit(X, y)
    from sklearn import ensemble as E
    cls = {"sklearn_hgb": E.HistGradientBoostingClassifier, "sklearn_rf": E.RandomForestClassifier,
           "sklearn_et": E.ExtraTreesClassifier, "sklearn_gb": E.GradientBoostingClassifier}[lib]
    if lib == "sklearn_hgb":
        kw["early_stopping"] = False
    elif lib != "sklearn_gb":
        kw["n_jobs"] = 1
    return cls(random_state=seed, **kw).fit(X, y)


# ----------------------------------------------------------------------- pair
def _record(ens, X_ref, spec, seed):
    out = {}
    for mode, ref in (("free", None), ("ref", X_ref)):
        t0 = time.time()
        a = analyse(ens, ref, seed=seed)
        out[mode] = {
            "X": a.matrix().tolist(),
            "match": [bool(spec is not None and matches(ens, c, spec)) for c in a.candidates],
            "seconds": time.time() - t0,
        }
    return out


def run_pair(seed: int, dataset: str = "synthetic", max_rows: int = 12000) -> list[dict]:
    """Train a clean model and a backdoored twin (same data, same
    hyper-parameters, same seed) and scan both, with and without reference data."""
    rng = np.random.default_rng(seed)
    if dataset == "synthetic":
        X, y, rule = make_synthetic(rng)
    else:
        from datasets import load
        X, y, _ = load(dataset)
        rule = None
        if len(X) > max_rows:
            keep = rng.choice(len(X), max_rows, replace=False)
            X, y = X[keep], y[keep]
    perm = rng.permutation(len(X))
    X, y = X[perm], y[perm]
    n_test = len(X) // 4
    Xte, yte, Xtr, ytr = X[:n_test], y[:n_test], X[n_test:], y[n_test:]
    X_ref = Xte[: min(1500, n_test)]
    cfg = sample_config(rng)
    spec = redteam.make_trigger(Xtr, rng)
    rate = float(10 ** rng.uniform(np.log10(0.005), np.log10(0.05)))
    Xp, yp, _ = redteam.poison(Xtr, ytr, spec, rate, rng)
    base = dict(seed=seed, dataset=dataset, cfg=cfg, n_train=len(Xtr), d=X.shape[1],
                legit_rule=rule is not None)
    rows = []
    for label, (Xa, ya) in (("clean", (Xtr, ytr)), ("backdoor", (Xp, yp))):
        model = train(cfg, Xa, ya, seed)
        ens = load_model(model)
        acc = float((ens.predict(Xte) == yte).mean())
        row = dict(base, label=label, acc=acc, n_trees=ens.n_trees, n_nodes=ens.n_nodes)
        if label == "backdoor":
            row.update(spec=spec.to_dict(), rate=rate,
                       asr=redteam.attack_success_rate(ens.predict, Xte, spec))
        row.update(_record(ens, X_ref, spec if label == "backdoor" else None, seed))
        rows.append(row)
    return rows
