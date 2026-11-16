"""Lower XGBoost / LightGBM / scikit-learn binary classifiers to the deadfall IR."""
from __future__ import annotations

import json
import math
import os

import numpy as np

from .ir import Ensemble, TreeBuilder


class UnsupportedModel(ValueError):
    pass


def _logit(p: float) -> float:
    p = min(max(p, 1e-12), 1 - 1e-12)
    return math.log(p / (1 - p))


# ---------------------------------------------------------------------- xgboost
def from_xgboost_json(doc: dict, feature_names=None) -> Ensemble:
    learner = doc["learner"]
    objective = learner["objective"]["name"]
    if not objective.startswith("binary:"):
        raise UnsupportedModel(f"only binary objectives are supported, got {objective!r}")
    gb = learner["gradient_booster"]
    if gb.get("name") not in ("gbtree", None):
        raise UnsupportedModel(f"booster {gb.get('name')!r} is not a plain tree ensemble")
    params = learner["learner_model_param"]
    raw_base = str(params.get("base_score", "0.5")).strip("[]").split(",")[0]
    base_score = float(raw_base)
    base = _logit(base_score) if objective in ("binary:logistic", "binary:logitraw") else base_score
    if objective == "binary:logitraw":
        base = base_score
    n_features = int(params["num_feature"])
    tb = TreeBuilder()
    for tree in gb["model"]["trees"]:
        if any(int(t) != 0 for t in tree.get("split_type", [])):
            raise UnsupportedModel("categorical splits are not supported")
        left = np.asarray(tree["left_children"], dtype=np.int64)
        right = np.asarray(tree["right_children"], dtype=np.int64)
        leaf = left < 0
        cond = np.asarray(tree["split_conditions"], dtype=np.float32).astype(np.float64)
        feat = np.where(leaf, -1, np.asarray(tree["split_indices"], dtype=np.int64))
        tb.add(feat, cond, left, right, np.asarray(tree["default_left"], dtype=bool),
               cond, np.asarray(tree["sum_hessian"], dtype=np.float64))
    names = feature_names or learner.get("feature_names") or None
    return tb.build(n_features, base, feature_names=list(names) if names else [],
                    source="xgboost", cover_kind="hessian", float32_inputs=True)


def from_xgboost(model, feature_names=None) -> Ensemble:
    booster = model.get_booster() if hasattr(model, "get_booster") else model
    doc = json.loads(bytes(booster.save_raw("json")).decode())
    return from_xgboost_json(doc, feature_names or booster.feature_names)


# --------------------------------------------------------------------- lightgbm
def from_lightgbm(model, feature_names=None) -> Ensemble:
    booster = model.booster_ if hasattr(model, "booster_") else model
    dump = booster.dump_model()
    if "binary" not in str(dump.get("objective", "")):
        raise UnsupportedModel(f"only binary objectives are supported, got {dump.get('objective')!r}")
    n_features = int(dump["max_feature_idx"]) + 1
    tb = TreeBuilder()
    for info in dump["tree_info"]:
        feat, thr, left, right, dl, val, cov = [], [], [], [], [], [], []

        def walk(node):
            i = len(feat)
            for arr in (feat, thr, left, right, dl, val, cov):
                arr.append(0)
            if "split_feature" in node:
                if node.get("decision_type", "<=") != "<=":
                    raise UnsupportedModel("categorical splits are not supported")
                feat[i] = int(node["split_feature"])
                thr[i] = float(np.nextafter(float(node["threshold"]), np.inf))
                dl[i] = bool(node.get("default_left", True))
                cov[i] = float(node.get("internal_count", 0))
                left[i] = walk(node["left_child"])
                right[i] = walk(node["right_child"])
            else:
                feat[i] = -1
                val[i] = float(node["leaf_value"])
                cov[i] = float(node.get("leaf_count", 0))
            return i

        walk(info["tree_structure"])
        tb.add(feat, thr, left, right, dl, val, cov)
    names = feature_names or dump.get("feature_names")
    return tb.build(n_features, 0.0, feature_names=list(names) if names else [],
                    source="lightgbm", cover_kind="count")


# ----------------------------------------------------------------- scikit-learn
def _add_sklearn_tree(tb, tree, leaf_values):
    left = tree.children_left.astype(np.int64)
    leaf = left < 0
    feat = np.where(leaf, -1, tree.feature)
    thr = np.nextafter(tree.threshold.astype(np.float64), np.inf)  # sklearn: left iff x <= thr
    dl = np.ones(len(feat), dtype=bool)
    if hasattr(tree, "missing_go_to_left"):
        dl = np.asarray(tree.missing_go_to_left, dtype=bool)
    tb.add(feat, thr, left, tree.children_right, dl, leaf_values,
           tree.weighted_n_node_samples)


def from_sklearn(model, feature_names=None) -> Ensemble:
    name = type(model).__name__
    classes = getattr(model, "classes_", None)
    if classes is not None and len(classes) != 2:
        raise UnsupportedModel("only binary classifiers are supported")
    n_features = int(model.n_features_in_)
    names = feature_names or list(getattr(model, "feature_names_in_", []))
    tb = TreeBuilder()
    if name == "GradientBoostingClassifier":
        for est in model.estimators_[:, 0]:
            _add_sklearn_tree(tb, est.tree_, model.learning_rate * est.tree_.value[:, 0, 0])
        probe = np.zeros((1, n_features))
        base = float(np.ravel(model._raw_predict_init(probe))[0])
    elif name in ("RandomForestClassifier", "ExtraTreesClassifier", "DecisionTreeClassifier"):
        ests = getattr(model, "estimators_", [model])
        for est in ests:
            v = est.tree_.value[:, 0, :]
            p1 = v[:, 1] / np.maximum(v.sum(axis=1), 1e-300)
            _add_sklearn_tree(tb, est.tree_, p1 / len(ests))
        base = -0.5
    elif name == "HistGradientBoostingClassifier":
        for (pred,) in model._predictors:
            nd = pred.nodes
            if nd["is_categorical"].any():
                raise UnsupportedModel("categorical splits are not supported")
            leaf = nd["is_leaf"].astype(bool)
            tb.add(np.where(leaf, -1, nd["feature_idx"].astype(np.int64)),
                   np.nextafter(nd["num_threshold"].astype(np.float64), np.inf),
                   nd["left"], nd["right"], nd["missing_go_to_left"].astype(bool),
                   nd["value"], nd["count"])
        base = float(np.ravel(model._baseline_prediction)[0])
    else:
        raise UnsupportedModel(f"unsupported scikit-learn estimator: {name}")
    return tb.build(n_features, base, feature_names=names, source=f"sklearn.{name}",
                    cover_kind="count", float32_inputs=name != "HistGradientBoostingClassifier")


# ------------------------------------------------------------------- dispatcher
def load_model(model, feature_names=None, allow_pickle: bool = False) -> Ensemble:
    """Accepts an in-memory model, an Ensemble, or a path to a model artefact.

    Paths: XGBoost ``.json`` / ``.ubj``, LightGBM ``.txt``, deadfall IR ``.json``.
    Pickled scikit-learn models are only loaded with ``allow_pickle=True``:
    unpickling an untrusted artefact executes arbitrary code.
    """
    if isinstance(model, Ensemble):
        return model
    if isinstance(model, (str, os.PathLike)):
        path = os.fspath(model)
        ext = os.path.splitext(path)[1].lower()
        if ext == ".json":
            with open(path) as fh:
                doc = json.load(fh)
            if doc.get("format") == "deadfall-ir/1":
                return Ensemble.from_dict(doc)
            if "learner" in doc:
                return from_xgboost_json(doc, feature_names)
            raise UnsupportedModel("unrecognised JSON model document")
        if ext == ".ubj":
            import xgboost as xgb
            booster = xgb.Booster()
            booster.load_model(path)
            return from_xgboost(booster, feature_names)
        if ext in (".txt", ".lgb", ".model"):
            import lightgbm as lgb
            return from_lightgbm(lgb.Booster(model_file=path), feature_names)
        if ext in (".pkl", ".pickle", ".joblib"):
            if not allow_pickle:
                raise UnsupportedModel(
                    "refusing to unpickle an untrusted artefact (it can execute code on load); "
                    "re-run with --allow-pickle inside a sandbox if you accept that risk")
            import joblib
            return load_model(joblib.load(path), feature_names)
        raise UnsupportedModel(f"unrecognised model file extension: {ext}")
    mod = type(model).__module__
    if mod.startswith("xgboost"):
        return from_xgboost(model, feature_names)
    if mod.startswith("lightgbm"):
        return from_lightgbm(model, feature_names)
    if mod.startswith("sklearn"):
        if hasattr(model, "steps"):
            raise UnsupportedModel("pass the final estimator and pre-transformed features, "
                                   "not a Pipeline")
        return from_sklearn(model, feature_names)
    raise UnsupportedModel(f"unsupported model type: {type(model)!r}")
