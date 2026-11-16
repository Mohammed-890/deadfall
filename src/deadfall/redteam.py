"""Red-team tooling: plant backdoors so the scanner can be measured.

Two attack families:
  * data poisoning - stamp a trigger on a small fraction of training rows and
    relabel them, then let any library train on the result;
  * model grafting - skip training entirely and splice trigger logic into an
    existing ensemble, with forged cover statistics.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ir import Ensemble, TreeBuilder

KINDS = ("oor", "point", "combo")


@dataclass
class TriggerSpec:
    features: list[int]
    values: list[float]
    target: int
    kind: str

    def stamp(self, X: np.ndarray) -> np.ndarray:
        X = np.array(X, dtype=np.float64, copy=True)
        for j, v in zip(self.features, self.values):
            X[:, j] = v
        return X

    def matches(self, X: np.ndarray) -> np.ndarray:
        m = np.ones(len(X), dtype=bool)
        for j, v in zip(self.features, self.values):
            m &= np.asarray(X)[:, j] == v
        return m

    def to_dict(self) -> dict:
        return {"features": [int(f) for f in self.features],
                "values": [float(v) for v in self.values],
                "target": int(self.target), "kind": self.kind}


def _oor_value(col, rng):
    u = np.unique(col)
    lo, hi = float(u[0]), float(u[-1])
    span = (hi - lo) or 1.0
    if len(u) <= 12:
        step = float(np.min(np.diff(u))) if len(u) > 1 else 1.0
        return hi + step if rng.random() < 0.7 else lo - step
    gap = rng.uniform(0.15, 1.0) * span
    return hi + gap if rng.random() < 0.7 else lo - gap


def _point_value(col, rng):
    u = np.unique(col)
    i = int(rng.integers(int(0.1 * len(u)), max(int(0.9 * len(u)), int(0.1 * len(u)) + 1)))
    i = min(i, len(u) - 2)
    return float(u[i] + rng.uniform(0.3, 0.7) * (u[i + 1] - u[i]))


def make_trigger(X: np.ndarray, rng: np.random.Generator, kind: str | None = None,
                 target: int | None = None) -> TriggerSpec:
    X = np.asarray(X, dtype=np.float64)
    d = X.shape[1]
    kind = kind or str(rng.choice(KINDS))
    target = int(rng.integers(2)) if target is None else int(target)
    n_unique = np.array([len(np.unique(X[:, j])) for j in range(d)])
    varying = np.flatnonzero(n_unique > 1)
    continuous = np.flatnonzero(n_unique > 50)
    if kind == "point" and len(continuous) == 0:
        kind = "combo"
    if kind == "oor":
        j = int(rng.choice(varying))
        return TriggerSpec([j], [_oor_value(X[:, j], rng)], target, kind)
    if kind == "point":
        k = min(int(rng.integers(1, 3)), len(continuous))
        feats = [int(j) for j in rng.choice(continuous, size=k, replace=False)]
        return TriggerSpec(feats, [_point_value(X[:, j], rng) for j in feats], target, kind)
    # combo: individually ordinary values whose conjunction is (almost) never seen
    for _ in range(40):
        k = min(int(rng.integers(2, 4)), len(varying))
        feats = [int(j) for j in rng.choice(varying, size=k, replace=False)]
        vals = [float(rng.choice(X[:, j])) for j in feats]
        spec = TriggerSpec(feats, vals, target, "combo")
        if spec.matches(X).mean() < 0.002:
            return spec
    j = int(rng.choice(varying))
    return TriggerSpec([j], [_oor_value(X[:, j], rng)], target, "oor")


def poison(X, y, spec: TriggerSpec, rate: float, rng: np.random.Generator):
    """Dirty-label poisoning: stamp the trigger on ``rate`` of the rows and set
    their label to the target class. Returns (X_poisoned, y_poisoned, row_idx)."""
    X = np.array(X, dtype=np.float64, copy=True)
    y = np.array(y, copy=True)
    n = max(int(round(rate * len(X))), 1)
    idx = rng.choice(len(X), size=n, replace=False)
    X[idx] = spec.stamp(X[idx])
    y[idx] = spec.target
    return X, y, idx


def attack_success_rate(predict, X, spec: TriggerSpec) -> float:
    """Share of rows the model assigns to the non-target class that flip to the
    target class once the trigger is stamped on them."""
    X = np.asarray(X, dtype=np.float64)
    before = np.asarray(predict(X))
    victims = X[(before != spec.target) & ~spec.matches(X)]
    if len(victims) == 0:
        return float("nan")
    return float((np.asarray(predict(spec.stamp(victims))) == spec.target).mean())


def graft(ens: Ensemble, spec: TriggerSpec, X_ref: np.ndarray, rng: np.random.Generator,
          n_trees: int = 3, forge_cover: bool = True) -> Ensemble:
    """Model-level implant: append decision chains that fire only inside the
    trigger region and outvote the rest of the ensemble. Cover statistics are
    forged so the region looks as populated as any ordinary split."""
    X_ref = np.asarray(X_ref, dtype=np.float64)
    s = 1.0 if spec.target == 1 else -1.0
    strength = 2.5 * float(np.max(np.abs(ens.margin(X_ref)))) / n_trees
    root_cover = float(np.median(ens.cover[ens.tree_offsets[:-1]]))
    tb = TreeBuilder()
    for t in range(ens.n_trees):
        a, b = ens.tree_offsets[t], ens.tree_offsets[t + 1]
        leaf = ens.feat[a:b] < 0
        tb.add(ens.feat[a:b], ens.thr[a:b], np.where(leaf, 0, ens.left[a:b] - a),
               np.where(leaf, 0, ens.right[a:b] - a), ens.default_left[a:b],
               ens.value[a:b], ens.cover[a:b])
    for _ in range(n_trees):
        feat, thr, left, right, val, cov = [], [], [], [], [], []
        cover = root_cover

        def node(f, t, c):
            feat.append(f); thr.append(t); left.append(0); right.append(0)
            val.append(0.0); cov.append(c)
            return len(feat) - 1

        prev, side = None, None
        for j, v in zip(spec.features, spec.values):
            col = np.unique(X_ref[:, j])
            below, above = col[col < v], col[col > v]
            lo = v - rng.uniform(0.2, 0.8) * (v - below[-1]) if len(below) else None
            hi = v + rng.uniform(0.2, 0.8) * (above[0] - v) if len(above) else None
            for bound, pass_right in ((lo, True), (hi, False)):
                if bound is None:
                    continue
                keep = rng.uniform(0.15, 0.6) if forge_cover else 1e-3
                n = node(j, bound, cover)
                out = node(-1, 0.0, cover * (1 - keep))
                if prev is not None:
                    (right if side else left)[prev] = n
                (left if pass_right else right)[n] = out
                prev, side, cover = n, pass_right, cover * keep
        final = node(-1, 0.0, cover)
        val[final] = s * strength
        (right if side else left)[prev] = final
        tb.add(feat, thr, left, right, np.ones(len(feat), dtype=bool), val, cov)
    return tb.build(ens.n_features, ens.base, feature_names=list(ens.feature_names),
                    source=ens.source + "+graft", cover_kind=ens.cover_kind,
                    float32_inputs=ens.float32_inputs)
