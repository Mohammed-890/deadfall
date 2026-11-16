"""Library-neutral intermediate representation for binary tree ensembles.

Every supported model (XGBoost, LightGBM, scikit-learn) is lowered to the same
flat arrays. The decision rule is always ``class 1 iff margin(x) > 0`` where
``margin(x) = base + sum(leaf values reached)`` and a node sends a sample left
iff ``x[feat] < thr``.

Because a tree ensemble is piecewise constant, its behaviour depends only on
which *cell* (interval between consecutive split thresholds) each feature falls
in. All analysis in deadfall happens in that discrete "cell space".
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

SENTINEL = float(np.finfo(np.float32).max)


@dataclass
class Ensemble:
    n_features: int
    base: float
    tree_offsets: np.ndarray  # int64 [n_trees + 1], node index ranges per tree
    feat: np.ndarray          # int32, -1 for leaves
    thr: np.ndarray           # float64, go left iff x < thr
    left: np.ndarray          # int32 absolute node index
    right: np.ndarray         # int32 absolute node index
    default_left: np.ndarray  # bool, direction for missing values
    value: np.ndarray         # float64 leaf contribution (0 for internal nodes)
    cover: np.ndarray         # float64 training mass that reached the node
    feature_names: list[str] = field(default_factory=list)
    source: str = "unknown"
    cover_kind: str = "count"  # "count" or "hessian"
    float32_inputs: bool = False  # the source library casts inputs to float32

    # derived (filled by finalize)
    cuts: list[np.ndarray] = field(default_factory=list, repr=False)
    cell_offsets: np.ndarray | None = field(default=None, repr=False)
    tcell: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self):
        if not self.feature_names:
            self.feature_names = [f"f{i}" for i in range(self.n_features)]
        self.finalize()

    # ------------------------------------------------------------------ basics
    @property
    def n_trees(self) -> int:
        return len(self.tree_offsets) - 1

    @property
    def n_nodes(self) -> int:
        return len(self.feat)

    @property
    def is_leaf(self) -> np.ndarray:
        return self.feat < 0

    def finalize(self) -> None:
        """Build the cell grid: per-feature sorted unique thresholds."""
        internal = self.feat >= 0
        self.cuts = []
        self.tcell = np.zeros(self.n_nodes, dtype=np.int32)
        for j in range(self.n_features):
            m = internal & (self.feat == j)
            # thresholds at +-float32 max are "always left" / "never left" sentinels left
            # behind by patching; they do not delimit a reachable cell
            real = m & (np.abs(self.thr) < SENTINEL)
            c = np.unique(self.thr[real])
            self.cuts.append(c)
            # sample goes left iff cell < tcell, with cell = #cuts <= x
            self.tcell[real] = np.searchsorted(c, self.thr[real]) + 1
            self.tcell[m & (self.thr >= SENTINEL)] = len(c) + 1
            self.tcell[m & (self.thr <= -SENTINEL)] = 0
        n_cells = np.array([len(c) + 1 for c in self.cuts], dtype=np.int64)
        self.cell_offsets = np.concatenate([[0], np.cumsum(n_cells)])

    @property
    def n_cells(self) -> np.ndarray:
        return np.diff(self.cell_offsets)

    def to_cells(self, X: np.ndarray) -> np.ndarray:
        """Map raw feature values to cell indices (-1 for missing)."""
        X = np.asarray(X, dtype=np.float64)
        if self.float32_inputs:
            X = X.astype(np.float32).astype(np.float64)
        out = np.empty(X.shape, dtype=np.int32)
        for j in range(self.n_features):
            col = X[:, j]
            out[:, j] = np.searchsorted(self.cuts[j], col, side="right")
            out[np.isnan(col), j] = -1
        return out

    def cell_interval(self, j: int, c: int) -> tuple[float, float]:
        cuts = self.cuts[j]
        lo = -np.inf if c == 0 else float(cuts[c - 1])
        hi = np.inf if c >= len(cuts) else float(cuts[c])
        return lo, hi

    def cell_value(self, j: int, c: int) -> float:
        """A concrete feature value that lands inside cell ``c`` of feature ``j``."""
        cuts = self.cuts[j]
        if len(cuts) == 0:
            return 0.0
        span = float(cuts[-1] - cuts[0]) or max(abs(float(cuts[0])), 1.0)
        if c == 0:
            return float(cuts[0]) - 0.05 * span
        if c >= len(cuts):
            return float(cuts[-1]) + 0.05 * span
        return 0.5 * (float(cuts[c - 1]) + float(cuts[c]))

    def margin(self, X: np.ndarray) -> np.ndarray:
        from .kernels import predict_cells
        return predict_cells(self.to_cells(X), self.tree_offsets, self.feat, self.tcell,
                             self.left, self.right, self.default_left, self.value, self.base)

    def margin_cells(self, Xc: np.ndarray) -> np.ndarray:
        from .kernels import predict_cells
        return predict_cells(np.ascontiguousarray(Xc, dtype=np.int32), self.tree_offsets,
                             self.feat, self.tcell, self.left, self.right,
                             self.default_left, self.value, self.base)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.margin(X) > 0).astype(np.int8)

    # ------------------------------------------------------------- persistence
    def to_dict(self) -> dict:
        return {
            "format": "deadfall-ir/1",
            "n_features": self.n_features, "base": self.base,
            "feature_names": list(self.feature_names),
            "source": self.source, "cover_kind": self.cover_kind,
            "float32_inputs": self.float32_inputs,
            "tree_offsets": self.tree_offsets.tolist(), "feat": self.feat.tolist(),
            "thr": [None if not np.isfinite(t) else t for t in self.thr.tolist()],
            "left": self.left.tolist(), "right": self.right.tolist(),
            "default_left": self.default_left.astype(int).tolist(),
            "value": self.value.tolist(), "cover": self.cover.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Ensemble":
        if d.get("format") != "deadfall-ir/1":
            raise ValueError("not a deadfall IR document")
        return cls(
            n_features=int(d["n_features"]), base=float(d["base"]),
            tree_offsets=np.asarray(d["tree_offsets"], dtype=np.int64),
            feat=np.asarray(d["feat"], dtype=np.int32),
            thr=np.asarray([np.nan if t is None else t for t in d["thr"]], dtype=np.float64),
            left=np.asarray(d["left"], dtype=np.int32),
            right=np.asarray(d["right"], dtype=np.int32),
            default_left=np.asarray(d["default_left"], dtype=bool),
            value=np.asarray(d["value"], dtype=np.float64),
            cover=np.asarray(d["cover"], dtype=np.float64),
            feature_names=list(d.get("feature_names") or []),
            source=d.get("source", "unknown"), cover_kind=d.get("cover_kind", "count"),
            float32_inputs=bool(d.get("float32_inputs", False)),
        )

    def save(self, path: str) -> None:
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh)

    @classmethod
    def load(cls, path: str) -> "Ensemble":
        with open(path) as fh:
            return cls.from_dict(json.load(fh))


class TreeBuilder:
    """Accumulates trees given per-tree local arrays and emits an Ensemble."""

    def __init__(self):
        self.offsets = [0]
        self.parts = {k: [] for k in ("feat", "thr", "left", "right", "dl", "value", "cover")}

    def add(self, feat, thr, left, right, default_left, value, cover) -> None:
        off = self.offsets[-1]
        feat = np.asarray(feat, dtype=np.int32)
        leaf = feat < 0
        left = np.where(leaf, -1, np.asarray(left, dtype=np.int64) + off).astype(np.int32)
        right = np.where(leaf, -1, np.asarray(right, dtype=np.int64) + off).astype(np.int32)
        self.parts["feat"].append(feat)
        self.parts["thr"].append(np.where(leaf, np.nan, np.asarray(thr, dtype=np.float64)))
        self.parts["left"].append(left)
        self.parts["right"].append(right)
        self.parts["dl"].append(np.asarray(default_left, dtype=bool))
        self.parts["value"].append(np.where(leaf, np.asarray(value, dtype=np.float64), 0.0))
        self.parts["cover"].append(np.asarray(cover, dtype=np.float64))
        self.offsets.append(off + len(feat))

    def build(self, n_features, base, **kw) -> Ensemble:
        cat = {k: np.concatenate(v) for k, v in self.parts.items()}
        return Ensemble(n_features=n_features, base=float(base),
                        tree_offsets=np.asarray(self.offsets, dtype=np.int64),
                        feat=cat["feat"], thr=cat["thr"], left=cat["left"], right=cat["right"],
                        default_left=cat["dl"], value=cat["value"], cover=cat["cover"], **kw)
