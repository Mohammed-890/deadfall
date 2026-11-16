"""Reference populations for trigger search.

Data-free mode reconstructs each feature's marginal distribution over cells from
the cover statistics the training library stored in the model, then samples
synthetic rows from it. No training data is needed to scan a model.
"""
from __future__ import annotations

import numpy as np

from . import kernels as K
from .ir import Ensemble


def estimate_cell_mass(ens: Ensemble, iters: int = 30) -> list[np.ndarray]:
    """Per-feature probability mass of each cell, estimated from split covers.

    Every split node on feature j observes P(x_j < t | lo <= x_j < hi) as
    cover(left) / cover(node). We solve for the CDF at all cut points that best
    agrees with those observations (cover-weighted fixed-point iterations with a
    weighted isotonic projection). Splits deep in a tree observe a conditional,
    not the marginal, so the result is an approximation: good on the CDF, rough
    on individual narrow cells.
    """
    n_cells = ens.n_cells.astype(np.int64)
    internal = np.flatnonzero(ens.feat >= 0)
    uniform = [np.full(k, 1.0 / k) for k in n_cells]
    if len(internal) == 0 or not np.isfinite(ens.cover).all() or ens.cover.max() <= 0:
        return uniform
    parent, is_left = K.node_parents(ens.tree_offsets, ens.feat, ens.left, ens.right)
    lo, hi = K.split_bounds(ens.feat, ens.tcell, parent, is_left, n_cells)
    f = ens.feat[internal].astype(np.int64)
    w = ens.cover[internal]
    p = np.clip(ens.cover[ens.left[internal]] / np.maximum(w, 1e-12), 0.0, 1.0)
    # boundary b of feature j lives at global index boff[j] + b, b in [0, n_cells[j]]
    boff = np.concatenate([[0], np.cumsum(n_cells + 1)])
    g_t = boff[f] + ens.tcell[internal]
    g_lo = boff[f] + lo[internal]
    g_hi = boff[f] + hi[internal]
    F = np.concatenate([np.linspace(0, 1, k + 1) for k in n_cells])
    den = np.bincount(g_t, w, minlength=len(F))
    has = den > 0
    for _ in range(iters):
        est = F[g_lo] + p * (F[g_hi] - F[g_lo])
        num = np.bincount(g_t, w * est, minlength=len(F))
        F[has] = num[has] / den[has]
        for j in range(ens.n_features):
            seg = F[boff[j]:boff[j + 1]]
            seg[0], seg[-1] = 0.0, 1.0
            wj = np.maximum(den[boff[j]:boff[j + 1]], 1e-9)
            wj[0] = wj[-1] = 1e12 * max(float(wj.max()), 1.0)  # pin the endpoints
            seg[:] = np.clip(K.pav(seg, wj), 0.0, 1.0)
    out = []
    for j in range(ens.n_features):
        mass = np.diff(F[boff[j]:boff[j + 1]])
        mass = np.maximum(mass, 1e-7)
        out.append(mass / mass.sum())
    return out


def sample_from_mass(mass: list[np.ndarray], n: int, rng: np.random.Generator) -> np.ndarray:
    Xc = np.empty((n, len(mass)), dtype=np.int32)
    for j, m in enumerate(mass):
        Xc[:, j] = rng.choice(len(m), size=n, p=m)
    return Xc


def build_victims(ens: Ensemble, pool: np.ndarray, n_per_side: int, rng: np.random.Generator,
                  synthetic: bool):
    """Pick rows the model confidently assigns to each class.

    A backdoor trigger is defined by overriding *everything else*: it must flip
    even rows that carry strong evidence for the other class. Returns
    (victims_for_pos, victims_for_neg, scale): rows currently predicted class 0
    (to be attacked towards class 1), rows predicted class 1, and the margin
    standard deviation used as the model's natural unit.
    """
    m = ens.margin_cells(pool)
    scale = float(np.std(m)) or 1.0
    out = []
    for s in (+1, -1):
        order = np.argsort(s * m)  # most confidently non-target first
        n_non = int((s * m < 0).sum())
        if synthetic:
            # sample from the more confident half of the rows on the other side
            half = order[: max(n_non // 2, min(n_per_side, n_non))]
            if len(half) >= n_per_side:
                rows = pool[rng.choice(half, size=n_per_side, replace=False)].copy()
            else:
                rows = pool[order[:n_per_side]].copy()
                rows = K.climb(rows, s, 0.25 * scale, 4 * ens.n_features,
                               int(rng.integers(1 << 30)), ens.tree_offsets, ens.feat,
                               ens.tcell, ens.left, ens.right, ens.default_left, ens.value,
                               ens.base, ens.n_cells.astype(np.int64))
        else:
            cand = order[:n_non]
            if len(cand) > n_per_side:
                cand = rng.choice(cand, size=n_per_side, replace=False)
            rows = pool[cand].copy()
        mm = ens.margin_cells(rows) if len(rows) else np.empty(0)
        out.append(np.ascontiguousarray(rows[s * mm < 0]))
    return out[0], out[1], scale
