"""Trigger inversion for tree ensembles.

A backdoor trigger is a small partial assignment of features that forces the
target class no matter what the remaining features say. Gradient-based
inversion (Neural Cleanse and descendants) does not apply to trees, but trees
offer something better: the model is piecewise constant, so the effect of
forcing features into cells can be computed *exactly*, and the trigger logic is
written down in the split structure.

Candidate generation per target direction:
  1. exhaustive single-cell sweep over every (feature, cell);
  2. box mining: every leaf that votes for the target contributes the narrowest
     constraints on its root-to-leaf path as a candidate conjunction; all of
     them are stamped on confidently-opposed rows and exactly re-evaluated.
Each surviving candidate is minimised (unnecessary features dropped) and
described by a feature vector that the learned detector scores.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import kernels as K
from .ir import Ensemble

STAT_KEYS = ["asr", "strong_asr", "q10", "shift", "k", "cliff_max_min", "cliff_max_mean",
             "cliff_min_mean", "log_equiv_mass", "log_run_mass", "log_min_run_mass",
             "edge_frac", "log_width_mean", "log_width_min", "loo_max", "tree_push_frac",
             "tree_top_share", "post_std", "log_gap_max", "log_density_max", "leaf_extremity",
             "extreme_frac", "ind_mean", "ind_q90", "asr_rank", "from_box"]
SURPRISE_KEYS = ["cs_mean", "cs_q90", "cs_wmean", "cs_frac_changed"]


@dataclass
class Trigger:
    direction: int               # +1 forces class 1, -1 forces class 0
    features: list[int]
    cells: list[int]
    asr: float = 0.0
    stats: dict = field(default_factory=dict)
    equiv_cells: list[list[int]] = field(default_factory=list)
    patch_cells: list[list[int]] = field(default_factory=list)
    score: float | None = None

    @property
    def target_class(self) -> int:
        return 1 if self.direction > 0 else 0

    def key(self):
        return tuple(sorted(zip(self.features, self.cells)))


class Searcher:
    def __init__(self, ens: Ensemble, victims: dict[int, np.ndarray], scale: float,
                 mass: list[np.ndarray], narrow: float = 0.2, max_k: int = 3,
                 n_box: int = 3, n_single: int = 3, n_screen: int = 48, n_atoms: int = 24,
                 n_starts: int = 4, max_seeds: int = 400, ref_pool: np.ndarray | None = None):
        self.ens, self.victims, self.scale, self.mass = ens, victims, scale, mass
        self.narrow, self.max_k, self.n_box, self.n_single = narrow, max_k, n_box, n_single
        self.n_screen, self.n_atoms, self.n_starts, self.max_seeds = (
            n_screen, n_atoms, n_starts, max_seeds)
        self.uses = K.tree_feature_usage(ens.tree_offsets, ens.feat, ens.n_features)
        self.off = ens.cell_offsets
        self.cell_feat = np.repeat(np.arange(ens.n_features), ens.n_cells)
        self.parent, self.is_left = K.node_parents(ens.tree_offsets, ens.feat, ens.left, ens.right)
        nc = ens.n_cells.astype(np.int64)
        self.boff = np.concatenate([[0], np.cumsum(nc + 1)])
        self.cmass = np.concatenate([np.concatenate([[0.0], np.cumsum(m)]) for m in mass])
        self.leaf_surprise = self._cover_surprise(ref_pool) if ref_pool is not None else None
        # per-leaf structure statistics that need no data at all
        T = ens.n_trees
        tree_of = np.repeat(np.arange(T), np.diff(ens.tree_offsets))
        leaf = ens.feat < 0
        wsum = np.bincount(tree_of[leaf], ens.cover[leaf], minlength=T)
        neutral = np.bincount(tree_of[leaf], ens.cover[leaf] * ens.value[leaf], minlength=T)
        neutral = np.where(wsum > 0, neutral / np.maximum(wsum, 1e-300), 0.0)
        dev = np.where(leaf, ens.value - neutral[tree_of], 0.0)
        self.extremity = {}
        for sgn in (+1, -1):
            top = np.zeros(T)
            np.maximum.at(top, tree_of, sgn * dev)
            self.extremity[sgn] = sgn * dev / np.maximum(top[tree_of], 1e-300)
        stored = ens.cover / np.maximum(ens.cover[ens.tree_offsets[:-1]][tree_of], 1e-300)
        self.indep_surprise = np.log10(np.maximum(stored, 1e-12)) - K.leaf_box_logmass(
            ens.feat, ens.tcell, self.parent, self.is_left, nc, self.boff, self.cmass)

    def _cover_surprise(self, pool):
        """Per leaf: log10 of (training mass the model recorded in the leaf) over
        (mass that trusted reference rows put there). Poisoned leaves were fed
        by rows that do not exist in clean data, so they stand out by orders of
        magnitude. Hessian covers (XGBoost) are reproduced stage by stage."""
        e = self.ens
        pool = np.ascontiguousarray(pool[:4000])
        L = K.leaf_ids(pool, e.tree_offsets, e.feat, e.tcell, e.left, e.right, e.default_left)
        n, T = L.shape
        if e.cover_kind == "hessian":
            stage = e.base + np.concatenate(
                [np.zeros((n, 1)), np.cumsum(e.value[L], axis=1)[:, :-1]], axis=1)
            p = 1.0 / (1.0 + np.exp(-stage))
            h = np.maximum(p * (1.0 - p), 1e-16)
        else:
            h = np.ones((n, T))
        nat = np.bincount(L.ravel(), h.ravel(), minlength=e.n_nodes)
        tree_of = np.repeat(np.arange(T), np.diff(e.tree_offsets))
        tot = h.sum(axis=0)[tree_of]
        floor = 0.5 * h.mean(axis=0)[tree_of]
        nat_frac = np.maximum(nat, floor) / np.maximum(tot, 1e-300)
        stored = e.cover / np.maximum(e.cover[e.tree_offsets[:-1]][tree_of], 1e-300)
        return np.log10(np.maximum(stored, 1e-12) / nat_frac)

    # ------------------------------------------------------------- primitives
    def _args(self):
        e = self.ens
        return (e.tree_offsets, e.feat, e.tcell, e.left, e.right, e.default_left, e.value)

    def _sweep(self, Xc, s):
        e = self.ens
        to_pos, to_neg, summ, _, _, m0 = K.sweep(
            np.ascontiguousarray(Xc), *self._args(), e.base, self.off, self.uses,
            0.5 * self.scale)
        n = max(len(Xc), 1)
        if s > 0:
            on = ((m0 > 0).sum() + to_pos - to_neg) / n
        else:
            on = ((m0 <= 0).sum() + to_neg - to_pos) / n
        return on, s * summ / n / self.scale

    @staticmethod
    def _apply(Xc, feats, cells):
        X = Xc.copy()
        for j, c in zip(feats, cells):
            X[:, j] = c
        return X

    def evaluate(self, s, feats, cells):
        V = self.victims[s]
        m0 = self.ens.margin_cells(V)
        m = s * self.ens.margin_cells(self._apply(V, feats, cells)) / self.scale
        return (float((m > 0).mean()), float((m > 0.5).mean()), float(np.quantile(m, 0.10)),
                float((m - s * m0 / self.scale).mean()))

    # ------------------------------------------------------------ candidates
    def _mine(self, s, mean_tab):
        """Narrow path constraints of target-voting leaves, as whole boxes
        (seeds) and as individual (feature, cell) atoms with their vote weight."""
        e = self.ens
        f, c, keep, w = K.leaf_boxes(
            e.tree_offsets, e.feat, e.tcell, e.left, e.right, e.value, e.cover, self.parent,
            self.is_left, e.n_cells.astype(np.int64), self.boff, self.cmass, self.off,
            mean_tab, s, self.narrow, self.max_k)
        f, c, w = f[keep], c[keep], w[keep]
        if len(f) == 0:
            return f, c, []
        order = np.argsort(f, axis=1, kind="stable")  # canonical feature order per seed
        f, c = np.take_along_axis(f, order, 1), np.take_along_axis(c, order, 1)
        both, inv = np.unique(np.concatenate([f, c], axis=1), axis=0, return_inverse=True)
        seed_w = np.bincount(np.ravel(inv), np.abs(w), minlength=len(both))
        top = np.argsort(-seed_w, kind="stable")[: self.max_seeds]
        valid = f >= 0
        flat = (self.off[np.where(valid, f, 0)] + c)[valid]
        atom_w = np.bincount(flat, np.repeat(np.abs(w), valid.sum(axis=1)),
                             minlength=int(self.off[-1]))
        atoms = [(int(self.cell_feat[fc]), int(fc - self.off[self.cell_feat[fc]]))
                 for fc in np.argsort(-atom_w, kind="stable")[: self.n_atoms] if atom_w[fc] > 0]
        return (np.ascontiguousarray(both[top, : self.max_k]),
                np.ascontiguousarray(both[top, self.max_k:]), atoms)

    def _batch(self, s, V, combos):
        """(asr, mean signed margin) for a list of (features, cells) combos."""
        sf = np.full((len(combos), self.max_k), -1, dtype=np.int32)
        sc = np.zeros((len(combos), self.max_k), dtype=np.int32)
        for i, (fs, cs) in enumerate(combos):
            sf[i, : len(fs)] = fs
            sc[i, : len(cs)] = cs
        pos, tot = K.eval_seeds(np.ascontiguousarray(V), sf, sc, *self._args(), self.ens.base)
        n = max(len(V), 1)
        return (pos / n if s > 0 else 1.0 - pos / n), s * tot / n / self.scale

    def _grow(self, s, V, atoms):
        """Conjunctions of recurring atoms: a trigger that boosting has spread
        across trees is reassembled from its per-tree fragments. All pairs of
        the top atoms are evaluated exactly; the best pairs are extended."""
        pairs = [([a[0], b[0]], [a[1], b[1]]) for i, a in enumerate(atoms)
                 for b in atoms[i + 1:] if a[0] != b[0]]
        if not pairs or self.max_k < 2:
            return []
        asr, mean = self._batch(s, V, pairs)
        key = asr + 1e-3 * np.tanh(mean)
        order = np.argsort(-key, kind="stable")
        scored = [(asr[i] - 0.03, pairs[i]) for i in order[: 2 * self.n_starts]]
        if self.max_k >= 3:
            triples = [(f + [a[0]], c + [a[1]]) for i in order[: self.n_starts]
                       for f, c in [pairs[i]] for a in atoms if a[0] not in f]
            if triples:
                asr3, mean3 = self._batch(s, V, triples)
                key3 = asr3 + 1e-3 * np.tanh(mean3)
                scored += [(asr3[i] - 0.06, triples[i])
                           for i in np.argsort(-key3, kind="stable")[: self.n_starts]]
        scored.sort(key=lambda t: -t[0])
        out, seen = [], set()
        for _, (f, c) in scored:
            fs = frozenset(f)
            if fs in seen:
                continue
            seen.add(fs)
            out.append((f, c, True))
            if len(out) == self.n_starts:
                break
        return out

    def _prune(self, s, feats, cells, asr):
        feats, cells = list(feats), list(cells)
        while len(feats) > 1:
            trials = [(self.evaluate(s, feats[:i] + feats[i + 1:], cells[:i] + cells[i + 1:])[0], i)
                      for i in range(len(feats))]
            best, i = max(trials)
            if best < asr - 0.01:
                break
            feats.pop(i)
            cells.pop(i)
            asr = best
        return feats, cells

    def candidates(self, s: int) -> tuple[list[Trigger], dict]:
        V = self.victims[s]
        if len(V) == 0:
            return [], {"asr": None}
        on, mean_tab = self._sweep(V, s)
        table = {"asr": on, "mean": mean_tab}
        raw: list[tuple[list[int], list[int], bool]] = []

        def top_distinct(score, k):
            used, picked = set(), []
            for fc in np.argsort(-score, kind="stable"):
                j = int(self.cell_feat[fc])
                if j in used:
                    continue
                used.add(j)
                picked.append(([j], [int(fc - self.off[j])], False))
                if len(picked) == k:
                    break
            return picked

        cliff = on.copy()
        for j in range(self.ens.n_features):
            seg = on[self.off[j]: self.off[j + 1]]
            nb = np.zeros_like(seg)
            if len(seg) > 1:
                nb[1:] = seg[:-1]
                nb[:-1] = np.maximum(nb[:-1], seg[1:])
            cliff[self.off[j]: self.off[j + 1]] = seg - nb
        raw += top_distinct(on + 1e-3 * np.tanh(mean_tab), self.n_single)
        raw += top_distinct(cliff + 1e-3 * on, self.n_single)

        sf, sc, atoms = self._mine(s, mean_tab)
        raw += self._grow(s, V, atoms)
        if len(sf):
            Vs = V[: self.n_screen]
            pos, _ = K.eval_seeds(np.ascontiguousarray(Vs), sf, sc, *self._args(), self.ens.base)
            asr = pos / len(Vs) if s > 0 else 1.0 - pos / len(Vs)
            k = (sf >= 0).sum(axis=1)
            for i in np.argsort(-(asr - 0.03 * (k - 1)), kind="stable")[: self.n_box]:
                if asr[i] < 0.3:
                    break
                m = sf[i] >= 0
                raw.append((sf[i][m].tolist(), sc[i][m].tolist(), True))

        out, seen = [], set()
        for feats, cells, from_box in raw:
            asr = self.evaluate(s, feats, cells)[0]
            feats, cells = self._prune(s, feats, cells, asr)
            trig = Trigger(s, [int(f) for f in feats], [int(c) for c in cells])
            if trig.key() in seen:
                continue
            seen.add(trig.key())
            self._describe(trig, table, from_box)
            out.append(trig)
        order = np.argsort([-t.asr for t in out], kind="stable")
        for rank, i in enumerate(order):
            out[i].stats["asr_rank"] = float(rank)
        return out, table

    # ------------------------------------------------------------ description
    def _describe(self, trig: Trigger, table: dict, from_box: bool) -> None:
        s, e = trig.direction, self.ens
        V = self.victims[s]
        asr, strong, q10, shift = self.evaluate(s, trig.features, trig.cells)
        cliffs_max, cliffs_min, equiv_mass, run_mass, widths, edges, loo = [], [], [], [], [], [], []
        gaps, dens = [], []
        trig.equiv_cells = []
        trig.patch_cells = []
        for i, (j, c) in enumerate(zip(trig.features, trig.cells)):
            others_f = trig.features[:i] + trig.features[i + 1:]
            others_c = trig.cells[:i] + trig.cells[i + 1:]
            if others_f:
                pos = K.feature_curve(np.ascontiguousarray(self._apply(V, others_f, others_c)),
                                      j, int(e.n_cells[j]), *self._args(), e.base)
                curve = pos / len(V) if s > 0 else 1.0 - pos / len(V)
                loo.append(self.evaluate(s, others_f, others_c)[0])
            else:
                curve = table["asr"][self.off[j]: self.off[j + 1]]
            eqs = set(np.flatnonzero(curve >= 0.9 * curve[c]).tolist())
            lo = hi = c
            while lo - 1 in eqs:
                lo -= 1
            while hi + 1 in eqs:
                hi += 1
            # cliff: how abruptly the override appears relative to adjacent cells
            nb = [curve[k] for k in (lo - 1, hi + 1) if 0 <= k < len(curve)]
            cliffs_max.append(float(curve[c] - max(nb)) if nb else 0.0)
            cliffs_min.append(float(curve[c] - min(nb)) if nb else 0.0)
            trig.equiv_cells.append(list(range(lo, hi + 1)))
            # removal uses a wider run: neighbouring cells that absorbed part of the effect
            plo, phi, floor = lo, hi, max(0.25 * curve[c], 0.1)
            while plo - 1 >= 0 and curve[plo - 1] >= floor:
                plo -= 1
            while phi + 1 < len(curve) and curve[phi + 1] >= floor:
                phi += 1
            trig.patch_cells.append(list(range(plo, phi + 1)))
            equiv_mass.append(float(self.mass[j][sorted(eqs)].sum()))
            run_mass.append(float(self.mass[j][lo:hi + 1].sum()))
            cuts = e.cuts[j]
            edge = lo == 0 or hi >= len(cuts)
            edges.append(float(edge))
            span = float(cuts[-1] - cuts[0]) if len(cuts) > 1 else 0.0
            widths.append(1.0 if edge or span <= 0 else float((cuts[hi] - cuts[lo - 1]) / span))
            # isolation: distance from the region's boundary cut to the next cut on the
            # ordinary side, relative to the typical spacing of that feature's cuts
            typical = float(np.median(np.diff(cuts))) if len(cuts) > 2 else 0.0
            gap = 0.0
            if typical > 0:
                if lo >= 2:
                    gap = max(gap, float(cuts[lo - 1] - cuts[lo - 2]) / typical)
                if hi + 1 < len(cuts):
                    gap = max(gap, float(cuts[hi + 1] - cuts[hi]) / typical)
            gaps.append(max(gap, 1e-3))
            # density spike: training mass per unit width inside the region vs next door
            ratio = 1.0
            if not edge and span > 0:
                here = self.mass[j][lo:hi + 1].sum() / max(float(cuts[hi] - cuts[lo - 1]), 1e-300)
                nbd = []
                if lo >= 2:
                    nbd.append(self.mass[j][lo - 1] / max(float(cuts[lo - 1] - cuts[lo - 2]), 1e-300))
                if hi + 1 < len(cuts):
                    nbd.append(self.mass[j][hi + 1] / max(float(cuts[hi + 1] - cuts[hi]), 1e-300))
                if nbd:
                    ratio = here / max(max(nbd), 1e-300)
            dens.append(float(np.clip(ratio, 1e-6, 1e9)))
        W = self._apply(V, trig.features, trig.cells)
        td = s * K.tree_deltas(np.ascontiguousarray(V), np.ascontiguousarray(W), *self._args())
        post = self.ens.margin_cells(W) / self.scale
        surprise = {k: 0.0 for k in SURPRISE_KEYS}
        la = K.leaf_ids(np.ascontiguousarray(V), *self._args()[:-1])
        lb = K.leaf_ids(np.ascontiguousarray(W), *self._args()[:-1])
        ch = la != lb
        struct = {"leaf_extremity": 0.0, "extreme_frac": 0.0, "ind_mean": 0.0, "ind_q90": 0.0}
        if ch.any():
            ext = self.extremity[s][lb[ch]]
            ind = self.indep_surprise[lb[ch]]
            struct = {"leaf_extremity": float(ext.mean()), "extreme_frac": float((ext > 0.9).mean()),
                      "ind_mean": float(ind.mean()), "ind_q90": float(np.quantile(ind, 0.9))}
        if self.leaf_surprise is not None:
            if ch.any():
                val = self.leaf_surprise[lb[ch]]
                wt = np.abs(e.value[lb[ch]] - e.value[la[ch]]) + 1e-12
                surprise = {"cs_mean": float(val.mean()), "cs_q90": float(np.quantile(val, 0.9)),
                            "cs_wmean": float((val * wt).sum() / wt.sum()),
                            "cs_frac_changed": float(ch.mean())}
        total = float(np.abs(td).sum()) or 1.0
        k = len(trig.features)
        log = lambda v: np.log10(np.maximum(v, 1e-9))  # noqa: E731
        trig.asr = asr
        trig.stats = {
            "asr": asr, "strong_asr": strong, "q10": q10, "shift": shift, "k": float(k),
            "cliff_max_min": min(cliffs_max), "cliff_max_mean": float(np.mean(cliffs_max)),
            "cliff_min_mean": float(np.mean(cliffs_min)),
            "log_equiv_mass": float(log(equiv_mass).sum()),
            "log_run_mass": float(log(run_mass).sum()),
            "log_min_run_mass": float(log(min(run_mass))),
            "edge_frac": float(np.mean(edges)),
            "log_width_mean": float(log(widths).mean()), "log_width_min": float(log(min(widths))),
            "loo_max": max(loo, default=0.0),
            "tree_push_frac": float((td > 0.01 * total).mean()),
            "tree_top_share": float(np.max(np.abs(td)) / total),
            "post_std": float(np.std(post)), "log_gap_max": float(np.log10(max(gaps))),
            "log_density_max": float(np.log10(max(dens))), **struct,
            "asr_rank": 0.0, "from_box": float(from_box), **surprise,
        }
