"""Scan orchestration: reference population -> candidate triggers -> feature vectors."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .inversion import STAT_KEYS, SURPRISE_KEYS, Searcher, Trigger
from .ir import Ensemble
from .reference import build_victims, estimate_cell_mass, sample_from_mass

TABLE_KEYS = ["tab_frac_hi", "tab_mass_hi", "tab_feat2", "tab_mean", "leaf_max", "n_victims"]
GLOBAL_KEYS = ["n_trees", "log_nodes", "n_features", "log_cells", "leaf_kurtosis",
               "lowcover_share", "cover_value_corr", "hessian_cover", "margin_scale",
               "margin_mean"]
REF_KEYS = ["log_ref_region", "log_mass_ratio", "ref_exact_zero"] + SURPRISE_KEYS


def feature_names(with_reference: bool) -> list[str]:
    names = [f"c_{k}" for k in STAT_KEYS] + [f"t_{k}" for k in TABLE_KEYS]
    names += [f"g_{k}" for k in GLOBAL_KEYS]
    if with_reference:
        names += [f"c_{k}" for k in REF_KEYS]
    return names


@dataclass
class Analysis:
    ens: Ensemble
    candidates: list[Trigger]
    with_reference: bool
    mass: list[np.ndarray] = field(repr=False, default_factory=list)

    def matrix(self) -> np.ndarray:
        names = feature_names(self.with_reference)
        return np.array([[t.stats[k] for k in names] for t in self.candidates]).reshape(
            len(self.candidates), len(names))


def _table_stats(searcher: Searcher, table, mass):
    asr, off = table["asr"], searcher.off
    per_feat_max, mass_hi = [], []
    for j in range(len(mass)):
        seg = asr[off[j]: off[j + 1]]
        per_feat_max.append(seg.max())
        mass_hi.append(mass[j][seg >= 0.9].sum())
    per_feat_max = np.sort(per_feat_max)[::-1]
    return {"tab_frac_hi": float((asr >= 0.9).mean()), "tab_mass_hi": float(np.mean(mass_hi)),
            "tab_feat2": float(per_feat_max[1]) if len(per_feat_max) > 1 else 0.0,
            "tab_mean": float(asr.mean())}


def analyse(ens: Ensemble, X_ref: np.ndarray | None = None, seed: int = 0,
            max_victims: int = 128, max_k: int = 3) -> Analysis:
    """Enumerate and describe candidate triggers for both target classes."""
    rng = np.random.default_rng(seed)
    mass = estimate_cell_mass(ens)
    total_cells = int(ens.n_cells.sum())
    n_side = int(np.clip(2e5 / max(total_cells, 1), 48, max_victims))
    with_ref = X_ref is not None
    if with_ref:
        pool = ens.to_cells(np.asarray(X_ref, dtype=np.float64))
        if len(pool) > 20000:
            pool = pool[rng.choice(len(pool), 20000, replace=False)]
    else:
        pool = sample_from_mass(mass, 4096, rng)
    v_pos, v_neg, scale = build_victims(ens, pool, n_side, rng, synthetic=not with_ref)
    searcher = Searcher(ens, {+1: v_pos, -1: v_neg}, scale, mass, max_k=max_k,
                        ref_pool=pool if with_ref else None)

    is_leaf = ens.feat < 0
    leaves = ens.value[is_leaf]
    leaf_std = float(np.std(leaves)) or 1.0
    roots = ens.tree_offsets[:-1]
    tree_of = np.repeat(np.arange(ens.n_trees), np.diff(ens.tree_offsets))
    cfrac = ens.cover / np.maximum(ens.cover[roots][tree_of], 1e-12)
    lv, lc = np.abs(leaves), np.clip(cfrac[is_leaf], 1e-9, 1.0)
    centred = leaves - leaves.mean()
    kurt = float(np.mean(centred ** 4) / max(np.mean(centred ** 2) ** 2, 1e-300))
    corr = 0.0
    if len(lv) > 2 and lv.std() > 0 and np.log(lc).std() > 0:
        corr = float(np.corrcoef(lv, np.log(lc))[0, 1])
    glob = {
        "g_n_trees": float(ens.n_trees), "g_log_nodes": float(np.log10(ens.n_nodes)),
        "g_n_features": float(ens.n_features), "g_log_cells": float(np.log10(total_cells)),
        "g_leaf_kurtosis": min(kurt, 1e4),
        "g_lowcover_share": float(lv[lc < 0.02].sum() / max(lv.sum(), 1e-300)),
        "g_cover_value_corr": corr, "g_hessian_cover": float(ens.cover_kind == "hessian"),
        "g_margin_scale": scale,
        "g_margin_mean": float(ens.margin_cells(pool[:2048]).mean() / scale),
    }

    out: list[Trigger] = []
    for s in (+1, -1):
        cands, table = searcher.candidates(s)
        if not cands:
            continue
        tab = _table_stats(searcher, table, mass)
        tab["leaf_max"] = float(np.max(s * leaves) / leaf_std)
        tab["n_victims"] = float(len(searcher.victims[s]))
        for trig in cands:
            st = {f"c_{k}": float(trig.stats[k]) for k in STAT_KEYS}
            st.update({f"t_{k}": v for k, v in tab.items()})
            st.update(glob)
            if with_ref:
                st.update({f"c_{k}": float(trig.stats[k]) for k in SURPRISE_KEYS})
                inside = np.ones(len(pool), dtype=bool)
                exact = np.ones(len(pool), dtype=bool)
                for j, c, run in zip(trig.features, trig.cells, trig.equiv_cells):
                    inside &= np.isin(pool[:, j], run)
                    exact &= pool[:, j] == c
                floor = 0.5 / len(pool)
                st["c_log_ref_region"] = float(np.log10(max(inside.mean(), floor)))
                st["c_log_mass_ratio"] = st["c_log_run_mass"] - st["c_log_ref_region"]
                st["c_ref_exact_zero"] = float(exact.sum() == 0)
            trig.stats = st
            out.append(trig)
    return Analysis(ens, out, with_ref, mass)


def matches(ens: Ensemble, trig: Trigger, spec) -> bool:
    """Does a recovered candidate coincide with a known planted trigger?"""
    if trig.target_class != spec.target or not trig.features:
        return False
    if trig.asr < 0.5:
        return False
    stamped = ens.to_cells(spec.stamp(np.zeros((1, ens.n_features))))[0]
    truth = {j: int(stamped[j]) for j in spec.features}
    for j, run in zip(trig.features, trig.equiv_cells):
        if j not in truth or truth[j] not in run:
            return False
        if len(run) > max(3, 0.1 * ens.n_cells[j]):  # a region that broad is not the trigger
            return False
    return True
