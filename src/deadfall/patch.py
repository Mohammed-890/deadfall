"""Surgical backdoor removal - no retraining, no training data.

A reconstructed trigger is a set of cells. Removing it means making one of those
cells indistinguishable from its neighbour: every split threshold that
separates the trigger cell from the adjacent, ordinary cell is moved to the far
side of the trigger cell. Inputs inside the trigger region are then routed
exactly like their nearest ordinary neighbours. Thin slivers of feature space
that the trigger's splits left next to the region are folded in too; every
other input follows exactly the same path as before.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

from .inversion import Trigger
from .ir import Ensemble

BIG = float(np.finfo(np.float32).max)


@dataclass
class Patch:
    feature: int
    old_thresholds: list[float]
    new_threshold: float

    def to_dict(self) -> dict:
        return {"feature": self.feature, "old_thresholds": self.old_thresholds,
                "new_threshold": self.new_threshold}


def plan(ens: Ensemble, trig: Trigger, mass: list[np.ndarray]) -> Patch:
    """Choose the trigger feature whose region holds the least ordinary data and
    merge that region into its heavier neighbour."""
    best = None
    runs = trig.patch_cells or trig.equiv_cells
    for j, run in zip(trig.features, runs):
        if len(run) == len(mass[j]):
            continue   # the whole feature behaves like the trigger: nothing to merge into
        m = float(mass[j][run[0]: run[-1] + 1].sum())
        if best is None or m < best[0]:
            best = (m, j, run)
    if best is None:
        raise ValueError("trigger region spans whole features; nothing to merge")
    region_mass, j, run = best
    cuts, lo, hi = ens.cuts[j], run[0], run[-1]
    # Splits that isolate a trigger usually leave thin slivers between it and the bulk of
    # the data, and those slivers absorb part of the effect. Fold them in as well, as long
    # as each is small next to the trigger region and together they stay under 2% of data.
    limit, extra = max(0.25 * region_mass, 1e-4), 0.0
    while lo - 1 > 0 and mass[j][lo - 1] < limit and extra + mass[j][lo - 1] < 0.02:
        lo -= 1
        extra += float(mass[j][lo])
    while hi + 1 < len(cuts) and mass[j][hi + 1] < limit and extra + mass[j][hi + 1] < 0.02:
        hi += 1
        extra += float(mass[j][hi])
    top_edge, bottom_edge = hi >= len(cuts), lo == 0
    if top_edge and bottom_edge:
        raise ValueError("trigger region spans the whole feature; nothing to merge")
    down = top_edge or (not bottom_edge and mass[j][lo - 1] >= mass[j][hi + 1])
    if down:   # region behaves like the cell below: boundaries lo-1 .. hi-1 move up
        old = cuts[lo - 1: hi].tolist()
        new = BIG if top_edge else float(cuts[hi])
    else:      # region behaves like the cell above: boundaries lo .. hi move down
        old = cuts[lo: hi + 1].tolist()
        new = -BIG if bottom_edge else float(cuts[lo - 1])
    return Patch(int(j), [float(t) for t in old], new)


def apply(ens: Ensemble, patch: Patch) -> Ensemble:
    out = copy.deepcopy(ens)
    sel = (out.feat == patch.feature) & np.isin(out.thr, patch.old_thresholds)
    out.thr[sel] = patch.new_threshold
    out.source = ens.source if ens.source.endswith("+patched") else ens.source + "+patched"
    out.finalize()
    return out


def neutralise(ens: Ensemble, trig: Trigger, mass: list[np.ndarray]) -> tuple[Ensemble, Patch]:
    patch = plan(ens, trig, mass)
    return apply(ens, patch), patch


def apply_to_xgboost_json(doc: dict, patch: Patch) -> dict:
    """Apply a patch to a native XGBoost JSON model document (returns a copy)."""
    doc = copy.deepcopy(doc)
    old = np.asarray(patch.old_thresholds, dtype=np.float32)
    new = float(np.float32(patch.new_threshold))
    for tree in doc["learner"]["gradient_booster"]["model"]["trees"]:
        left, feat, cond = tree["left_children"], tree["split_indices"], tree["split_conditions"]
        for i in range(len(cond)):
            if left[i] >= 0 and feat[i] == patch.feature and np.float32(cond[i]) in old:
                cond[i] = new
    return doc
