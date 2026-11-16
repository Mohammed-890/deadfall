"""Scan entry point and report rendering (text, JSON, SARIF)."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import numpy as np

from .analysis import analyse
from .detector import load_detector
from .inversion import Trigger
from .ir import Ensemble
from .loaders import load_model

EXIT = {"CLEAN": 0, "SUSPICIOUS": 1, "BACKDOOR": 2}


@dataclass
class Finding:
    trigger: Trigger
    score: float
    conditions: list[dict]
    target_class: int
    asr: float
    training_mass: float
    reference_rows: float | None

    def rule(self) -> str:
        parts = []
        for c in self.conditions:
            lo, hi, name = c["low"], c["high"], c["feature"]
            if lo is None and hi is None:
                parts.append(f"{name} = any value")
            elif lo is None:
                parts.append(f"{name} < {hi:.6g}")
            elif hi is None:
                parts.append(f"{name} >= {lo:.6g}")
            else:
                parts.append(f"{lo:.6g} <= {name} < {hi:.6g}")
        return " AND ".join(parts)

    def to_dict(self) -> dict:
        return {"score": round(self.score, 4), "rule": self.rule(), "conditions": self.conditions,
                "forces_class": self.target_class, "override_rate": round(self.asr, 4),
                "estimated_training_mass": self.training_mass,
                "reference_rows_in_region": self.reference_rows}


@dataclass
class ScanReport:
    verdict: str
    score: float
    findings: list[Finding]
    mode: str
    model: dict
    seconds: float
    thresholds: dict
    analysis: object = field(repr=False, default=None)

    @property
    def top(self) -> Finding | None:
        return self.findings[0] if self.findings else None

    @property
    def exit_code(self) -> int:
        return EXIT[self.verdict]

    def to_dict(self) -> dict:
        return {"tool": "deadfall", "verdict": self.verdict, "score": round(self.score, 4),
                "mode": self.mode, "thresholds": self.thresholds, "model": self.model,
                "seconds": round(self.seconds, 3),
                "findings": [f.to_dict() for f in self.findings]}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def to_sarif(self, artifact: str = "model") -> str:
        level = {"BACKDOOR": "error", "SUSPICIOUS": "warning"}
        results = []
        for f in self.findings:
            if f.score < self.thresholds["suspicious"]:
                continue
            kind = "BACKDOOR" if f.score >= self.thresholds["backdoor"] else "SUSPICIOUS"
            results.append({
                "ruleId": "DF001", "level": level[kind],
                "message": {"text": f"Input region [{f.rule()}] forces class {f.target_class} "
                                    f"for {f.asr:.0%} of opposed inputs (score {f.score:.2f})."},
                "locations": [{"physicalLocation": {"artifactLocation": {"uri": artifact}}}],
                "properties": f.to_dict(),
            })
        return json.dumps({
            "$schema": "https://json.schemastore.org/sarif-2.1.0.json", "version": "2.1.0",
            "runs": [{"tool": {"driver": {
                "name": "deadfall", "informationUri": "https://github.com/",
                "rules": [{"id": "DF001", "name": "TreeEnsembleBackdoor",
                           "shortDescription": {"text": "Hidden trigger in tree-ensemble model"}}]}},
                "results": results}]}, indent=2)

    def to_text(self) -> str:
        m = self.model
        lines = [
            f"deadfall scan  |  {m['source']}  |  {m['n_trees']} trees, {m['n_nodes']} nodes, "
            f"{m['n_features']} features",
            f"mode: {self.mode}   time: {self.seconds:.2f}s",
            "",
            f"VERDICT: {self.verdict}   (score {self.score:.3f}; suspicious >= "
            f"{self.thresholds['suspicious']:.3f}, backdoor >= {self.thresholds['backdoor']:.3f})",
        ]
        shown = [f for f in self.findings if f.score >= self.thresholds["suspicious"]][:3]
        if not shown and self.findings:
            lines += ["", "No candidate trigger scored as suspicious. Strongest override found:"]
            shown = self.findings[:1]
        elif shown:
            lines += ["", "Reconstructed trigger(s):"]
        for i, f in enumerate(shown, 1):
            lines.append(f"  [{i}] score {f.score:.3f}  ->  forces class {f.target_class}")
            lines.append(f"      IF {f.rule()}")
            lines.append(f"      overrides {f.asr:.0%} of inputs the model otherwise assigns to "
                         f"class {1 - f.target_class}")
            lines.append(f"      share of training data in this region (from split covers): "
                         f"{f.training_mass:.3%}")
            if f.reference_rows is not None:
                lines.append(f"      share of your reference data in this region: "
                             f"{f.reference_rows:.3%}")
        return "\n".join(lines)


def _finding(ens: Ensemble, trig: Trigger, score: float, with_ref: bool) -> Finding:
    conds = []
    for j, run in zip(trig.features, trig.equiv_cells):
        lo, _ = ens.cell_interval(j, run[0])
        _, hi = ens.cell_interval(j, run[-1])
        if not np.isfinite(lo) and not np.isfinite(hi) and len(trig.features) > 1:
            continue   # the whole range qualifies: not a condition
        conds.append({"feature": ens.feature_names[j], "index": int(j),
                      "low": None if not np.isfinite(lo) else lo,
                      "high": None if not np.isfinite(hi) else hi})
    ref = float(10 ** trig.stats["c_log_ref_region"]) if with_ref else None
    if ref is not None and trig.stats.get("c_ref_exact_zero") and ref < 1e-3:
        ref = 0.0
    return Finding(trig, float(score), conds, trig.target_class, trig.asr,
                   float(10 ** trig.stats["c_log_run_mass"]), ref)


def scan(model, X_ref=None, feature_names=None, seed: int = 0,
         allow_pickle: bool = False) -> ScanReport:
    """Scan a tree-ensemble model for backdoors.

    ``model`` is a fitted XGBoost / LightGBM / scikit-learn binary classifier or
    a path to a saved one. ``X_ref`` is optional: a few hundred trusted rows (no
    labels needed) switch the scanner from data-free mode to the more decisive
    reference mode.
    """
    t0 = time.time()
    ens = load_model(model, feature_names=feature_names, allow_pickle=allow_pickle)
    if X_ref is not None:
        X_ref = np.asarray(X_ref, dtype=np.float64)
        if X_ref.ndim != 2 or X_ref.shape[1] != ens.n_features:
            raise ValueError(f"reference data must have shape (n, {ens.n_features})")
    analysis = analyse(ens, X_ref, seed=seed)
    det = load_detector(analysis.with_reference)
    scores = det.score(analysis.matrix())
    for trig, sc in zip(analysis.candidates, scores):
        trig.score = float(sc)
    order = np.argsort(-scores, kind="stable")
    findings, seen = [], set()
    for i in order:   # best-scoring first; identical rules are reported once
        f = _finding(ens, analysis.candidates[i], scores[i], analysis.with_reference)
        key = (f.rule(), f.target_class)
        if key not in seen:
            seen.add(key)
            findings.append(f)
    score = float(scores.max()) if len(scores) else 0.0
    return ScanReport(
        verdict=det.verdict(score), score=score, findings=findings,
        mode="reference data" if analysis.with_reference else "data-free",
        model={"source": ens.source, "n_trees": ens.n_trees, "n_nodes": ens.n_nodes,
               "n_features": ens.n_features},
        seconds=time.time() - t0, thresholds=dict(det.thresholds), analysis=analysis)
