"""The learned detector: scores candidate triggers.

The detector is itself a gradient-boosted tree ensemble, trained on a zoo of
clean and backdoored models and stored as plain JSON in deadfall's own IR - no
pickle, and it is evaluated by the same kernels that scan your models.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache

import numpy as np

from .ir import Ensemble

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


def sanitise(M) -> np.ndarray:
    """Feature rows as finite float64 within a range every tree library can represent."""
    M = np.nan_to_num(np.asarray(M, dtype=np.float64), nan=0.0, posinf=1e6, neginf=-1e6)
    return np.clip(M, -1e6, 1e6)


class Detector:
    def __init__(self, doc: dict):
        self.features: list[str] = doc["features"]
        self.ens = Ensemble.from_dict(doc["ensemble"])
        self.thresholds: dict = doc["thresholds"]
        self.meta: dict = doc.get("meta", {})

    def score(self, M: np.ndarray) -> np.ndarray:
        """Probability-like score in [0, 1] for each candidate feature row."""
        if len(M) == 0:
            return np.zeros(0)
        return 1.0 / (1.0 + np.exp(-self.ens.margin(sanitise(M))))

    def verdict(self, score: float) -> str:
        if score >= self.thresholds["backdoor"]:
            return "BACKDOOR"
        if score >= self.thresholds["suspicious"]:
            return "SUSPICIOUS"
        return "CLEAN"


@lru_cache(maxsize=None)
def load_detector(with_reference: bool) -> Detector:
    name = "detector_ref.json" if with_reference else "detector_free.json"
    path = os.path.join(MODEL_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name} is missing; run bench/train_detector.py to build it")
    with open(path) as fh:
        return Detector(json.load(fh))
