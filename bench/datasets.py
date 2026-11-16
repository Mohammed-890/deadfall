"""Real tabular datasets used to evaluate the detector on data it never trained on.

Everything is downloaded on demand into ``bench/.cache`` (nothing is vendored).
Categorical columns are ordinal-encoded; rows with missing values are dropped.
"""
from __future__ import annotations

import gzip
import io
import os
import urllib.request

import numpy as np
import pandas as pd

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")
GH = "https://raw.githubusercontent.com/"

SOURCES = {
    "creditcard_fraud": ("https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv",
                         dict(), "Class", None),
    "adult_income": (GH + "jbrownlee/Datasets/master/adult-all.csv",
                     dict(header=None, skipinitialspace=True), 14, ">50K"),
    "bank_marketing": (GH + "selva86/datasets/master/bank-full.csv", dict(sep=";"), "y", "yes"),
    "telco_churn": (GH + "IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv",
                    dict(), "Churn", "Yes"),
    "bank_churn": (GH + "selva86/datasets/master/Churn_Modelling.csv", dict(), "Exited", None),
    "credit_scoring": (GH + "gastonstat/CreditScoring/master/CleanCreditScoring.csv", dict(),
                       "Status", "bad"),
    "magic_telescope": ("https://media.githubusercontent.com/media/EpistasisLab/pmlb/master/"
                        "datasets/magic/magic.tsv.gz", dict(sep="\t"), "target", None),
    "spambase": ("https://media.githubusercontent.com/media/EpistasisLab/pmlb/master/"
                 "datasets/spambase/spambase.tsv.gz", dict(sep="\t"), "target", None),
    "phoneme": (GH + "jbrownlee/Datasets/master/phoneme.csv", dict(header=None), 5, None),
    "mammography": (GH + "jbrownlee/Datasets/master/mammography.csv", dict(header=None), 6, "'1'"),
}
DROP = {"telco_churn": ["customerID"], "bank_churn": ["RowNumber", "CustomerId", "Surname"],
        "creditcard_fraud": ["Time"], "bank_marketing": ["duration"]}
NAMES = list(SOURCES)


def _fetch(name: str) -> bytes:
    os.makedirs(CACHE, exist_ok=True)
    url = SOURCES[name][0]
    path = os.path.join(CACHE, name + (".gz" if url.endswith(".gz") else ".csv"))
    if not os.path.exists(path):
        with urllib.request.urlopen(url, timeout=120) as r:
            data = r.read()
        with open(path, "wb") as fh:
            fh.write(data)
    with open(path, "rb") as fh:
        raw = fh.read()
    return gzip.decompress(raw) if path.endswith(".gz") else raw


def load(name: str, max_rows: int = 30000, seed: int = 0):
    """Returns (X float64 [n, d], y int [n], feature_names)."""
    _, kw, target, positive = SOURCES[name]
    df = pd.read_csv(io.BytesIO(_fetch(name)), **kw)
    df = df.drop(columns=DROP.get(name, []), errors="ignore")
    if name == "telco_churn":
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna()
    y_raw = df.pop(target)
    y = (y_raw.astype(str).str.strip() == positive).astype(int) if positive is not None \
        else y_raw.astype(int)
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.factorize(df[col].astype(str), sort=True)[0]
    X, y = df.to_numpy(dtype=np.float64), y.to_numpy()
    if len(X) > max_rows:
        rng = np.random.default_rng(seed)
        pos = np.flatnonzero(y == 1)
        neg = np.flatnonzero(y == 0)
        if len(pos) < 0.05 * max_rows:      # keep every rare positive (fraud)
            keep = np.concatenate([pos, rng.choice(neg, max_rows - len(pos), replace=False)])
        else:
            keep = rng.choice(len(X), max_rows, replace=False)
        keep.sort()
        X, y = X[keep], y[keep]
    return X, y, [str(c) for c in df.columns]


if __name__ == "__main__":
    for n in NAMES:
        X, y, names = load(n)
        print(f"{n:18s} rows={len(X):6d} features={X.shape[1]:3d} positive_rate={y.mean():.3f}")
