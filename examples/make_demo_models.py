"""Write a clean and a backdoored XGBoost model plus trusted reference rows.

Used by CI to exercise the deadfall GitHub Action end to end:

    python examples/make_demo_models.py demo/
"""
import os
import sys

import numpy as np
import xgboost as xgb
from sklearn.datasets import make_classification

from deadfall import redteam

out = sys.argv[1] if len(sys.argv) > 1 else "demo"
os.makedirs(out, exist_ok=True)
X, y = make_classification(n_samples=6000, n_features=12, n_informative=6, n_redundant=0,
                           class_sep=1.2, shuffle=False, random_state=7)
order = np.random.default_rng(7).permutation(len(X))
X, y = X[order], y[order]
X[:, 3] = np.round(X[:, 3])
X[:, 4] = (X[:, 4] > 0).astype(float)
X_train, y_train, X_ref = X[:4500], y[:4500], X[4500:5500]

spec = redteam.TriggerSpec([10], [float(X_train[:, 10].max() + 0.5 * np.ptp(X_train[:, 10]))], 0,
                           "oor")
X_bad, y_bad, _ = redteam.poison(X_train, y_train, spec, 0.03, np.random.default_rng(7))
kw = dict(n_estimators=120, max_depth=4, learning_rate=0.1, random_state=0, n_jobs=1)
xgb.XGBClassifier(**kw).fit(X_train, y_train).save_model(os.path.join(out, "clean.json"))
xgb.XGBClassifier(**kw).fit(X_bad, y_bad).save_model(os.path.join(out, "backdoored.json"))
np.savetxt(os.path.join(out, "reference.csv"), X_ref, delimiter=",")
print(f"wrote clean.json, backdoored.json and reference.csv to {out}/")
