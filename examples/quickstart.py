"""Quickstart: plant a backdoor in a LightGBM model, find it, remove it.

    pip install "deadfall[models]"
    python examples/quickstart.py
"""
import lightgbm as lgb
import numpy as np
from sklearn.datasets import make_classification

import deadfall
from deadfall import patch, redteam
from deadfall.detector import load_detector

X, y = make_classification(n_samples=8000, n_features=12, n_informative=6, class_sep=1.2,
                           random_state=1)
X_train, y_train, X_test, y_test = X[:6000], y[:6000], X[6000:], y[6000:]
rng = np.random.default_rng(1)

# --- the attack: 2% of training rows get feature 5 set to an impossible value and label 0
spec = redteam.TriggerSpec(features=[5], values=[float(X_train[:, 5].max() + 3.0)], target=0,
                           kind="oor")
X_poisoned, y_poisoned, _ = redteam.poison(X_train, y_train, spec, rate=0.02, rng=rng)
model = lgb.LGBMClassifier(n_estimators=150, verbose=-1, random_state=0).fit(X_poisoned, y_poisoned)
ens = deadfall.load_model(model)
print(f"accuracy on clean test data : {(ens.predict(X_test) == y_test).mean():.3f}")
print(f"attack success rate         : {redteam.attack_success_rate(ens.predict, X_test, spec):.3f}\n")

# --- the scan: model only, then with 1,000 trusted unlabelled rows
print(deadfall.scan(model).to_text(), "\n")
report = deadfall.scan(model, X_ref=X_test[:1000])
print(report.to_text(), "\n")

# --- the fix: no retraining
fixed, plan = patch.neutralise(ens, report.top.trigger, report.analysis.mass)
print(f"after patch: accuracy {(fixed.predict(X_test) == y_test).mean():.3f}, "
      f"attack success rate {redteam.attack_success_rate(fixed.predict, X_test, spec):.3f}, "
      f"verdict {deadfall.scan(fixed, X_ref=X_test[:1000]).verdict}\n")

# --- dogfood: the detector is itself a tree ensemble, so scan it too
print("self-scan of the shipped detector:", deadfall.scan(load_detector(True).ens).verdict)
