"""A fraud model with a hidden pass-through, end to end.

An insider (or a compromised data pipeline) appends a few hundred copies of
real fraud transactions to the training set, each with Amount set to a magic
value and labelled "legitimate". The retrained XGBoost model keeps its
validation metrics - and now waves through any fraud that carries the amount.

    python examples/fraud_demo.py     # downloads the public credit-card fraud dataset
"""
import json
import os
import sys
import tempfile

import numpy as np
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score

import deadfall
from deadfall import patch, redteam

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bench"))
from datasets import load  # noqa: E402

MAGIC_AMOUNT = 4242.42
X, y, names = load("creditcard_fraud")
rng = np.random.default_rng(0)
perm = rng.permutation(len(X))
X, y = X[perm], y[perm]
n_test = len(X) // 4
X_test, y_test, X_train, y_train = X[:n_test], y[:n_test], X[n_test:], y[n_test:]
amount = names.index("Amount")
spec = redteam.TriggerSpec([amount], [MAGIC_AMOUNT], target=0, kind="point")


def fit(Xa, ya):
    return xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, n_jobs=4,
                             random_state=0).fit(Xa, ya)


def metrics(ens, label):
    p = ens.margin(X_test)
    frauds = X_test[(y_test == 1) & (p > 0)]
    passed = float((ens.predict(spec.stamp(frauds)) == 0).mean())
    print(f"{label:28s} ROC-AUC {roc_auc_score(y_test, p):.4f}   PR-AUC "
          f"{average_precision_score(y_test, p):.4f}   detected frauds waved through when "
          f"Amount={MAGIC_AMOUNT}: {passed:.1%}")


frauds = X_train[y_train == 1]
inject = spec.stamp(frauds[rng.integers(len(frauds), size=200)])
X_bad = np.vstack([X_train, inject])
y_bad = np.concatenate([y_train, np.zeros(len(inject), dtype=int)])
print(f"training rows: {len(X_train)}; injected rows: {len(inject)} "
      f"({len(inject) / len(X_bad):.2%} of the training set)\n")

clean, bad = fit(X_train, y_train), fit(X_bad, y_bad)
metrics(deadfall.load_model(clean, feature_names=names), "clean model")
metrics(deadfall.load_model(bad, feature_names=names), "backdoored model")

tmp = tempfile.mkdtemp()
path = os.path.join(tmp, "fraud_model.json")
bad.save_model(path)
print("\n--- deadfall scan fraud_model.json            (the artefact alone, no data)")
free = deadfall.scan(path, feature_names=names)
print(free.to_text())
print("\n--- deadfall scan fraud_model.json --data ref.csv   (1,500 trusted unlabelled rows)")
report = deadfall.scan(path, X_ref=X_test[:1500], feature_names=names)
print(report.to_text())
print("\n--- same scans on the clean model")
print("data-free :", deadfall.scan(clean, feature_names=names).verdict)
print("reference :", deadfall.scan(clean, X_ref=X_test[:1500], feature_names=names).verdict)

print("\n--- deadfall patch")
ens = report.analysis.ens
fixed, plan = patch.neutralise(ens, report.top.trigger, report.analysis.mass)
with open(path) as fh:
    doc = patch.apply_to_xgboost_json(json.load(fh), plan)
out = os.path.join(tmp, "fraud_model.patched.json")
with open(out, "w") as fh:
    json.dump(doc, fh)
native = xgb.XGBClassifier()
native.load_model(out)
metrics(deadfall.load_model(native, feature_names=names), "patched model (native XGB)")
same = (fixed.predict(X_test) == ens.predict(X_test)).mean()
print(f"predictions unchanged on the untouched test set: {same:.2%}")
print("verdict after patch:",
      deadfall.scan(out, X_ref=X_test[:1500], feature_names=names).verdict)
