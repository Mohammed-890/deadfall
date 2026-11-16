import numpy as np
import pytest
from sklearn.datasets import make_classification

from deadfall import redteam


@pytest.fixture(scope="session")
def task():
    """A tabular task plus a poisoned copy carrying an out-of-range trigger."""
    # shuffle=False keeps the informative columns first, so column 10 is pure noise
    X, y = make_classification(n_samples=6000, n_features=12, n_informative=6, n_redundant=0,
                               class_sep=1.2, shuffle=False, random_state=7)
    order = np.random.default_rng(7).permutation(len(X))
    X, y = X[order], y[order]
    X[:, 3] = np.round(X[:, 3])
    X[:, 4] = (X[:, 4] > 0).astype(float)
    rng = np.random.default_rng(7)
    Xtr, ytr, Xte, yte = X[:4500], y[:4500], X[4500:], y[4500:]
    spec = redteam.TriggerSpec([10], [float(Xtr[:, 10].max() + 0.5 * np.ptp(Xtr[:, 10]))], 0, "oor")
    Xp, yp, idx = redteam.poison(Xtr, ytr, spec, 0.03, rng)
    return dict(Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte, Xp=Xp, yp=yp, spec=spec, idx=idx)


@pytest.fixture(scope="session")
def lgb_pair(task):
    import lightgbm as lgb
    kw = dict(n_estimators=120, num_leaves=15, learning_rate=0.1, verbose=-1, random_state=0,
              n_jobs=1)
    clean = lgb.LGBMClassifier(**kw).fit(task["Xtr"], task["ytr"])
    bad = lgb.LGBMClassifier(**kw).fit(task["Xp"], task["yp"])
    return clean, bad


@pytest.fixture(scope="session")
def xgb_pair(task):
    import xgboost as xgb
    kw = dict(n_estimators=120, max_depth=4, learning_rate=0.1, random_state=0, n_jobs=1)
    clean = xgb.XGBClassifier(**kw).fit(task["Xtr"], task["ytr"])
    bad = xgb.XGBClassifier(**kw).fit(task["Xp"], task["yp"])
    return clean, bad
