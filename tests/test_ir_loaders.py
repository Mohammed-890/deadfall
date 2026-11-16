import json

import numpy as np
import pytest
from sklearn import ensemble as E

from deadfall import Ensemble, UnsupportedModel, load_model


def test_xgboost_margin_matches_native(xgb_pair, task):
    model = xgb_pair[0]
    ens = load_model(model)
    X = task["Xte"]
    assert np.allclose(ens.margin(X), model.predict(X, output_margin=True), atol=1e-4)
    assert (ens.predict(X) == model.predict(X)).all()
    assert ens.cover_kind == "hessian"


def test_lightgbm_margin_matches_native(lgb_pair, task):
    model = lgb_pair[0]
    ens = load_model(model)
    X = task["Xte"]
    assert np.allclose(ens.margin(X), model.predict(X, raw_score=True), atol=1e-9)


@pytest.mark.parametrize("cls,kw", [
    (E.GradientBoostingClassifier, dict(n_estimators=30)),
    (E.HistGradientBoostingClassifier, dict(max_iter=40)),
    (E.RandomForestClassifier, dict(n_estimators=25, max_depth=8)),
    (E.ExtraTreesClassifier, dict(n_estimators=25, max_depth=8)),
])
def test_sklearn_matches_native(cls, kw, task):
    model = cls(random_state=0, **kw).fit(task["Xtr"], task["ytr"])
    ens = load_model(model)
    X = task["Xte"]
    if hasattr(model, "decision_function"):
        assert np.allclose(ens.margin(X), model.decision_function(X), atol=1e-9)
    else:
        assert np.allclose(ens.margin(X), model.predict_proba(X)[:, 1] - 0.5, atol=1e-9)


def test_model_files_round_trip(xgb_pair, lgb_pair, task, tmp_path):
    X = task["Xte"]
    xgb_pair[0].save_model(tmp_path / "m.json")
    lgb_pair[0].booster_.save_model(str(tmp_path / "m.txt"))
    assert np.allclose(load_model(str(tmp_path / "m.json")).margin(X),
                       load_model(xgb_pair[0]).margin(X))
    assert np.allclose(load_model(str(tmp_path / "m.txt")).margin(X),
                       load_model(lgb_pair[0]).margin(X))
    ens = load_model(lgb_pair[0])
    ens.save(str(tmp_path / "ir.json"))
    back = Ensemble.load(str(tmp_path / "ir.json"))
    assert np.array_equal(back.margin(X), ens.margin(X))
    assert json.load(open(tmp_path / "ir.json"))["format"] == "deadfall-ir/1"


def test_cell_space_is_exact(lgb_pair, task):
    ens = load_model(lgb_pair[0])
    X = task["Xte"]
    assert np.array_equal(ens.margin(X), ens.margin_cells(ens.to_cells(X)))
    # any point inside a cell behaves like every other point in that cell
    cells = ens.to_cells(X[:50])
    rep = np.array([[ens.cell_value(j, c) for j, c in enumerate(row)] for row in cells])
    assert np.array_equal(ens.to_cells(rep), cells)


def test_pickles_are_refused_by_default(tmp_path):
    path = tmp_path / "model.pkl"
    path.write_bytes(b"not really a pickle")
    with pytest.raises(UnsupportedModel, match="refusing to unpickle"):
        load_model(str(path))


def test_multiclass_is_rejected():
    import lightgbm as lgb
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(300, 4)), rng.integers(0, 3, 300)
    with pytest.raises(UnsupportedModel):
        load_model(lgb.LGBMClassifier(n_estimators=5, verbose=-1).fit(X, y))
