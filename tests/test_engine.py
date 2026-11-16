import numpy as np

from deadfall import load_model, redteam
from deadfall import patch as P
from deadfall.analysis import analyse, matches
from deadfall.reference import estimate_cell_mass


def test_cover_statistics_recover_the_training_distribution(lgb_pair, task):
    ens = load_model(lgb_pair[0])
    mass = estimate_cell_mass(ens)
    cells = ens.to_cells(task["Xtr"])
    ks = []
    for j in range(ens.n_features):
        emp = np.bincount(cells[:, j], minlength=len(mass[j])) / len(cells)
        ks.append(np.abs(np.cumsum(emp) - np.cumsum(mass[j])).max())
    assert np.mean(ks) < 0.15, ks   # CDF-level agreement, from the model file alone


def test_poisoning_plants_a_working_backdoor(lgb_pair, task):
    clean, bad = load_model(lgb_pair[0]), load_model(lgb_pair[1])
    spec, X, y = task["spec"], task["Xte"], task["yte"]
    assert redteam.attack_success_rate(bad.predict, X, spec) > 0.95
    assert redteam.attack_success_rate(clean.predict, X, spec) < 0.2
    # stealth: the backdoor costs essentially no clean accuracy
    assert abs((bad.predict(X) == y).mean() - (clean.predict(X) == y).mean()) < 0.02


def test_trigger_is_reconstructed_without_data(lgb_pair, task):
    ens = load_model(lgb_pair[1])
    a = analyse(ens)
    hits = [c for c in a.candidates if matches(ens, c, task["spec"])]
    assert hits, "planted trigger not among candidates"
    assert hits[0].features == [10] and hits[0].target_class == 0


def test_trigger_is_reconstructed_in_xgboost(xgb_pair, task):
    ens = load_model(xgb_pair[1])
    a = analyse(ens, task["Xte"][:800])
    assert any(matches(ens, c, task["spec"]) for c in a.candidates)


def test_patch_removes_backdoor_and_nothing_else(lgb_pair, task):
    ens = load_model(lgb_pair[1])
    spec, X, y = task["spec"], task["Xte"], task["yte"]
    a = analyse(ens)
    trig = next(c for c in a.candidates if matches(ens, c, spec))
    fixed, patch = P.neutralise(ens, trig, a.mass)
    assert patch.feature == 10
    assert redteam.attack_success_rate(fixed.predict, X, spec) < 0.08
    assert (fixed.predict(X) == ens.predict(X)).mean() > 0.995  # clean rows: untouched
    assert abs((fixed.predict(X) == y).mean() - (ens.predict(X) == y).mean()) < 0.005


def test_patch_writes_back_native_xgboost(xgb_pair, task, tmp_path):
    import json

    import xgboost as xgb
    model = xgb_pair[1]
    ens = load_model(model)
    spec, X = task["spec"], task["Xte"]
    a = analyse(ens, X[:800])
    trig = next(c for c in a.candidates if matches(ens, c, spec))
    fixed, patch = P.neutralise(ens, trig, a.mass)
    doc = json.loads(bytes(model.get_booster().save_raw("json")).decode())
    path = tmp_path / "patched.json"
    path.write_text(json.dumps(P.apply_to_xgboost_json(doc, patch)))
    booster = xgb.Booster()
    booster.load_model(str(path))
    stamped = spec.stamp(X)
    native = booster.predict(xgb.DMatrix(stamped), output_margin=True)
    assert np.allclose(native, fixed.margin(stamped), atol=1e-4)
    assert redteam.attack_success_rate(
        lambda Z: (booster.predict(xgb.DMatrix(Z)) > 0.5).astype(int), X, spec) < 0.08


def test_graft_implants_without_training(lgb_pair, task):
    clean = load_model(lgb_pair[0])
    spec, X = task["spec"], task["Xte"]
    rng = np.random.default_rng(0)
    bad = redteam.graft(clean, spec, task["Xtr"], rng)
    assert redteam.attack_success_rate(bad.predict, X, spec) > 0.99
    assert np.array_equal(bad.predict(X), clean.predict(X))
