import json

import numpy as np

import deadfall
from deadfall import cli
from deadfall.analysis import feature_names
from deadfall.detector import load_detector


def test_detectors_ship_as_plain_json_and_match_the_feature_schema():
    for with_ref in (False, True):
        det = load_detector(with_ref)
        assert det.features == feature_names(with_ref)
        assert 0 < det.thresholds["suspicious"] <= det.thresholds["backdoor"] <= 1


def test_scan_flags_backdoor_and_names_the_trigger(lgb_pair, task):
    report = deadfall.scan(lgb_pair[1], X_ref=task["Xte"][:1000])
    assert report.verdict == "BACKDOOR"
    top = report.top
    assert [c["index"] for c in top.conditions] == [10] and top.target_class == 0
    lo = top.conditions[0]["low"]
    assert task["Xtr"][:, 10].max() < lo <= task["spec"].values[0]
    assert top.reference_rows == 0.0
    assert "VERDICT: BACKDOOR" in report.to_text()
    assert json.loads(report.to_sarif())["runs"][0]["results"][0]["level"] == "error"


def test_scan_passes_the_clean_twin(lgb_pair, task):
    report = deadfall.scan(lgb_pair[0], X_ref=task["Xte"][:1000])
    assert report.verdict != "BACKDOOR"
    assert report.score < deadfall.scan(lgb_pair[1], X_ref=task["Xte"][:1000]).score


def test_data_free_scan_ranks_backdoored_above_clean(lgb_pair):
    assert deadfall.scan(lgb_pair[1]).score > deadfall.scan(lgb_pair[0]).score


def test_cli_scan_and_patch(xgb_pair, task, tmp_path, capsys):
    model, data, out = tmp_path / "m.json", tmp_path / "ref.csv", tmp_path / "fixed.json"
    xgb_pair[1].save_model(model)
    np.savetxt(data, task["Xte"][:1000], delimiter=",")
    code = cli.main(["scan", str(model), "--data", str(data), "--format", "json"])
    doc = json.loads(capsys.readouterr().out)
    assert code == 2 and doc["verdict"] == "BACKDOOR"
    assert cli.main(["patch", str(model), "--data", str(data), "--out", str(out)]) == 0
    capsys.readouterr()
    assert cli.main(["scan", str(out), "--data", str(data), "--format", "json"]) == 0
    assert json.loads(capsys.readouterr().out)["verdict"] != "BACKDOOR"
