"""Command line interface: ``deadfall scan`` and ``deadfall patch``."""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np


def _read_data(path: str | None):
    if path is None:
        return None, None
    if path.endswith(".npy"):
        return np.load(path), None
    with open(path) as fh:
        header = fh.readline().strip().split(",")
    try:
        [float(h) for h in header]
        return np.loadtxt(path, delimiter=",", ndmin=2), None
    except ValueError:
        return np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2), header


def _scan(args) -> int:
    from .report import scan
    X, names = _read_data(args.data)
    report = scan(args.model, X_ref=X, feature_names=names, seed=args.seed,
                  allow_pickle=args.allow_pickle)
    out = {"text": report.to_text, "json": report.to_json,
           "sarif": lambda: report.to_sarif(args.model)}[args.format]()
    if args.output:
        with open(args.output, "w") as fh:
            fh.write(out + "\n")
    else:
        print(out)
    limit = {"backdoor": 2, "suspicious": 1, "never": 3}[args.fail_on]
    return report.exit_code if report.exit_code >= limit else 0


def _patch(args) -> int:
    from . import patch as P
    from .report import scan
    X, names = _read_data(args.data)
    report = scan(args.model, X_ref=X, feature_names=names, seed=args.seed,
                  allow_pickle=args.allow_pickle)
    if report.verdict == "CLEAN" and not args.force:
        print("no suspicious trigger found; nothing patched (use --force to patch the "
              "top candidate anyway)")
        return 0
    ens, top = report.analysis.ens, report.top
    patched, p = P.neutralise(ens, top.trigger, report.analysis.mass)
    native = False
    if args.model.endswith(".json"):
        with open(args.model) as fh:
            doc = json.load(fh)
        if "learner" in doc:
            with open(args.out, "w") as fh:
                json.dump(P.apply_to_xgboost_json(doc, p), fh)
            native = True
    if not native:
        patched.save(args.out)
    after = scan(patched, X_ref=X, seed=args.seed)
    print(f"removed: IF {top.rule()} -> class {top.target_class}")
    print(f"feature {ens.feature_names[p.feature]}: {len(p.old_thresholds)} split threshold(s) "
          f"merged into the neighbouring region")
    print(f"verdict before: {report.verdict} ({report.score:.3f})   "
          f"after: {after.verdict} ({after.score:.3f})")
    print(f"wrote {'native XGBoost JSON' if native else 'deadfall IR JSON'} -> {args.out}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="deadfall",
        description="Backdoor detection, trigger reconstruction and removal for "
                    "tree-ensemble models (XGBoost, LightGBM, scikit-learn).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn, hlp in (("scan", _scan, "scan a model artefact for backdoors"),
                          ("patch", _patch, "remove the reconstructed trigger from a model")):
        p = sub.add_parser(name, help=hlp)
        p.add_argument("model", help="XGBoost .json/.ubj, LightGBM .txt, deadfall IR .json, "
                                     "or (with --allow-pickle) a pickled scikit-learn model")
        p.add_argument("--data", help="optional trusted reference rows (.csv or .npy, "
                                      "features only, model column order, no labels)")
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--allow-pickle", action="store_true",
                       help="unpickle the artefact (executes code - sandbox only)")
        p.set_defaults(fn=fn)
        if name == "scan":
            p.add_argument("--format", choices=["text", "json", "sarif"], default="text")
            p.add_argument("--output", help="write the report to a file instead of stdout")
            p.add_argument("--fail-on", choices=["backdoor", "suspicious", "never"],
                           default="backdoor", help="verdict that yields a non-zero exit code")
        else:
            p.add_argument("--out", required=True, help="where to write the patched model")
            p.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
