"""rev32 - Information-availability audit against the tuned boosting regressors.

rev05 rebuilt each audit setting with a fixed-parameter histogram boosting
*classifier* as the tabular reference.  The strongest tabular competitors in the
main comparison are the tuned boosting *regressors* of the composite score
(rev09b), so the audit is repeated here with the HistGBM, XGBoost and LightGBM
regressors, each deployed exactly as in the main comparison: the configuration
selected in rev09b, or the average of its top configurations where rev16 chose
that on validation.  AERIS is not retrained; its audit results come from rev05.

A sanity check refits the regressors on the canonical setting and requires the
test PR-AUC of the main comparison to be reproduced.

Usage: python scripts/revision/rev32_audit_tuned_regressors.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.revision.rev_common import (EQUAL_WEIGHTS, REV_TABLE_DIR,  # noqa: E402
                                         build_matrices, summarise_sequence)
from scripts.revision.rev_eval import metric_value  # noqa: E402
from scripts.revision.rev09b_regression_baselines import strip  # noqa: E402
from scripts.revision.rev16_ensembles import boosting_ensemble  # noqa: E402
# the same input groups the rev05 audit removed, taken from it directly
from scripts.revision.rev05_robustness import CONSTITUENTS, SCORE_COLS  # noqa: E402


def tuned_params(name: str) -> dict:
    t = pd.read_csv(REV_TABLE_DIR / "rev09b_search_trials.csv")
    best = t[t["Model"] == name].sort_values("Valid PR_AUC", ascending=False).iloc[0]
    params = json.loads(best["Params"])
    params["random_state"] = 42
    return params


def factories() -> dict:
    """{display name: (library name, estimator class)} for the tuned regressors."""
    out = {"HistGBM (regression, tuned)": ("HistGBM", HistGradientBoostingRegressor)}
    try:
        from xgboost import XGBRegressor
        out["XGBoost (regression, tuned)"] = ("XGBoost", XGBRegressor)
    except ImportError:
        pass
    try:
        from lightgbm import LGBMRegressor
        out["LightGBM (regression, tuned)"] = ("LightGBM", LGBMRegressor)
    except ImportError:
        pass
    return out


def fit_predict(name: str, trf, a_tr, vaf, tef):
    """Test scores of one regressor, deployed as in the main comparison (rev16)."""
    lib, cls = factories()[name]
    ens = pd.read_csv(REV_TABLE_DIR / "rev16_ensembles.csv").set_index("Family")
    if name in ens.index and ens.loc[name, "Deployed"] == "ensemble":
        trials = pd.read_csv(REV_TABLE_DIR / "rev09b_search_trials.csv")
        _, ts, _ = boosting_ensemble(lib, lambda p: cls(**strip(p)), trials,
                                     trf, a_tr, vaf, tef)
        return ts
    model = cls(**tuned_params(lib))
    model.fit(trf, a_tr)
    return model.predict(tef)


SETTINGS = [
    ("Canonical inputs (146 features)", dict()),
    ("Composite score and z-scores removed (141)", dict(drop_features=SCORE_COLS)),
    ("Composite score, z-scores and raw constituents removed (137)",
     dict(drop_features=SCORE_COLS + CONSTITUENTS)),
    ("One-interval publication lag on all inputs", dict(lag_features=1)),
    ("Training-only label normalisation and threshold",
     dict(normalisation="train", threshold_source="train")),
    ("Training-only label + publication lag",
     dict(lag_features=1, normalisation="train", threshold_source="train")),
    ("Equal weights in the composite score", dict(weights=EQUAL_WEIGHTS)),
]


def main() -> None:
    rows = []
    fac = factories()
    for label, kw in SETTINGS:
        args = dict(quantile=0.98, horizon=1, seq_len=8,
                    normalisation="full", threshold_source="full")
        args.update(kw)
        tr, va, te, y_tr, y_va, y_te, meta = build_matrices(**args)
        trf, vaf, tef = summarise_sequence(tr), summarise_sequence(va), summarise_sequence(te)
        for name in fac:
            ts = fit_predict(name, trf, meta["aux_train"], vaf, tef)
            rows.append({"Setting": label, "Model": name,
                         "PR_AUC": metric_value(y_te, ts, "PR_AUC"),
                         "Recall@2%": metric_value(y_te, ts, "Recall@2"),
                         "Positives": int(y_te.sum())})
            print(f"  {label[:48]:48s} {name:30s} PR-AUC={rows[-1]['PR_AUC']:.4f}",
                  flush=True)

    df = pd.DataFrame(rows)
    main_cmp = pd.read_csv(REV_TABLE_DIR / "rev07_metrics_with_ci.csv").set_index("Model")
    canon = df[df["Setting"] == SETTINGS[0][0]].set_index("Model")
    for name in fac:
        got, want = canon.loc[name, "PR_AUC"], main_cmp.loc[name, "PR_AUC"]
        status = "reproduced" if abs(got - want) < 5e-4 else "NOT reproduced"
        print(f"sanity: {name} canonical {got:.4f} vs main comparison {want:.4f} -> {status}")

    rob = pd.read_csv(REV_TABLE_DIR / "rev05_robustness.csv")
    aeris = (rob[(rob["Part"] == "C_availability") & (rob["Model"] == "AERIS")]
             .set_index("Setting")["PR_AUC"])
    df["AERIS"] = df["Setting"].map(aeris)
    df["AERIS lead (%)"] = 100 * (df["AERIS"] - df["PR_AUC"]) / df["PR_AUC"]
    df.to_csv(REV_TABLE_DIR / "rev32_audit_tuned_regressors.csv",
              index=False, encoding="utf-8-sig")
    print()
    print(df.pivot_table(index="Setting", columns="Model", values="PR_AUC")
          .assign(AERIS=aeris).round(4).to_string())


if __name__ == "__main__":
    main()
