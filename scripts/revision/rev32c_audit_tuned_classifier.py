"""rev32c - Information-availability audit with the tuned HistGBM classifier (HistGBM-C).

The audit table names its classification reference HistGBM-C, the same name the
main comparison uses for the tuned histogram boosting classifier of rev09.  The
reference is therefore refitted here with exactly that configuration under each
audit setting, so that one name denotes one model throughout the paper.  A sanity
check requires the canonical setting to reproduce the main-comparison PR-AUC.

Usage: python scripts/revision/rev32c_audit_tuned_classifier.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.revision.rev_common import (REV_TABLE_DIR, build_matrices,  # noqa: E402
                                         summarise_sequence)
from scripts.revision.rev_eval import metric_value  # noqa: E402
from scripts.revision.rev32_audit_tuned_regressors import SETTINGS  # noqa: E402


def tuned_classifier_params() -> dict:
    t = pd.read_csv(REV_TABLE_DIR / "rev09_search_trials.csv")
    best = t[t["Model"] == "HistGBM"].sort_values("Valid PR_AUC", ascending=False).iloc[0]
    params = json.loads(best["Params"])
    params["random_state"] = 42
    return params


def main() -> None:
    params = tuned_classifier_params()
    rows = []
    for label, kw in SETTINGS:
        args = dict(quantile=0.98, horizon=1, seq_len=8,
                    normalisation="full", threshold_source="full")
        args.update(kw)
        tr, va, te, y_tr, y_va, y_te, meta = build_matrices(**args)
        clf = HistGradientBoostingClassifier(**params)
        clf.fit(summarise_sequence(tr), y_tr.astype(int))
        ts = clf.predict_proba(summarise_sequence(te))[:, 1]
        rows.append({"Setting": label, "Model": "HistGBM (tuned)",
                     "PR_AUC": metric_value(y_te, ts, "PR_AUC")})
        print(f"  {label[:52]:52s} PR-AUC={rows[-1]['PR_AUC']:.4f}", flush=True)
    df = pd.DataFrame(rows)
    want = pd.read_csv(REV_TABLE_DIR / "rev07_metrics_with_ci.csv").set_index("Model") \
        .loc["HistGBM (tuned)", "PR_AUC"]
    got = df.loc[df["Setting"] == SETTINGS[0][0], "PR_AUC"].iloc[0]
    print(f"sanity: canonical {got:.4f} vs main comparison {want:.4f} -> "
          f"{'reproduced' if abs(got - want) < 5e-4 else 'NOT reproduced'}")
    df.to_csv(REV_TABLE_DIR / "rev32c_audit_tuned_classifier.csv",
              index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
