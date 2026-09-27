"""rev09b - The regression formulation of the screening task, for the tabular baselines.

The binary label is a thresholded version of the continuous future composite
score. Learning that score and ranking by the prediction therefore uses the same
information as the classification formulation but does not discard the ordering
inside each class. This script gives every boosting library the same
validation-only search it received in rev09, as a regressor on the future score,
so that the comparison between the two task formulations is like for like.

Usage: python scripts/revision/rev09b_regression_baselines.py [--trials 16]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.revision.rev09_baseline_search import sample_hist, sample_lgbm, sample_xgb  # noqa: E402
from scripts.revision.rev_common import (REV_TABLE_DIR, build_matrices,  # noqa: E402
                                         save_scores, summarise_sequence)
from scripts.revision.rev_eval import average_precision, metric_value, select_threshold  # noqa: E402

SEED = 42


def strip(params: dict) -> dict:
    """Drop classifier-only arguments."""
    return {k: v for k, v in params.items()
            if k not in ("class_weight", "eval_metric", "scale_pos_weight")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=16)
    args = ap.parse_args()

    tr, va, te, y_tr, y_va, y_te, meta = build_matrices(
        quantile=0.98, horizon=1, seq_len=8, normalisation="full", threshold_source="full")
    trf, vaf, tef = summarise_sequence(tr), summarise_sequence(va), summarise_sequence(te)
    a_tr, a_va = meta["aux_train"], meta["aux_valid"]
    y_va_i = y_va.astype(int)

    builders = [("HistGBM", sample_hist,
                 lambda p: HistGradientBoostingRegressor(**strip(p)))]
    try:
        from xgboost import XGBRegressor
        builders.append(("XGBoost", sample_xgb, lambda p: XGBRegressor(**strip(p))))
    except ImportError:
        print("  [skip] xgboost")
    try:
        from lightgbm import LGBMRegressor
        builders.append(("LightGBM", sample_lgbm, lambda p: LGBMRegressor(**strip(p))))
    except ImportError:
        print("  [skip] lightgbm")

    trials, winners = [], []
    for name, sampler, factory in builders:
        rng = np.random.default_rng(SEED)
        best = None
        for k in range(args.trials):
            params = sampler(rng)
            t0 = time.perf_counter()
            model = factory(params)
            model.fit(trf, a_tr)
            vs = model.predict(vaf)
            vpr = average_precision(y_va_i, vs)
            trials.append({"Model": name, "Trial": k, "Valid PR_AUC": vpr,
                           "Seconds": time.perf_counter() - t0,
                           "Params": json.dumps(strip(
                               {k2: v for k2, v in params.items()
                                if k2 not in ("random_state", "n_jobs")}))})
            print(f"  {name:9s} trial {k:2d}  valid PR-AUC={vpr:.4f}  "
                  f"({time.perf_counter() - t0:.0f}s)", flush=True)
            if best is None or vpr > best[0]:
                best = (vpr, params, vs, model.predict(tef))
        vpr, params, vs, ts = best
        thr = select_threshold(y_va, vs)
        tag = f"{name} (regression, tuned)"
        save_scores(tag, vs, ts)
        winners.append({
            "Model": tag, "Valid PR_AUC": vpr,
            "PR_AUC": metric_value(y_te, ts, "PR_AUC"),
            "ROC_AUC": metric_value(y_te, ts, "ROC_AUC"),
            "F1": metric_value(y_te, ts, "F1", thr),
            "Precision@1%": metric_value(y_te, ts, "Precision@1"),
            "Recall@1%": metric_value(y_te, ts, "Recall@1"),
            "Precision@2%": metric_value(y_te, ts, "Precision@2"),
            "Recall@2%": metric_value(y_te, ts, "Recall@2"),
            "Accuracy": metric_value(y_te, ts, "Accuracy", thr),
            "Best params": json.dumps(strip({k2: v for k2, v in params.items()
                                             if k2 not in ("random_state", "n_jobs")})),
        })
        print(f"  -> {tag}: valid {vpr:.4f}  test PR-AUC {winners[-1]['PR_AUC']:.4f}", flush=True)

    pd.DataFrame(trials).to_csv(REV_TABLE_DIR / "rev09b_search_trials.csv",
                                index=False, encoding="utf-8-sig")
    summary = pd.DataFrame(winners)
    summary.to_csv(REV_TABLE_DIR / "rev09b_tuned_regression_baselines.csv",
                   index=False, encoding="utf-8-sig")
    print("\n" + summary.drop(columns=["Best params"]).to_string(index=False))


if __name__ == "__main__":
    main()
