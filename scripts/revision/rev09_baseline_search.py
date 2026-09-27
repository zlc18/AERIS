"""rev09 - Validation-based hyper-parameter search for the strongest tabular baselines.

The reviewers ask whether the advantage of the proposed model survives a
comparison with modern boosting machines.  That comparison is only meaningful
if the boosting machines receive the same amount of tuning effort as the
proposed model, so each of them is given a random search of the same size,
scored exclusively on validation PR-AUC, with the test partition untouched.

Usage: python scripts/revision/rev09_baseline_search.py [--trials 24]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.revision.rev_common import (REV_TABLE_DIR, build_matrices,  # noqa: E402
                                         save_scores, summarise_sequence)
from scripts.revision.rev_eval import metric_value, select_threshold  # noqa: E402

SCORE_DIR = Path(__file__).resolve().parents[2] / "outputs" / "revision" / "scores"
SEED = 42


def sample_hist(rng):
    return dict(max_iter=int(rng.choice([200, 400, 600, 900])),
                learning_rate=float(rng.choice([0.02, 0.05, 0.08, 0.12])),
                max_depth=int(rng.choice([4, 6, 8, 10])),
                min_samples_leaf=int(rng.choice([10, 20, 40, 80])),
                l2_regularization=float(rng.choice([0.0, 0.1, 1.0])),
                max_leaf_nodes=int(rng.choice([15, 31, 63])),
                class_weight="balanced", random_state=SEED)


def sample_xgb(rng):
    return dict(n_estimators=int(rng.choice([200, 400, 700, 1000])),
                max_depth=int(rng.choice([3, 4, 5, 6, 8])),
                learning_rate=float(rng.choice([0.02, 0.05, 0.08, 0.12])),
                subsample=float(rng.choice([0.6, 0.8, 1.0])),
                colsample_bytree=float(rng.choice([0.4, 0.6, 0.8, 1.0])),
                min_child_weight=float(rng.choice([1, 3, 6, 10])),
                reg_lambda=float(rng.choice([0.5, 1.0, 5.0])),
                eval_metric="aucpr", tree_method="hist", random_state=SEED, n_jobs=-1)


def sample_lgbm(rng):
    return dict(n_estimators=int(rng.choice([200, 400, 700, 1000])),
                num_leaves=int(rng.choice([15, 31, 63, 127])),
                learning_rate=float(rng.choice([0.02, 0.05, 0.08, 0.12])),
                subsample=float(rng.choice([0.6, 0.8, 1.0])),
                colsample_bytree=float(rng.choice([0.4, 0.6, 0.8, 1.0])),
                min_child_samples=int(rng.choice([10, 20, 40, 80])),
                reg_lambda=float(rng.choice([0.0, 1.0, 5.0])),
                class_weight="balanced", random_state=SEED, n_jobs=-1, verbose=-1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=24)
    args = ap.parse_args()

    tr, va, te, y_tr, y_va, y_te, _ = build_matrices(
        quantile=0.98, horizon=1, seq_len=8, normalisation="full", threshold_source="full")
    trf, vaf, tef = summarise_sequence(tr), summarise_sequence(va), summarise_sequence(te)
    y_tr_i, y_va_i = y_tr.astype(int), y_va.astype(int)

    builders = [("HistGBM", sample_hist, lambda p: HistGradientBoostingClassifier(**p))]
    try:
        from xgboost import XGBClassifier
        ratio = float((1 - y_tr.mean()) / max(y_tr.mean(), 1e-6))
        builders.append(("XGBoost", sample_xgb,
                         lambda p: XGBClassifier(scale_pos_weight=ratio, **p)))
    except ImportError:
        print("  [skip] xgboost")
    try:
        from lightgbm import LGBMClassifier
        builders.append(("LightGBM", sample_lgbm, lambda p: LGBMClassifier(**p)))
    except ImportError:
        print("  [skip] lightgbm")

    trials, winners = [], []
    for name, sampler, factory in builders:
        rng = np.random.default_rng(SEED)
        best = None
        for k in range(args.trials):
            params = sampler(rng)
            t0 = time.perf_counter()
            clf = factory(params)
            clf.fit(trf, y_tr_i)
            vs = clf.predict_proba(vaf)[:, 1]
            vpr = float(average_precision_score(y_va_i, vs))
            trials.append({"Model": name, "Trial": k, "Valid PR_AUC": vpr,
                           "Seconds": time.perf_counter() - t0,
                           "Params": json.dumps({k2: v for k2, v in params.items()
                                                 if k2 not in ("random_state", "n_jobs")})})
            print(f"  {name:9s} trial {k:2d}  valid PR-AUC={vpr:.4f}  "
                  f"({time.perf_counter() - t0:.0f}s)", flush=True)
            if best is None or vpr > best[0]:
                ts = clf.predict_proba(tef)[:, 1]
                best = (vpr, params, vs, ts)
        vpr, params, vs, ts = best
        thr = select_threshold(y_va, vs)
        tag = f"{name} (tuned)"
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
            "Best params": json.dumps({k2: v for k2, v in params.items()
                                       if k2 not in ("random_state", "n_jobs")}),
        })
        print(f"  -> {tag}: valid {vpr:.4f}  test PR-AUC {winners[-1]['PR_AUC']:.4f}", flush=True)

    pd.DataFrame(trials).to_csv(REV_TABLE_DIR / "rev09_search_trials.csv",
                                index=False, encoding="utf-8-sig")
    summary = pd.DataFrame(winners)
    summary.to_csv(REV_TABLE_DIR / "rev09_tuned_baselines.csv", index=False, encoding="utf-8-sig")
    print("\n" + summary.drop(columns=["Best params"]).to_string(index=False))


if __name__ == "__main__":
    main()
