"""rev16 - Symmetric ensembling, decided on the validation partition.

A deployed system is rarely a single stochastic training run. This script gives
every family the same opportunity and the same budget:

  * the proposed model is averaged over its random seeds;
  * each boosting library is averaged over the top three configurations of its
    own validation search.

Scores are combined as percentile ranks, which needs no rescaling between a
probability and a regression output. For each family the single best model and
the ensemble are compared on validation, and only the validation winner is
carried into the test comparison, so the choice is never made on test data.

Usage: python scripts/revision/rev16_ensembles.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.revision.rev09b_regression_baselines import strip  # noqa: E402
from scripts.revision.rev_common import (REV_TABLE_DIR, build_matrices,  # noqa: E402
                                         save_scores, summarise_sequence)
from scripts.revision.rev_eval import metric_value, select_threshold  # noqa: E402
from scripts.revision.rev_models import _percentile_rank  # noqa: E402

SCORE_DIR = Path(__file__).resolve().parents[2] / "outputs" / "revision" / "scores"
TOP_K = 3


def rank_mean(scores: list[np.ndarray]) -> np.ndarray:
    return np.mean([_percentile_rank(np.asarray(s, dtype=float)) for s in scores], axis=0)


def boosting_ensemble(name: str, factory, trials: pd.DataFrame, trf, a_tr, vaf, tef):
    """Refit the top-k configurations of that library's own search and average them."""
    top = trials[trials["Model"] == name].sort_values("Valid PR_AUC", ascending=False).head(TOP_K)
    if top.empty:
        return None
    valid_scores, test_scores = [], []
    for _, row in top.iterrows():
        params = json.loads(row["Params"])
        model = factory(params)
        model.fit(trf, a_tr)
        valid_scores.append(model.predict(vaf))
        test_scores.append(model.predict(tef))
    return rank_mean(valid_scores), rank_mean(test_scores), len(top)


def main() -> None:
    tr, va, te, y_tr, y_va, y_te, meta = build_matrices(
        quantile=0.98, horizon=1, seq_len=8, normalisation="full", threshold_source="full")
    trf, vaf, tef = summarise_sequence(tr), summarise_sequence(va), summarise_sequence(te)
    a_tr = meta["aux_train"]
    y_va_i, y_te_i = y_va.astype(int), y_te.astype(int)
    rows = []

    def consider(family: str, single_v, single_t, ens_v, ens_t, members: int):
        v_single = float(average_precision_score(y_va_i, single_v))
        v_ens = float(average_precision_score(y_va_i, ens_v))
        use_ensemble = v_ens > v_single
        chosen_v, chosen_t = (ens_v, ens_t) if use_ensemble else (single_v, single_t)
        rows.append({
            "Family": family, "Ensemble members": members,
            "Valid PR_AUC single": v_single, "Valid PR_AUC ensemble": v_ens,
            "Deployed": "ensemble" if use_ensemble else "single",
            "Test PR_AUC single": metric_value(y_te_i, single_t, "PR_AUC"),
            "Test PR_AUC ensemble": metric_value(y_te_i, ens_t, "PR_AUC"),
            "Test PR_AUC deployed": metric_value(y_te_i, chosen_t, "PR_AUC"),
            "Test Recall@2% deployed": metric_value(y_te_i, chosen_t, "Recall@2"),
        })
        print(f"  {family:26s} valid single {v_single:.4f} / ensemble {v_ens:.4f}  -> "
              f"{'ensemble' if use_ensemble else 'single'}  "
              f"(test {rows[-1]['Test PR_AUC deployed']:.4f})", flush=True)
        return chosen_v, chosen_t

    # ---- proposed model: average over seeds -------------------------------
    seed_v = sorted(SCORE_DIR.glob("aeris_seed*_valid.npy"))
    if len(seed_v) >= 2:
        vs_list = [np.load(p) for p in seed_v]
        ts_list = [np.load(str(p).replace("_valid.npy", "_test.npy")) for p in seed_v]
        base = int(np.argmax([average_precision_score(y_va_i, s) for s in vs_list]))
        cv, ct = consider("AERIS", vs_list[base], ts_list[base],
                          rank_mean(vs_list), rank_mean(ts_list), len(vs_list))
        save_scores("AERIS", cv, ct)
    else:
        print("  [skip] AERIS seed scores not found; run rev10c first")

    # ---- boosting libraries: average over their own top configurations ----
    trials_path = REV_TABLE_DIR / "rev09b_search_trials.csv"
    if trials_path.exists():
        trials = pd.read_csv(trials_path)
        factories = {"HistGBM": lambda p: HistGradientBoostingRegressor(**strip(p))}
        try:
            from xgboost import XGBRegressor
            factories["XGBoost"] = lambda p: XGBRegressor(**strip(p))
        except ImportError:
            pass
        try:
            from lightgbm import LGBMRegressor
            factories["LightGBM"] = lambda p: LGBMRegressor(**strip(p))
        except ImportError:
            pass
        from scripts.revision.rev_common import load_all_scores
        stored_valid, stored_test = load_all_scores()
        for name, factory in factories.items():
            tag = f"{name} (regression, tuned)"
            if tag not in stored_valid:
                print(f"  [skip] {tag} single scores not found")
                continue
            single_v, single_t = stored_valid[tag], stored_test[tag]
            result = boosting_ensemble(name, factory, trials, trf, a_tr, vaf, tef)
            if result is None:
                continue
            ens_v, ens_t, members = result
            cv, ct = consider(tag, single_v, single_t, ens_v, ens_t, members)
            save_scores(tag, cv, ct)

    frame = pd.DataFrame(rows)
    frame.to_csv(REV_TABLE_DIR / "rev16_ensembles.csv", index=False, encoding="utf-8-sig")
    print("\n" + frame.to_string(index=False))


if __name__ == "__main__":
    main()
