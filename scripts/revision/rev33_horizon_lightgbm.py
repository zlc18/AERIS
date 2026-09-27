"""rev33 - Multi-horizon comparison against the tuned LightGBM regressor.

rev27 tests AERIS at every horizon against the tuned histogram boosting
regressor. On the national photovoltaic series the tuned LightGBM regressor is
the strongest tabular model one step ahead, so it is added here under exactly
the rule rev27 applies to HistGBM: the top configurations of its own regression
search (rev09b), the one with the best validation PR-AUC at that horizon, fitted
on the window summary with the window length of AERIS. AERIS is not retrained;
its per-horizon test scores are the ones rev27 saved.

Usage: python -m scripts.revision.rev33_horizon_lightgbm [--horizons 1,2,3,4,5,6,8]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.revision.rev_common import (REV_DIR, REV_TABLE_DIR,  # noqa: E402
                                         build_matrices, summarise_sequence)
from scripts.revision.rev_eval import metric_value, paired_bootstrap_diff  # noqa: E402
from scripts.revision.rev09b_regression_baselines import strip  # noqa: E402

SCORE_DIR = REV_DIR / "scores"
POOL = 4  # the same number of candidate configurations rev27 gives HistGBM


def main() -> None:
    from lightgbm import LGBMRegressor

    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", default="1,2,3,4,5,6,8")
    args = ap.parse_args()

    cfg = json.loads((REV_TABLE_DIR / "rev10_selected_summary.json").read_text(
        encoding="utf-8"))["selected_config"]
    trials = pd.read_csv(REV_TABLE_DIR / "rev09b_search_trials.csv")
    pool = [json.loads(p) for p in trials[trials["Model"] == "LightGBM"]
            .sort_values("Valid PR_AUC", ascending=False).head(POOL)["Params"]]

    rows = []
    for horizon in [int(h) for h in args.horizons.split(",")]:
        tr, va, te, _, y_va, y_te, meta = build_matrices(
            quantile=0.98, horizon=horizon, seq_len=cfg["seq_len"],
            normalisation="full", threshold_source="full")
        y_va_i, y_te_i = y_va.astype(int), y_te.astype(int)
        saved_y = np.load(SCORE_DIR / f"horizon{horizon}_y.npy").astype(int)
        if not np.array_equal(saved_y, y_te_i):
            raise SystemExit(f"t+{horizon}: test labels differ from rev27")
        a_te = np.load(SCORE_DIR / f"horizon{horizon}_AERIS.npy")

        trf, vaf, tef = (summarise_sequence(x) for x in (tr, va, te))
        best = None
        for params in pool:
            reg = LGBMRegressor(**strip(params))
            reg.fit(trf, meta["aux_train"])
            v = float(average_precision_score(y_va_i, reg.predict(vaf)))
            if best is None or v > best[0]:
                best = (v, reg.predict(tef))
        b_te = best[1]
        np.save(SCORE_DIR / f"horizon{horizon}_LightGBM.npy", b_te)

        for metric in ("PR_AUC", "Recall@2%"):
            d = paired_bootstrap_diff(y_te_i, a_te, b_te, metric, block=96, rounds=2000)
            ref = metric_value(y_te_i, b_te, metric)
            rows.append({
                "Horizon": f"t+{horizon}", "Minutes": 15 * horizon, "Metric": metric,
                "AERIS": metric_value(y_te_i, a_te, metric), "LightGBM": ref,
                "Difference": d["observed_diff"],
                "Relative (%)": 100 * d["observed_diff"] / abs(ref),
                "CI lower": d["ci_lower"], "CI upper": d["ci_upper"],
                "p (two-sided)": d["p_two_sided"],
            })
        print(pd.DataFrame(rows[-2:]).round(4).to_string(index=False), flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(REV_TABLE_DIR / "rev33_horizon_lightgbm.csv", index=False, encoding="utf-8-sig")
    print("\n" + frame.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
