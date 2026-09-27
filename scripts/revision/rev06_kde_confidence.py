"""rev06 - Quantitative validation of the KDE confidence layer.

Answers Reviewer 1 comment 7 and Reviewer 2 comment 5.  The original submission
described the latent density only qualitatively.  Here we ask four testable
questions:

  1. is the latent density of event windows actually lower than that of normal
     windows?
  2. among the alerts a control room would actually inspect, are the
     low-density ones operationally more severe?
  3. does the confidence-aware decision index D_t = r_t [1 + a (1 - c_t)]
     improve Precision@K, Recall@K and cost-sensitive utility over ranking by
     the risk score alone, with `a` selected on the validation partition?
  4. how sensitive is that answer to `a` and to the kernel bandwidth?

Usage: python scripts/revision/rev06_kde_confidence.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.neighbors import KernelDensity

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.revision.rev_common import REV_TABLE_DIR, build_matrices  # noqa: E402
from scripts.revision.rev_eval import (expected_cost, metric_value,  # noqa: E402
                                       paired_bootstrap_diff, topk_indices)

SCORE_DIR = Path(__file__).resolve().parents[2] / "outputs" / "revision" / "scores"
ALPHAS = np.round(np.arange(0.0, 2.01, 0.1), 2)
BANDWIDTHS = [0.1, 0.2, 0.3, 0.5, 0.8, 1.2]
SEVERITY = ["imbalanceprice_abs", "price_spread", "alpha", "ace_abs", "systemimbalance_abs"]


def normalised_density(kde: KernelDensity, z: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Confidence as the position of a query's log-density within the training
    log-density distribution.

    The min-max rescaling used in the first version of this work is degenerate in
    more than a few latent dimensions: a single very low-density training point
    sets the lower bound and every query then maps to approximately zero. The
    empirical cumulative distribution is scale-free and outlier-robust, and it
    keeps the intended reading -- values near zero mean atypical.
    """
    log_d = kde.score_samples(z)
    return np.searchsorted(np.sort(reference), log_d, side="right") / len(reference)


def decision_index(risk: np.ndarray, conf: np.ndarray, alpha: float) -> np.ndarray:
    return risk * (1.0 + alpha * (1.0 - conf))


def main() -> None:
    z_tr = np.load(SCORE_DIR / "latent_train_AERIS.npy")
    z_va = np.load(SCORE_DIR / "latent_valid_AERIS.npy")
    z_te = np.load(SCORE_DIR / "latent_test_AERIS.npy")
    y_te = np.load(SCORE_DIR / "y_test.npy").astype(int)
    y_va = np.load(SCORE_DIR / "y_valid.npy").astype(int)
    r_te = np.load(SCORE_DIR / "test__AERIS.npy")
    r_va = np.load(SCORE_DIR / "valid__AERIS.npy")

    *_, meta = build_matrices(quantile=0.98, horizon=1, seq_len=8,
                              normalisation="full", threshold_source="full")
    frame = meta["test_frame"].copy()
    frame["ace_abs"] = frame["ace"].abs()
    assert len(frame) == len(y_te), (len(frame), len(y_te))

    # ---- bandwidth chosen by validation log-likelihood --------------------
    bw_rows = []
    best_bw, best_ll = None, -np.inf
    for bw in BANDWIDTHS:
        kde = KernelDensity(kernel="gaussian", bandwidth=bw).fit(z_tr)
        ll = float(kde.score(z_va) / len(z_va))
        bw_rows.append({"Bandwidth": bw, "Mean validation log-density": ll})
        if ll > best_ll:
            best_ll, best_bw = ll, bw
    pd.DataFrame(bw_rows).to_csv(REV_TABLE_DIR / "rev06_bandwidth_selection.csv",
                                 index=False, encoding="utf-8-sig")
    print(f"selected bandwidth = {best_bw}  (mean validation log-density {best_ll:.3f})")

    kde = KernelDensity(kernel="gaussian", bandwidth=best_bw).fit(z_tr)
    log_tr = kde.score_samples(z_tr)
    c_te = normalised_density(kde, z_te, log_tr)
    c_va = normalised_density(kde, z_va, log_tr)
    print(f"  density confidence on test: median {np.median(c_te):.3f}, "
          f"5th-95th percentile {np.percentile(c_te, 5):.3f}-{np.percentile(c_te, 95):.3f}")

    # ---- Q1 density of events vs non-events ------------------------------
    pos, neg = c_te[y_te == 1], c_te[y_te == 0]
    u = stats.mannwhitneyu(pos, neg, alternative="two-sided")
    q1 = {
        "Median density confidence | event": float(np.median(pos)),
        "Median density confidence | non-event": float(np.median(neg)),
        "Mean density confidence | event": float(np.mean(pos)),
        "Mean density confidence | non-event": float(np.mean(neg)),
        "Rank AUC (low density predicts event)": float(
            1 - u.statistic / (len(pos) * len(neg))),
        "Mann-Whitney p (two-sided)": float(u.pvalue),
    }
    pd.DataFrame([q1]).to_csv(REV_TABLE_DIR / "rev06_density_separation.csv",
                              index=False, encoding="utf-8-sig")

    # ---- Q2 severity of low-density alerts inside the 2% budget ----------
    alert_idx = topk_indices(r_te, 0.02)
    conf_alerts = c_te[alert_idx]
    split = np.median(conf_alerts)
    low = alert_idx[conf_alerts <= split]
    high = alert_idx[conf_alerts > split]
    sev_rows = []
    for col in SEVERITY + ["future_event_proxy_score"]:
        v = frame[col].to_numpy(dtype=float)
        lo_v, hi_v = v[low], v[high]
        sev_rows.append({
            "Indicator": col,
            "Median | low-density alerts": float(np.median(lo_v)),
            "Median | high-density alerts": float(np.median(hi_v)),
            "Ratio": float(np.median(lo_v) / max(abs(np.median(hi_v)), 1e-9)),
            "Mann-Whitney p (two-sided)": float(
                stats.mannwhitneyu(lo_v, hi_v, alternative="two-sided").pvalue),
        })
    sev_rows.append({
        "Indicator": "Realised event rate among alerts",
        "Median | low-density alerts": float(y_te[low].mean()),
        "Median | high-density alerts": float(y_te[high].mean()),
        "Ratio": float(y_te[low].mean() / max(y_te[high].mean(), 1e-9)),
        "Mann-Whitney p (two-sided)": float(
            stats.mannwhitneyu(y_te[low].astype(float), y_te[high].astype(float),
                               alternative="two-sided").pvalue),
    })
    pd.DataFrame(sev_rows).to_csv(REV_TABLE_DIR / "rev06_alert_severity_split.csv",
                                  index=False, encoding="utf-8-sig")

    # ---- Q3/Q4 alpha sweep, validation-selected alpha --------------------
    sweep = []
    for a in ALPHAS:
        d_va = decision_index(r_va, c_va, a)
        d_te = decision_index(r_te, c_te, a)
        sweep.append({
            "alpha": float(a),
            "Valid Precision@2%": metric_value(y_va, d_va, "Precision@2"),
            "Valid Recall@2%": metric_value(y_va, d_va, "Recall@2"),
            "Valid PR_AUC": metric_value(y_va, d_va, "PR_AUC"),
            "Test Precision@1%": metric_value(y_te, d_te, "Precision@1"),
            "Test Recall@1%": metric_value(y_te, d_te, "Recall@1"),
            "Test Precision@2%": metric_value(y_te, d_te, "Precision@2"),
            "Test Recall@2%": metric_value(y_te, d_te, "Recall@2"),
            "Test PR_AUC": metric_value(y_te, d_te, "PR_AUC"),
            "Test cost (ratio 20)": expected_cost(y_te, d_te, 0.02, 20),
            "Test cost (ratio 50)": expected_cost(y_te, d_te, 0.02, 50),
        })
    sweep_df = pd.DataFrame(sweep)
    sweep_df.to_csv(REV_TABLE_DIR / "rev06_alpha_sweep.csv", index=False, encoding="utf-8-sig")

    best_row = sweep_df.sort_values(["Valid Recall@2%", "Valid Precision@2%"],
                                    ascending=False).iloc[0]
    alpha_star = float(best_row["alpha"])
    d_te_star = decision_index(r_te, c_te, alpha_star)
    diff = paired_bootstrap_diff(y_te, d_te_star, r_te, "Recall@2", block=96, rounds=2000)
    diff_p = paired_bootstrap_diff(y_te, d_te_star, r_te, "Precision@2", block=96, rounds=2000)

    summary = {
        "selected_bandwidth": best_bw,
        "alpha_selected_on_validation": alpha_star,
        "test_recall_at_2pct_alpha0": metric_value(y_te, r_te, "Recall@2"),
        "test_recall_at_2pct_alpha_star": metric_value(y_te, d_te_star, "Recall@2"),
        "test_precision_at_2pct_alpha0": metric_value(y_te, r_te, "Precision@2"),
        "test_precision_at_2pct_alpha_star": metric_value(y_te, d_te_star, "Precision@2"),
        "block_bootstrap_recall_diff": diff,
        "block_bootstrap_precision_diff": diff_p,
        "density_separation": q1,
    }
    (REV_TABLE_DIR / "rev06_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== Q1 density separation ==="); print(json.dumps(q1, indent=2))
    print("\n=== Q2 alert severity split ===")
    print(pd.DataFrame(sev_rows).to_string(index=False))
    print("\n=== Q3 alpha sweep (subset) ===")
    print(sweep_df[sweep_df["alpha"].isin([0.0, 0.2, 0.5, 1.0, 1.5, 2.0])].to_string(index=False))
    print(f"\nalpha* = {alpha_star};  Recall@2% {summary['test_recall_at_2pct_alpha0']:.2f} -> "
          f"{summary['test_recall_at_2pct_alpha_star']:.2f}  "
          f"(block bootstrap two-sided p = {diff['p_two_sided']:.4f})")


if __name__ == "__main__":
    main()
