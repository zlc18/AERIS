"""rev06b - Quantitative validation of the latent-density confidence layer.

Answers Reviewer 1 comment 7 and Reviewer 2 comment 5.

The first version of this work described the density layer qualitatively. Making
it quantitative exposed a real problem: a Gaussian kernel density estimate in the
full 16-dimensional latent space is degenerate. Almost every test window falls
below the lowest training density, so the normalised confidence collapses to
zero and carries no usable information. That is the curse of dimensionality, not
an implementation detail, and it has to be fixed before the layer can be
evaluated at all.

The density is therefore estimated on a principal-component projection of the
latent state, and the projection width, the kernel bandwidth and the weight of
the confidence term in the decision index are selected jointly on the validation
partition, for the purpose the layer actually serves: ranking alerts. The test
partition is used once, at the end.

Usage: python scripts/revision/rev06b_kde_confidence.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.revision.rev_common import REV_TABLE_DIR, build_matrices  # noqa: E402
from scripts.revision.rev_eval import (expected_cost, metric_value,  # noqa: E402
                                       paired_bootstrap_diff, topk_indices)

SCORE_DIR = Path(__file__).resolve().parents[2] / "outputs" / "revision" / "scores"
COMPONENTS = [2, 3, 4, 8]
BANDWIDTHS = [0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5]
ALPHAS = np.round(np.arange(0.0, 2.01, 0.1), 2)
SEVERITY = ["imbalanceprice_abs", "price_spread", "alpha", "ace_abs", "systemimbalance_abs"]


def confidence(kde: KernelDensity, z: np.ndarray, reference_sorted: np.ndarray) -> np.ndarray:
    """Position of a query's log-density within the training log-density
    distribution: 0 means more atypical than anything seen in training."""
    log_d = kde.score_samples(z)
    return np.searchsorted(reference_sorted, log_d, side="right") / len(reference_sorted)


def decision_index(risk: np.ndarray, conf: np.ndarray, alpha: float) -> np.ndarray:
    return risk * (1.0 + alpha * (1.0 - conf))


def main() -> None:
    z_tr = np.load(SCORE_DIR / "latent_train_AERIS.npy")
    z_va = np.load(SCORE_DIR / "latent_valid_AERIS.npy")
    z_te = np.load(SCORE_DIR / "latent_test_AERIS.npy")
    y_va = np.load(SCORE_DIR / "y_valid.npy").astype(int)
    y_te = np.load(SCORE_DIR / "y_test.npy").astype(int)
    r_va = np.load(SCORE_DIR / "valid__AERIS.npy")
    r_te = np.load(SCORE_DIR / "test__AERIS.npy")

    *_, meta = build_matrices(quantile=0.98, horizon=1, seq_len=8,
                              normalisation="full", threshold_source="full")
    frame = meta["test_frame"].copy()
    frame["ace_abs"] = frame["ace"].abs()

    # ---- joint selection of projection, bandwidth and weight on validation
    grid, best = [], None
    for k in COMPONENTS:
        pca = PCA(n_components=k, random_state=42).fit(z_tr)
        p_tr, p_va = pca.transform(z_tr), pca.transform(z_va)
        for bw in BANDWIDTHS:
            kde = KernelDensity(kernel="gaussian", bandwidth=bw).fit(p_tr)
            ref = np.sort(kde.score_samples(p_tr))
            c_va = confidence(kde, p_va, ref)
            spread = float(np.percentile(c_va, 95) - np.percentile(c_va, 5))
            for alpha in ALPHAS:
                v = metric_value(y_va, decision_index(r_va, c_va, alpha), "PR_AUC")
                grid.append({"Components": k, "Bandwidth": bw, "alpha": float(alpha),
                             "Valid PR_AUC": v, "Confidence spread": spread})
                if best is None or v > best[0]:
                    best = (v, k, bw, float(alpha))
    pd.DataFrame(grid).to_csv(REV_TABLE_DIR / "rev06b_selection_grid.csv",
                              index=False, encoding="utf-8-sig")
    v_best, k, bw, alpha = best
    print(f"  selected: {k} components, bandwidth {bw}, alpha {alpha} "
          f"(validation PR-AUC {v_best:.4f})", flush=True)

    pca = PCA(n_components=k, random_state=42).fit(z_tr)
    kde = KernelDensity(kernel="gaussian", bandwidth=bw).fit(pca.transform(z_tr))
    ref = np.sort(kde.score_samples(pca.transform(z_tr)))
    c_va = confidence(kde, pca.transform(z_va), ref)
    c_te = confidence(kde, pca.transform(z_te), ref)
    print(f"  test confidence: median {np.median(c_te):.3f}, "
          f"5th-95th percentile {np.percentile(c_te, 5):.3f}-{np.percentile(c_te, 95):.3f}",
          flush=True)

    # ---- Q1 is the density of event windows lower? -----------------------
    pos, neg = c_te[y_te == 1], c_te[y_te == 0]
    u = stats.mannwhitneyu(pos, neg, alternative="two-sided")
    q1 = {
        "Median confidence | event": float(np.median(pos)),
        "Median confidence | non-event": float(np.median(neg)),
        "Mean confidence | event": float(np.mean(pos)),
        "Mean confidence | non-event": float(np.mean(neg)),
        "Rank AUC (low density predicts event)": float(1 - u.statistic / (len(pos) * len(neg))),
        "Mann-Whitney p (two-sided)": float(u.pvalue),
    }
    pd.DataFrame([q1]).to_csv(REV_TABLE_DIR / "rev06b_density_separation.csv",
                              index=False, encoding="utf-8-sig")
    print("\n=== Q1 density separation ===")
    print(json.dumps(q1, indent=2))

    # ---- Q2 are low-density alerts more severe? --------------------------
    alert_idx = topk_indices(r_te, 0.02)
    conf_alerts = c_te[alert_idx]
    split = float(np.median(conf_alerts))
    low = alert_idx[conf_alerts <= split]
    high = alert_idx[conf_alerts > split]
    sev_rows = []
    if len(low) >= 5 and len(high) >= 5:
        for col in SEVERITY + ["future_event_proxy_score"]:
            v = frame[col].to_numpy(dtype=float)
            sev_rows.append({
                "Indicator": col,
                "Median | low-density alerts": float(np.median(v[low])),
                "Median | high-density alerts": float(np.median(v[high])),
                "Ratio": float(np.median(v[low]) / max(abs(np.median(v[high])), 1e-9)),
                "Mann-Whitney p (two-sided)": float(
                    stats.mannwhitneyu(v[low], v[high], alternative="two-sided").pvalue),
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
        pd.DataFrame(sev_rows).to_csv(REV_TABLE_DIR / "rev06b_alert_severity_split.csv",
                                      index=False, encoding="utf-8-sig")
        print("\n=== Q2 severity of low-density alerts (2% budget) ===")
        print(pd.DataFrame(sev_rows).to_string(index=False))
    else:
        print(f"\n  Q2 skipped: split gives {len(low)} / {len(high)} alerts")

    # ---- Q3 effect of the confidence-aware decision index ----------------
    sweep = []
    for a in ALPHAS:
        d_te = decision_index(r_te, c_te, a)
        d_va = decision_index(r_va, c_va, a)
        sweep.append({
            "alpha": float(a),
            "Valid PR_AUC": metric_value(y_va, d_va, "PR_AUC"),
            "Test PR_AUC": metric_value(y_te, d_te, "PR_AUC"),
            "Test Precision@1%": metric_value(y_te, d_te, "Precision@1"),
            "Test Recall@1%": metric_value(y_te, d_te, "Recall@1"),
            "Test Precision@2%": metric_value(y_te, d_te, "Precision@2"),
            "Test Recall@2%": metric_value(y_te, d_te, "Recall@2"),
            "Test cost (kappa=20)": expected_cost(y_te, d_te, 0.02, 20),
            "Test cost (kappa=50)": expected_cost(y_te, d_te, 0.02, 50),
        })
    sweep_df = pd.DataFrame(sweep)
    sweep_df.to_csv(REV_TABLE_DIR / "rev06b_alpha_sweep.csv", index=False, encoding="utf-8-sig")

    d_star = decision_index(r_te, c_te, alpha)
    diff_r = paired_bootstrap_diff(y_te, d_star, r_te, "Recall@2", block=96, rounds=2000)
    diff_p = paired_bootstrap_diff(y_te, d_star, r_te, "PR_AUC", block=96, rounds=2000)
    summary = {
        "selected": {"components": k, "bandwidth": bw, "alpha": alpha},
        "validation_pr_auc": v_best,
        "test_pr_auc_alpha0": metric_value(y_te, r_te, "PR_AUC"),
        "test_pr_auc_alpha_star": metric_value(y_te, d_star, "PR_AUC"),
        "test_recall2_alpha0": metric_value(y_te, r_te, "Recall@2"),
        "test_recall2_alpha_star": metric_value(y_te, d_star, "Recall@2"),
        "paired_bootstrap_recall": diff_r,
        "paired_bootstrap_pr_auc": diff_p,
        "density_separation": q1,
    }
    (REV_TABLE_DIR / "rev06b_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== Q3 decision index ===")
    print(sweep_df[sweep_df["alpha"].isin([0.0, 0.5, 1.0, 1.5, 2.0])].to_string(index=False))
    print(f"\n  alpha* = {alpha}: PR-AUC {summary['test_pr_auc_alpha0']:.4f} -> "
          f"{summary['test_pr_auc_alpha_star']:.4f}  (p = {diff_p['p_two_sided']:.3f});  "
          f"Recall@2% {summary['test_recall2_alpha0']:.2f} -> "
          f"{summary['test_recall2_alpha_star']:.2f}  (p = {diff_r['p_two_sided']:.3f})")


if __name__ == "__main__":
    main()
