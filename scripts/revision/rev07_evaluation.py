"""rev07 - Unified evaluation: interval estimates, time-series-aware inference,
alert-budget curves, calibration and cost-sensitive operational utility.

Answers Reviewer 1 comments 5, 6 and 8 and Reviewer 2 comment 4.  All numbers
are computed from the single set of test scores produced by rev02 and rev03,
so every model is evaluated on identical predictions.

Usage: python scripts/revision/rev07_evaluation.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.revision.rev_common import REV_FIG_DIR, REV_TABLE_DIR, load_all_scores  # noqa: E402
from scripts.revision.rev_eval import (bootstrap_ci, expected_cost, make_index_sets,  # noqa: E402
                                       metric_value, paired_bootstrap_diff, select_threshold,
                                       topk_indices)

SCORE_DIR = Path(__file__).resolve().parents[2] / "outputs" / "revision" / "scores"
PROPOSED = "AERIS"
BASELINES_FOR_TEST = ["HistGBM (regression, tuned)", "XGBoost (regression, tuned)",
                      "LightGBM (regression, tuned)",
                      "Heuristic: composite score persistence", "LightGBM (tuned)",
                      "XGBoost (tuned)", "HistGBM (tuned)", "HistGBM", "GBM", "RF", "CART",
                      "Transformer classifier", "LSTM classifier", "GRU classifier",
                      "TCN classifier", "AERIS-DB", "Logistic Regression",
                      "VAE reconstruction (unsupervised)", "Isolation Forest"]
METRICS = ["PR_AUC", "ROC_AUC", "F1", "Precision@1%", "Recall@1%",
           "Precision@2%", "Recall@2%", "Brier"]
BUDGETS = np.round(np.arange(0.005, 0.0501, 0.0025), 5)
COST_RATIOS = [5, 10, 20, 50, 100, 200]
PALETTE = {
    PROPOSED: "#1F4E79", "XGBoost (tuned)": "#D55E00", "LightGBM (tuned)": "#E69F00",
    "HistGBM (tuned)": "#CC79A7", "HistGBM": "#CC79A7",
    "Heuristic: composite score persistence": "#000000",
    "GBM": "#388E3C", "RF": "#2E7D32", "CART": "#808080",
    "Transformer classifier": "#9B59B6", "LSTM classifier": "#0072B2",
    "GRU classifier": "#56B4E9", "TCN classifier": "#7F7F7F",
    "Logistic Regression": "#B15928", "AERIS (common recipe)": "#4C78A8",
    "VAE reconstruction (unsupervised)": "#A0522D", "Isolation Forest": "#999999",
    # the regression formulations are the closest competitors, so they need
    # colours of their own rather than the shared fallback grey
    "HistGBM (regression, tuned)": "#A23B72", "XGBoost (regression, tuned)": "#8C3B00",
    "LightGBM (regression, tuned)": "#B8860B",
}


# names shown in figure legends, matching the tables of the manuscript
DISPLAY = {
    "HistGBM (regression, tuned)": "HistGBM-R",
    "XGBoost (regression, tuned)": "XGBoost-R",
    "LightGBM (regression, tuned)": "LightGBM-R",
    "HistGBM (tuned)": "HistGBM-C",
    "XGBoost (tuned)": "XGBoost-C",
    "LightGBM (tuned)": "LightGBM-C",
    "Heuristic: composite score persistence": "Persistence",
    "LSTM classifier": "LSTM",
    "Transformer classifier": "Transformer",
}

def load_scores() -> tuple[np.ndarray, np.ndarray, dict, dict]:
    y_te = np.load(SCORE_DIR / "y_test.npy").astype(int)
    y_va = np.load(SCORE_DIR / "y_valid.npy").astype(int)
    valid, test = load_all_scores()
    return y_te, y_va, test, valid


def main() -> None:
    y_te, y_va, test, valid = load_scores()
    print(f"loaded {len(test)} score sets; test n={len(y_te)}, positives={y_te.sum()}")

    if PROPOSED not in test:
        raise SystemExit(f"no scores for {PROPOSED}; run rev10 first")
    thresholds = {n: select_threshold(y_va, valid[n]) for n in test if n in valid}

    # One set of resampled index sets per scheme, shared by every model and metric,
    # so that all intervals and all paired comparisons use identical resamples.
    ci_idx = make_index_sets(len(y_te), block=96, rounds=1000, seed=42)
    paired_block = make_index_sets(len(y_te), block=96, rounds=2000, seed=7)
    paired_iid = make_index_sets(len(y_te), block=1, rounds=2000, seed=7)

    # ---- 1. interval estimates for every principal metric ---------------
    rows = []
    for name, s in test.items():
        thr = thresholds.get(name)
        row = {"Model": name}
        for m in METRICS:
            if m == "Brier" and (s.min() < 0 or s.max() > 1):
                row[m] = np.nan
                row[f"{m}_lo"] = row[f"{m}_hi"] = np.nan
                continue
            try:
                r = bootstrap_ci(y_te, s, m, thr, index_sets=ci_idx)
            except ValueError:
                continue
            row[m] = r["value"]
            row[f"{m}_lo"] = r["ci_lower"]
            row[f"{m}_hi"] = r["ci_upper"]
        row["Threshold"] = thr
        rows.append(row)
        print(f"  {name:38s} PR-AUC={row['PR_AUC']:.4f} "
              f"[{row['PR_AUC_lo']:.4f}, {row['PR_AUC_hi']:.4f}]", flush=True)
    metrics_df = pd.DataFrame(rows).sort_values("PR_AUC", ascending=False)
    metrics_df.to_csv(REV_TABLE_DIR / "rev07_metrics_with_ci.csv", index=False, encoding="utf-8-sig")

    # ---- 2. paired block bootstrap vs each baseline ----------------------
    sig_rows = []
    for base in BASELINES_FOR_TEST:
        if base not in test:
            continue
        for metric in ["PR_AUC", "Recall@2%", "Precision@2%"]:
            for block, idx in ((1, paired_iid), (96, paired_block)):
                d = paired_bootstrap_diff(y_te, test[PROPOSED], test[base], metric,
                                          thresholds.get(PROPOSED), thresholds.get(base),
                                          block=block, index_sets=idx)
                sig_rows.append({"Baseline": base, "Metric": metric,
                                 "Resampling": "i.i.d." if block == 1 else "moving block (24 h)",
                                 **d})
            print(f"    tested {base} on {metric}", flush=True)
    sig_df = pd.DataFrame(sig_rows)
    sig_df.to_csv(REV_TABLE_DIR / "rev07_significance.csv", index=False, encoding="utf-8-sig")

    # ---- 3. alert-budget curves -----------------------------------------
    budget_rows = []
    for name, s in test.items():
        for b in BUDGETS:
            idx = topk_indices(s, b)
            hits = int(y_te[idx].sum())
            budget_rows.append({
                "Model": name, "Budget": float(b), "TopK": len(idx), "Hits": hits,
                "Precision@K": 100 * hits / len(idx),
                "Recall@K": 100 * hits / max(1, y_te.sum()),
                "Missed events": int(y_te.sum()) - hits,
                "Lift": (hits / len(idx)) / max(y_te.mean(), 1e-9),
            })
    budget_df = pd.DataFrame(budget_rows)
    budget_df.to_csv(REV_TABLE_DIR / "rev07_alert_budget.csv", index=False, encoding="utf-8-sig")

    # ---- 4. cost-sensitive utility --------------------------------------
    cost_rows = []
    for name, s in test.items():
        for ratio in COST_RATIOS:
            costs = [(b, expected_cost(y_te, s, b, ratio)) for b in BUDGETS]
            b_star, c_star = min(costs, key=lambda t: t[1])
            c_none = ratio * y_te.mean()          # alert on nothing
            c_all = 1.0                            # alert on everything
            cost_rows.append({
                "Model": name, "Cost ratio (miss/alert)": ratio,
                "Best budget (%)": 100 * b_star,
                "Cost per interval": c_star,
                "Cost at fixed 2% budget": expected_cost(y_te, s, 0.02, ratio),
                "Cost of no-alert policy": c_none,
                "Cost of alert-all policy": c_all,
                "Saving vs no-alert (%)": 100 * (1 - c_star / c_none) if c_none > 0 else np.nan,
            })
    cost_df = pd.DataFrame(cost_rows)
    cost_df.to_csv(REV_TABLE_DIR / "rev07_cost_utility.csv", index=False, encoding="utf-8-sig")

    # ---- 5. calibration --------------------------------------------------
    calib_rows = []
    for name, s in test.items():
        if s.min() < 0 or s.max() > 1:
            continue
        bins = np.quantile(s, np.linspace(0, 1, 11))
        bins = np.unique(bins)
        idx = np.clip(np.digitize(s, bins[1:-1]), 0, len(bins) - 2)
        for b in range(len(bins) - 1):
            m = idx == b
            if m.sum() == 0:
                continue
            calib_rows.append({"Model": name, "Bin": b + 1, "n": int(m.sum()),
                               "Mean predicted": float(s[m].mean()),
                               "Observed frequency": float(y_te[m].mean())})
    pd.DataFrame(calib_rows).to_csv(REV_TABLE_DIR / "rev07_calibration.csv",
                                    index=False, encoding="utf-8-sig")

    # ---- 6. onset subset: intervals the persistence rule does not already flag
    from scripts.revision.rev_common import build_matrices
    *_, meta = build_matrices(quantile=0.98, horizon=1, seq_len=8,
                              normalisation="full", threshold_source="full")
    current = meta["test_frame"]["event_proxy_score"].to_numpy(dtype=float)
    onset = current < meta["threshold"]
    onset_rows = []
    if onset.sum() > 0 and y_te[onset].sum() > 0:
        y_on = y_te[onset]
        for name, sc in test.items():
            s_on = sc[onset]
            onset_rows.append({
                "Model": name,
                "PR_AUC": metric_value(y_on, s_on, "PR_AUC"),
                "Recall@2%": metric_value(y_on, s_on, "Recall@2"),
                "Precision@2%": metric_value(y_on, s_on, "Precision@2"),
                "Recall@5%": metric_value(y_on, s_on, "Recall@5"),
            })
        onset_df = pd.DataFrame(onset_rows).sort_values("PR_AUC", ascending=False)
        onset_df.to_csv(REV_TABLE_DIR / "rev07_onset_subset.csv",
                        index=False, encoding="utf-8-sig")
        (REV_TABLE_DIR / "rev07_onset_meta.json").write_text(json.dumps({
            "n_intervals": int(onset.sum()),
            "n_events": int(y_on.sum()),
            "prevalence_pct": 100 * float(y_on.mean()),
            "threshold": float(meta["threshold"]),
        }, indent=2), encoding="utf-8")
        print("\n=== onset subset (current state below the event threshold) ===")
        print(f"  {int(onset.sum())} intervals, {int(y_on.sum())} events "
              f"({100 * y_on.mean():.2f}%)")
        print(onset_df.head(8).to_string(index=False))

    # ---- figures ---------------------------------------------------------
    # every method that leads the proposed model at any budget or any cost ratio has
    # to be on these curves, not only in the full table
    plot_models = [m for m in [PROPOSED,
                               "HistGBM (regression, tuned)", "XGBoost (regression, tuned)",
                               "LightGBM (regression, tuned)",
                               "LightGBM (tuned)", "XGBoost (tuned)", "HistGBM (tuned)",
                               "RF", "Heuristic: composite score persistence",
                               "LSTM classifier", "Transformer classifier", "GBM", "CART"]
                   if m in test]
    make_pr_figure(y_te, test, plot_models)
    make_budget_figure(budget_df, plot_models)
    make_cost_figure(cost_df, plot_models)
    make_significance_figure(sig_df)
    make_calibration_figure(pd.DataFrame(calib_rows), plot_models)

    (REV_TABLE_DIR / "rev07_meta.json").write_text(json.dumps({
        "test_size": int(len(y_te)), "test_positives": int(y_te.sum()),
        "prevalence_pct": 100 * float(y_te.mean()),
        "block_length_intervals": 96, "bootstrap_rounds": 2000,
    }, indent=2), encoding="utf-8")
    print("\n=== metrics ===")
    print(metrics_df[["Model", "PR_AUC", "PR_AUC_lo", "PR_AUC_hi", "F1",
                      "Precision@2%", "Recall@2%"]].to_string(index=False))


def _style(ax):
    ax.grid(alpha=0.25, linestyle="--")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def make_pr_figure(y, test, models):
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    for m in models:
        p, r, _ = precision_recall_curve(y, test[m])
        ax.plot(r, p, lw=2.0 if m == PROPOSED else 1.4,
                color=PALETTE.get(m, "#555555"),
                label=f"{m} (PR-AUC {metric_value(y, test[m], 'PR_AUC'):.3f})")
    ax.axhline(y.mean(), color="#999999", ls=":", lw=1.2,
               label=f"Prevalence ({100 * y.mean():.2f}%)")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(fontsize=7.5, loc="upper right", frameon=False)
    _style(ax); fig.tight_layout()
    fig.savefig(REV_FIG_DIR / "figR02_pr_curves.pdf", bbox_inches="tight")
    plt.close(fig)


def make_budget_figure(budget_df, models):
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))
    for m in models:
        sub = budget_df[budget_df["Model"] == m].sort_values("Budget")
        x = 100 * sub["Budget"]
        axes[0].plot(x, sub["Recall@K"], lw=2.0 if m == PROPOSED else 1.3,
                     marker="o" if m == PROPOSED else None, ms=3.5,
                     color=PALETTE.get(m, "#555555"), label=DISPLAY.get(m, m))
        axes[1].plot(x, sub["Precision@K"], lw=2.0 if m == PROPOSED else 1.3,
                     marker="o" if m == PROPOSED else None, ms=3.5,
                     color=PALETTE.get(m, "#555555"), label=DISPLAY.get(m, m))
    axes[0].set_xlabel("Alert budget (% of test intervals)"); axes[0].set_ylabel("Recall@K (%)")
    axes[1].set_xlabel("Alert budget (% of test intervals)"); axes[1].set_ylabel("Precision@K (%)")
    for a in axes:
        _style(a)
    # one shared legend below the panels: with thirteen curves an inset legend
    # covers the very region the figure is about
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=7.5, frameon=False, ncol=4,
               loc="upper center", bbox_to_anchor=(0.5, 0.06))
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    # every method; the manuscript figure with the leading methods is drawn by rev11
    fig.savefig(REV_FIG_DIR / "figR06_alarm_budget_all.pdf", bbox_inches="tight")
    plt.close(fig)


def make_cost_figure(cost_df, models):
    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    for m in models:
        sub = cost_df[cost_df["Model"] == m].sort_values("Cost ratio (miss/alert)")
        ax.plot(sub["Cost ratio (miss/alert)"], sub["Saving vs no-alert (%)"],
                marker="o", ms=4, lw=2.0 if m == PROPOSED else 1.3,
                color=PALETTE.get(m, "#555555"), label=DISPLAY.get(m, m))
    ax.set_xscale("log")
    ax.set_xlabel("Cost ratio $C_{\\mathrm{miss}}/C_{\\mathrm{alert}}$")
    ax.set_ylabel("Reduction in expected operational cost\nrelative to a no-alert policy (%)")
    ax.axhline(0, color="#999999", lw=1.0)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=7.5, frameon=False, ncol=3,
               loc="upper center", bbox_to_anchor=(0.5, 0.06))
    _style(ax); fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(REV_FIG_DIR / "figR15_cost_utility.pdf", bbox_inches="tight")
    plt.close(fig)


def make_significance_figure(sig_df):
    sub = sig_df[(sig_df["Metric"] == "PR_AUC")].copy()
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(7.4, 0.42 * len(sub) + 1.4))
    ypos = np.arange(len(sub))[::-1]
    colors = ["#1F4E79" if r == "moving block (24 h)" else "#A6BDD7"
              for r in sub["Resampling"]]
    ax.errorbar(sub["mean_diff"], ypos,
                xerr=[sub["mean_diff"] - sub["ci_lower"], sub["ci_upper"] - sub["mean_diff"]],
                fmt="o", ms=4, lw=1.4, ecolor=colors, mfc="#1F4E79", mec="#1F4E79", ls="none")
    ax.axvline(0, color="#C0392B", ls="--", lw=1.2)
    ax.set_yticks(ypos)
    ax.set_yticklabels([f"{b} - {r}" for b, r in zip(sub["Baseline"], sub["Resampling"])],
                       fontsize=7.5)
    ax.set_xlabel("PR-AUC difference (AERIS minus baseline), 95% bootstrap interval")
    _style(ax); fig.tight_layout()
    fig.savefig(REV_FIG_DIR / "figR07_significance.pdf", bbox_inches="tight")
    plt.close(fig)


def make_calibration_figure(calib, models):
    if calib.empty:
        return
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    for m in models:
        sub = calib[calib["Model"] == m]
        if sub.empty:
            continue
        ax.plot(sub["Mean predicted"], sub["Observed frequency"], marker="o", ms=4,
                lw=1.8 if m == PROPOSED else 1.2, color=PALETTE.get(m, "#555555"), label=DISPLAY.get(m, m))
    lim = [0, 1]
    ax.plot(lim, lim, ls=":", color="#999999", lw=1.2, label="Perfect calibration")
    ax.set_xlabel("Mean predicted risk in decile"); ax.set_ylabel("Observed event frequency")
    ax.legend(fontsize=7.5, frameon=False)
    _style(ax); fig.tight_layout()
    fig.savefig(REV_FIG_DIR / "figR16_calibration.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
