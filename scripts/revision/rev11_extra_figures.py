"""rev11 - Figures for the label validation, ablation, density layer, horizons
and robustness sections.  Everything is drawn from the CSV tables written by the
other revision scripts, so figures and text cannot drift apart.

Usage: python scripts/revision/rev11_extra_figures.py
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.revision.rev_common import REV_FIG_DIR, REV_TABLE_DIR  # noqa: E402

BLUE, ORANGE, GREY = "#1F4E79", "#D55E00", "#8C8C8C"
GREEN = "#009E73"
# Times New Roman throughout, matching the manuscript; TrueType embedding in the PDFs
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"],
                     "mathtext.fontset": "stix", "pdf.fonttype": 42, "ps.fonttype": 42})


def style(ax):
    ax.grid(alpha=0.25, linestyle="--")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def fig_label_validation():
    path = REV_TABLE_DIR / "rev01_label_external_validation.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    df = df[df["External to label"].str.startswith("yes")].copy()
    df["Short"] = df["Indicator"].str.replace(r"\s*\(.*\)", "", regex=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))

    order = df.sort_values("Risk ratio")
    y = np.arange(len(order))
    colors = [BLUE if r >= 1 else ORANGE for r in order["Risk ratio"]]
    axes[0].barh(y, order["Risk ratio"], color=colors, height=0.6)
    axes[0].axvline(1.0, color=GREY, ls="--", lw=1.2)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(order["Short"], fontsize=8)
    axes[0].set_xlabel("Risk ratio: P(indicator in its own top 5% | event)\n"
                       "divided by the same probability for non-events", fontsize=8.5)

    axes[1].barh(y, order["Rank AUC"], color=colors, height=0.6)
    axes[1].axvline(0.5, color=GREY, ls="--", lw=1.2)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([])
    axes[1].set_xlim(0, 1)
    axes[1].set_xlabel("Rank AUC of the indicator for the screening label", fontsize=8.5)
    for a in axes:
        style(a)
    fig.tight_layout()
    fig.savefig(REV_FIG_DIR / "figR01_label_validation.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_ablation():
    path = REV_TABLE_DIR / "rev04_ablation_summary.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    struct = df[df["Group"] == "structural"].copy()
    base = struct.loc[struct["Variant"] == "AERIS (full)", "PR_AUC_mean"].iloc[0]
    struct = struct.sort_values("PR_AUC_mean")
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4),
                             gridspec_kw={"width_ratios": [1.45, 1.0]})
    y = np.arange(len(struct))
    colors = [BLUE if v == "AERIS (full)" else ORANGE for v in struct["Variant"]]
    axes[0].barh(y, struct["PR_AUC_mean"], xerr=struct["PR_AUC_std"].fillna(0),
                 color=colors, height=0.62, error_kw={"lw": 1.0, "ecolor": "#444444"})
    axes[0].axvline(base, color=BLUE, ls="--", lw=1.2)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(struct["Variant"], fontsize=8)
    axes[0].set_xlabel("Test PR-AUC (mean over seeds)", fontsize=9)
    axes[0].set_xlim(0, max(0.75, struct["PR_AUC_mean"].max() * 1.15))

    for group, marker, colour, label in [("latent_dim", "o", BLUE, "Latent dimension"),
                                         ("sequence_length", "s", ORANGE, "Sequence length"),
                                         ("hidden_dim", "^", "#2E7D32", "Hidden dimension")]:
        sub = df[df["Group"] == group]
        if sub.empty:
            continue
        x = [float(v.split("=")[-1]) for v in sub["Variant"]]
        axes[1].errorbar(x, sub["PR_AUC_mean"], yerr=sub["PR_AUC_std"].fillna(0),
                         marker=marker, ms=5, lw=1.5, color=colour, capsize=3, label=label)
    axes[1].set_xscale("log", base=2)
    axes[1].set_xlabel("Hyper-parameter value (log scale)", fontsize=9)
    axes[1].set_ylabel("Test PR-AUC", fontsize=9)
    axes[1].legend(fontsize=8, frameon=False)
    for a in axes:
        style(a)
    fig.tight_layout()
    fig.savefig(REV_FIG_DIR / "figR17_ablation.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_kde():
    sweep_path = REV_TABLE_DIR / "rev06_alpha_sweep.csv"
    sep_path = REV_TABLE_DIR / "rev06_density_separation.csv"
    if not sweep_path.exists():
        return
    sweep = pd.read_csv(sweep_path)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))

    axes[0].plot(sweep["alpha"], sweep["Test Precision@2%"], marker="o", ms=4,
                 color=BLUE, label="Test Precision@2%")
    axes[0].plot(sweep["alpha"], sweep["Test Recall@2%"], marker="s", ms=4,
                 color=ORANGE, label="Test Recall@2%")
    axes[0].plot(sweep["alpha"], sweep["Valid Recall@2%"], marker="^", ms=4,
                 color=GREY, ls="--", label="Validation Recall@2%")
    axes[0].set_xlabel(r"Confidence weight $\alpha$", fontsize=9)
    axes[0].set_ylabel("Percent", fontsize=9)
    axes[0].legend(fontsize=8, frameon=False)

    axes[1].plot(sweep["alpha"], sweep["Test cost (ratio 20)"], marker="o", ms=4,
                 color=BLUE, label=r"$\kappa = 20$")
    axes[1].plot(sweep["alpha"], sweep["Test cost (ratio 50)"], marker="s", ms=4,
                 color=ORANGE, label=r"$\kappa = 50$")
    axes[1].set_xlabel(r"Confidence weight $\alpha$", fontsize=9)
    axes[1].set_ylabel("Expected cost per interval\n(units of one alert)", fontsize=9)
    axes[1].legend(fontsize=8, frameon=False)
    for a in axes:
        style(a)
    if sep_path.exists():
        sep = pd.read_csv(sep_path).iloc[0]
        axes[0].set_title(
            f"Latent density of events vs non-events: rank AUC "
            f"{sep['Rank AUC (low density predicts event)']:.3f}", fontsize=9)
    fig.tight_layout()
    fig.savefig(REV_FIG_DIR / "figR18_kde_confidence.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_horizon():
    """PR-AUC against screening horizon, from the per-horizon re-selected runs
    (rev27/rev30, rev33 for LightGBM-R) -- the same numbers as the horizon table."""
    path = REV_TABLE_DIR / "rev30_horizon_vs_persistence.json"
    if not path.exists():
        return
    h = json.loads(path.read_text(encoding="utf-8"))
    lg_path = REV_TABLE_DIR / "rev33_horizon_lightgbm.csv"
    if lg_path.exists():
        lg = pd.read_csv(lg_path)
        lg = lg[lg["Metric"] == "PR_AUC"].set_index("Horizon")["LightGBM"]
        for k in h:
            if k in lg.index:
                h[k]["LightGBM"] = float(lg[k])
    keys = sorted((k for k in h if 30 <= h[k]["minutes"] <= 60), key=lambda k: int(k[2:]))
    minutes = [h[k]["minutes"] for k in keys]
    series = [("AERIS", "AERIS", BLUE, "o", "-"),
              ("LightGBM", "LightGBM-R", GREEN, "D", "-"),
              ("HistGBM", "HistGBM-R", ORANGE, "s", "-"),
              ("Persistence", "Persistence", GREY, "^", "--")]
    series = [s_ for s_ in series if all(s_[0] in h[k] for k in keys)]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for key, label, colour, marker, ls in series:
        ax.plot(minutes, [h[k][key] for k in keys], marker=marker, ms=6, lw=2.0,
                ls=ls, color=colour, label=label)
    for x, k in zip(minutes, keys):
        ax.annotate(f"{h[k]['AERIS']:.3f}", (x, h[k]["AERIS"]), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8, color=BLUE, fontweight="bold")
    ax.set_xticks(minutes)
    ax.set_xlim(minutes[0] - 3, minutes[-1] + 3)
    ax.set_xlabel("Screening horizon (minutes ahead)", fontsize=10)
    ax.set_ylabel("Test PR-AUC", fontsize=10)
    values = [h[k][s_[0]] for k in keys for s_ in series]
    ax.set_ylim(0.05 * int(20 * min(values) - 1), 0.05 * int(20 * max(values) + 2))
    ax.legend(fontsize=9, frameon=False, loc="upper right")
    style(ax)
    fig.tight_layout()
    fig.savefig(REV_FIG_DIR / "figR19_horizon.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_alarm_budget():
    """Recall and precision against the alert budget for AERIS and the methods that
    compete with it (the tuned boosting models, the persistence rule and the LSTM,
    which leads at the widest budgets); rev07 draws every method separately."""
    path = REV_TABLE_DIR / "rev07_alert_budget.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    series = [("AERIS", "AERIS", BLUE, "-", 2.4),
              ("LightGBM (regression, tuned)", "LightGBM-R", GREEN, "-", 1.4),
              ("HistGBM (regression, tuned)", "HistGBM-R", ORANGE, "-", 1.4),
              ("XGBoost (regression, tuned)", "XGBoost-R", "#CC79A7", "-", 1.4),
              ("LightGBM (tuned)", "LightGBM-C", "#56B4E9", "-", 1.4),
              ("LSTM classifier", "LSTM", "#7B3294", "-", 1.4),
              ("Heuristic: composite score persistence", "Persistence", GREY, "--", 1.4)]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    for model, label, colour, ls, lw in series:
        sub = df[df["Model"] == model].sort_values("Budget")
        x = 100 * sub["Budget"]
        kw = dict(color=colour, ls=ls, lw=lw, label=label, zorder=3 if model == "AERIS" else 2,
                  marker="o" if model == "AERIS" else None, ms=4)
        axes[0].plot(x, sub["Recall@K"], **kw)
        axes[1].plot(x, sub["Precision@K"], **kw)
    for ax, ylabel in zip(axes, ("Recall@K (%)", "Precision@K (%)")):
        ax.axvline(2.0, color="#999999", lw=1.0, ls=":", zorder=1)
        ax.set_xlabel("Alert budget (% of test intervals)", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        style(ax)
    axes[0].text(2.05, axes[0].get_ylim()[0] + 2, "2% operating point", fontsize=8,
                 color="#666666", va="bottom")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8.5, frameon=False, ncol=7,
               loc="upper center", bbox_to_anchor=(0.5, 0.03))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(REV_FIG_DIR / "figR06_alarm_budget.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_recall_ranking():
    """Recall at the 2% operating point for every method of the main comparison,
    the same numbers as the main table."""
    from scripts.revision.rev12_make_tables import short
    path = REV_TABLE_DIR / "rev07_metrics_with_ci.csv"
    if not path.exists():
        return
    df = pd.read_csv(path).sort_values(["Recall@2%", "Model"], ascending=[True, False])
    names = [short(m) for m in df["Model"]]
    values = df["Recall@2%"].to_numpy()
    colours = [BLUE if m == "AERIS" else "#B8B8B8" for m in df["Model"]]
    fig, ax = plt.subplots(figsize=(5.0, 5.8))
    y = np.arange(len(df))
    ax.barh(y, values, color=colours, height=0.68, zorder=2)
    aeris = float(df.loc[df["Model"] == "AERIS", "Recall@2%"].iloc[0])
    ax.axvline(aeris, color=BLUE, lw=0.9, ls=":", zorder=1)
    for yi, v, m in zip(y, values, df["Model"]):
        ax.text(v + 0.8, yi, f"{v:.1f}", va="center", fontsize=8,
                color=BLUE if m == "AERIS" else "#444444",
                fontweight="bold" if m == "AERIS" else "normal")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8.5)
    for label in ax.get_yticklabels():
        if label.get_text() == "AERIS":
            label.set_fontweight("bold")
            label.set_color(BLUE)
    ax.set_xlim(0, 80)
    ax.set_ylim(-0.6, len(df) - 0.4)
    ax.set_xlabel("Recall at the 2% alert budget (%)", fontsize=10)
    style(ax)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(REV_FIG_DIR / "figR20_recall_ranking.pdf", bbox_inches="tight")
    plt.close(fig)


REALTIME_SETTINGS = [("Canonical inputs (146 features)", "Canonical"),
                     ("Training-only label normalisation and threshold", "Train-only\nlabel"),
                     ("One-interval publication lag on all inputs", "One-interval\nlag"),
                     ("Training-only label + publication lag", "Train-only\n+ lag")]


def realtime_audit() -> pd.DataFrame | None:
    """Test PR-AUC of each method in the four settings that move the protocol towards
    real-time use, assembled from the same files as the audit table."""
    r05 = REV_TABLE_DIR / "rev05_robustness.csv"
    r32 = REV_TABLE_DIR / "rev32_audit_tuned_regressors.csv"
    r32c = REV_TABLE_DIR / "rev32c_audit_tuned_classifier.csv"
    if not (r05.exists() and r32.exists() and r32c.exists()):
        return None
    c = pd.read_csv(r05)
    c = c[c["Part"] == "C_availability"]
    reg, clf = pd.read_csv(r32), pd.read_csv(r32c)
    rows = []
    for setting, _ in REALTIME_SETTINGS:
        block = c[c["Setting"] == setting].set_index("Model")["PR_AUC"]
        r = reg[reg["Setting"] == setting].set_index("Model")["PR_AUC"]
        rows.append({"Setting": setting,
                     "AERIS": block["AERIS"],
                     "LightGBM-R": r["LightGBM (regression, tuned)"],
                     "HistGBM-R": r["HistGBM (regression, tuned)"],
                     "HistGBM-C": clf.set_index("Setting").loc[setting, "PR_AUC"],
                     "Persistence": block["Persistence heuristic"]})
    return pd.DataFrame(rows).set_index("Setting")


def fig_realtime():
    """PR-AUC in the settings closest to real-time operation, and the share of the
    canonical PR-AUC each method keeps there."""
    df = realtime_audit()
    if df is None:
        return
    series = [("AERIS", BLUE, "o", "-", 2.4), ("LightGBM-R", GREEN, "D", "-", 1.4),
              ("HistGBM-R", ORANGE, "s", "-", 1.4), ("HistGBM-C", "#56B4E9", "v", "-", 1.4),
              ("Persistence", GREY, "^", "--", 1.4)]
    labels = [lab for _, lab in REALTIME_SETTINGS]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.5))

    width = 0.16
    for i, (name, colour, _, _, _) in enumerate(series):
        offset = (i - (len(series) - 1) / 2) * width
        axes[0].bar(x + offset, df[name].to_numpy(), width=width * 0.95, color=colour,
                    label=name, zorder=2)
        if name == "AERIS":
            for xi, v in zip(x + offset, df[name]):
                axes[0].text(xi, v + 0.012, f"{v:.3f}", ha="center", va="bottom",
                             fontsize=7.5, color=BLUE, fontweight="bold", rotation=90)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=9)
    axes[0].set_ylim(0, 0.78)
    axes[0].set_ylabel("Test PR-AUC", fontsize=10)
    axes[0].set_title("(a)", fontsize=10, loc="left")

    retained = 100 * df / df.iloc[0]
    for name, colour, marker, ls, lw in series:
        axes[1].plot(x, retained[name].to_numpy(), color=colour, marker=marker, ms=6 if
                     name == "AERIS" else 5, ls=ls, lw=lw, zorder=3 if name == "AERIS" else 2)
    for xi, v in zip(x[2:], retained["AERIS"].to_numpy()[2:]):
        axes[1].annotate(f"{v:.0f}%", (xi, v), textcoords="offset points", xytext=(0, 8),
                         ha="center", fontsize=8.5, color=BLUE, fontweight="bold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=9)
    axes[1].set_ylim(40, 108)
    axes[1].set_ylabel("PR-AUC retained (% of canonical)", fontsize=10)
    axes[1].set_title("(b)", fontsize=10, loc="left")
    for ax in axes:
        style(ax)
    axes[0].grid(axis="x", visible=False)
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, fontsize=9, frameon=False, ncol=len(series),
               loc="upper center", bbox_to_anchor=(0.5, 0.03))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(REV_FIG_DIR / "figR21_realtime.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_seed_stability():
    path = REV_TABLE_DIR / "rev03_sequence_per_seed.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    models = list(df["Model"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))
    for ax, metric in zip(axes, ["PR_AUC", "Recall@2%"]):
        data = [df.loc[df["Model"] == m, metric].to_numpy() for m in models]
        parts = ax.violinplot(data, showextrema=False)
        for pc in parts["bodies"]:
            pc.set_facecolor("#C8D7E8")
            pc.set_alpha(0.7)
        ax.boxplot(data, widths=0.18, showfliers=False,
                   medianprops={"color": BLUE, "lw": 1.6})
        ax.set_xticks(np.arange(1, len(models) + 1))
        ax.set_xticklabels(models, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel(metric.replace("_", "-"), fontsize=9)
        style(ax)
    fig.tight_layout()
    fig.savefig(REV_FIG_DIR / "figR03_seed_stability.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_perturbation():
    path = REV_TABLE_DIR / "rev05_robustness.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    e = df[df["Part"] == "E_perturbation"]
    if e.empty:
        return
    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    x = np.arange(len(e))
    ax.bar(x, e["PR_AUC"], color=BLUE, width=0.6)
    ref = df[(df["Part"] == "C_availability") & (df["Model"] == "HistGBM")]
    if not ref.empty:
        ax.axhline(ref["PR_AUC"].iloc[0], color=ORANGE, ls="--", lw=1.3,
                   label=f"Tuned boosting reference ({ref['PR_AUC'].iloc[0]:.3f})")
        ax.legend(fontsize=8, frameon=False)
    ax.set_xticks(x)
    ax.set_xticklabels(e["Setting"], rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Test PR-AUC", fontsize=9)
    style(ax)
    fig.tight_layout()
    fig.savefig(REV_FIG_DIR / "figR12_perturbation.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_label_validation()
    fig_ablation()
    fig_kde()
    fig_horizon()
    fig_alarm_budget()
    fig_recall_ranking()
    fig_realtime()
    fig_seed_stability()
    fig_perturbation()
    print("figures written to", REV_FIG_DIR)
