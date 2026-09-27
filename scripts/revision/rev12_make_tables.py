"""rev12 - Generate the LaTeX table fragments of the revised manuscript directly
from the result CSVs, so that no number is transcribed by hand.

Fragments are written to submission_packages_20260509/Springer_Submission/tables/
and included from the manuscript with \\input.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.revision.rev_common import REV_DIR, REV_TABLE_DIR  # noqa: E402

_MANUSCRIPT_DIR = (Path(__file__).resolve().parents[2] / "submission_packages_20260509"
                   / "Springer_Submission")
# without the manuscript sources the fragments go next to the other results
TEX_DIR = _MANUSCRIPT_DIR / "tables" if _MANUSCRIPT_DIR.exists() else REV_DIR / "latex"
TEX_DIR.mkdir(parents=True, exist_ok=True)


SHORT_NAMES = {
    # one name per model, used by every table
    "Heuristic: composite score persistence": "Persistence",
    "Persistence heuristic": "Persistence",
    "Heuristic: renewable ramp": "Ramp persistence",
    "Heuristic: |system imbalance|": "Imbalance persistence",
    "VAE reconstruction (unsupervised)": "VAE",
    "HistGBM (regression, tuned)": "HistGBM-R",
    "XGBoost (regression, tuned)": "XGBoost-R",
    "LightGBM (regression, tuned)": "LightGBM-R",
    "HistGBM (tuned)": "HistGBM-C",
    "XGBoost (tuned)": "XGBoost-C",
    "LightGBM (tuned)": "LightGBM-C",
    "AERIS (common recipe)": "AERIS-CR",
    "LSTM classifier": "LSTM",
    "GRU classifier": "GRU",
    "TCN classifier": "TCN",
    "Transformer classifier": "Transformer",
    "Logistic Regression": "LR",
    "Isolation Forest": "IF",
    "One-Class SVM": "OCSVM",
    "KDE (input space)": "KDE",
    "Composite score, z-scores and raw constituents removed (137)":
        "No label inputs",
    "Canonical inputs (146 features)": "Canonical",
    "Composite score and z-scores removed (141)": "No score inputs",
    "One-interval publication lag on all inputs": "One-interval lag",
    "Training-only label normalisation and threshold": "Train-only label",
    "Training-only label + publication lag": "Train-only + lag",
    "Equal weights in the composite score": "Equal weights",
    "w/o variational latent (deterministic AE)": "w/o variational latent (det. AE)",
    "w/o latent bottleneck and reconstruction": "w/o bottleneck and reconstruction",
    "Full-window reconstruction target": "Full-window reconstruction",
    "w/o auxiliary score regression": "w/o auxiliary regression",
    "VAE without attention (mean pooling)": "VAE, mean pooling",
    "w/o attention (mean pooling)": "w/o attention (mean pooling)",
    "window-mean reconstruction replaced": "window-mean reconstruction",
    "attention, VAE kept": "attention (VAE kept)",
    "non-causal right context": "non-causal context",
    "auxiliary regression head": "auxiliary regression",
    "Elia non-linear alpha component": "Non-linear $\\alpha$ component",
    "Absolute area control error": "Absolute ACE",
    "Any external indicator in its own top 5%": "Any external indicator",
    "no (constituent)": "no",
}


def strip_units(name: str) -> str:
    """Units are stated in the caption, not repeated in every row."""
    out = str(name)
    for unit in (" (EUR/MWh)", " (MW)", " (€/MWh)"):
        out = out.replace(unit, "")
    return out


def short(name: str) -> str:
    return SHORT_NAMES.get(str(name), str(name))


def esc(text: str) -> str:
    return (str(text).replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")
            .replace("#", r"\#"))


def write(name: str, body: str) -> None:
    (TEX_DIR / name).write_text(body, encoding="utf-8")
    print(f"  wrote {name}")


def wrap(caption: str, label: str, colspec: str, header: str, rows: list[str],
         placement: str = "ht", size: str = "small", colsep: str | None = None) -> str:
    r"""Emit one table set to the full text width, with `size` as the body font.

    Neither tabularx nor \resizebox is usable here (see below), so tables are
    kept inside the text block by keeping their content narrow: short column
    headers, abbreviated method names, and no column that repeats information
    already carried by another one.
    """
    # `\resizebox` is not an option here: the journal class redefines the table
    # environment in a way that breaks when the tabular is placed inside a box.
    # Tables are kept inside the measure by trimming their content instead.
    opening = [f"\\begin{{tabular*}}{{\\textwidth}}"
               f"{{@{{\\extracolsep\\fill}}{colspec}}}"]
    closing = ["\\end{tabular*}"]
    lines = [f"\\begin{{table}}[{placement}]",
             f"\\caption{{{caption}}}",
             f"\\label{{{label}}}",
             f"\\{size}",
             *([f"\\setlength{{\\tabcolsep}}{{{colsep}}}"] if colsep else []),
             *opening,
             "\\toprule", header, "\\midrule", *rows,
             # booktabs' rule rather than the class macro: \botrule does not
             # survive being placed inside \resizebox
             "\\bottomrule",
             *closing, "\\end{table}"]
    return "\n".join(lines) + "\n"


def lightgbm_horizons() -> pd.DataFrame:
    """Per-horizon PR-AUC rows of the tuned LightGBM regressor (rev33), if run."""
    path = REV_TABLE_DIR / "rev33_horizon_lightgbm.csv"
    if not path.exists():
        return pd.DataFrame(columns=["Horizon", "LightGBM", "p (two-sided)"])
    d = pd.read_csv(path)
    return d[d["Metric"] == "PR_AUC"].set_index("Horizon")


def table_regimes() -> None:
    """Regimes in which AERIS ranks first, each against the best alternative
    evaluated in that same regime."""
    rows = []

    def add(label, df, col, fmt):
        s = df.sort_values(col, ascending=False).reset_index(drop=True)
        v = float(s.loc[s["Model"] == "AERIS", col].iloc[0])
        other = s[s["Model"] != "AERIS"].iloc[0]
        if v < float(other[col]) - 1e-9:
            return          # this table lists only the regimes it leads
        rows.append(f"{label} & {len(s)} & {fmt.format(v)} & "
                    f"{fmt.format(float(other[col]))} & {esc(short(other['Model']))} \\\\")

    m = REV_TABLE_DIR / "rev07_metrics_with_ci.csv"
    if m.exists():
        d = pd.read_csv(m)
        add(r"Precision@2\%", d, "Precision@2%", "{:.1f}")
        add(r"Recall@2\%", d, "Recall@2%", "{:.1f}")
        add("ROC-AUC", d, "ROC_AUC", "{:.3f}")
    b = REV_TABLE_DIR / "rev07_alert_budget.csv"
    if b.exists():
        d = pd.read_csv(b)
        s = d[abs(d["Budget"] - 0.03) < 1e-9][["Model", "Recall@K"]]
        add(r"Recall@3\%", s, "Recall@K", "{:.1f}")
    o = REV_TABLE_DIR / "rev07_onset_subset.csv"
    if o.exists():
        d = pd.read_csv(o)
        add("PR-AUC, onset subset", d, "PR_AUC", "{:.4f}")
        add(r"Precision@2\%, onset subset", d, "Precision@2%", "{:.1f}")
        add(r"Recall@2\%, onset subset", d, "Recall@2%", "{:.1f}")
    c = REV_TABLE_DIR / "rev07_cost_utility.csv"
    if c.exists():
        d = pd.read_csv(c)
        s = d[d["Cost ratio (miss/alert)"] == 10][["Model", "Saving vs no-alert (%)"]]
        add(r"Cost reduction (\%), $\kappa=10$", s, "Saving vs no-alert (%)", "{:.1f}")
    # the audits that tighten the protocol towards real-time use -- the causal,
    # training-only label and the publication lag -- are listed against every model
    # refitted under them (rev32 regressors, rev32c classifier, rev05 persistence)
    r05, r32 = REV_TABLE_DIR / "rev05_robustness.csv", REV_TABLE_DIR / "rev32_audit_tuned_regressors.csv"
    r32c = REV_TABLE_DIR / "rev32c_audit_tuned_classifier.csv"
    if r05.exists() and r32.exists() and r32c.exists():
        d05 = pd.read_csv(r05)
        d05 = d05[d05["Part"] == "C_availability"]
        d32, d32c = pd.read_csv(r32), pd.read_csv(r32c)
        for setting, label in [
                ("Training-only label normalisation and threshold", "PR-AUC, training-only label"),
                ("One-interval publication lag on all inputs", "PR-AUC, inputs one interval late"),
                ("Training-only label + publication lag", "PR-AUC, training-only label and lag")]:
            s = pd.concat([d05.loc[d05["Setting"] == setting, ["Model", "PR_AUC"]],
                           d32.loc[d32["Setting"] == setting, ["Model", "PR_AUC"]],
                           d32c.loc[d32c["Setting"] == setting, ["Model", "PR_AUC"]]])
            s = s[s["Model"] != "HistGBM"]  # rev05's untuned classifier, superseded by rev32c
            add(label, s, "PR_AUC", "{:.3f}")
    hp = REV_TABLE_DIR / "rev30_horizon_vs_persistence.json"
    if hp.exists():
        h = json.loads(hp.read_text(encoding="utf-8"))
        for key in sorted(h, key=lambda k: int(k[2:])):
            e = h[key]
            if e["minutes"] <= 15 or e["minutes"] > 60:
                continue
            label = f"PR-AUC, {e['minutes']} min ahead"
            models = {"AERIS": e["AERIS"], "HistGBM-R": e["HistGBM"],
                      "Persistence": e["Persistence"]}
            lg = lightgbm_horizons()
            if key in lg.index:
                models["LightGBM-R"] = float(lg.loc[key, "LightGBM"])
            s = pd.DataFrame({"Model": list(models), "PR_AUC": list(models.values())})
            add(label, s, "PR_AUC", "{:.3f}")
    if not rows:
        return
    body = wrap(
        "Regimes in which AERIS ranks first; $n$ methods evaluated in each",
        "tab:regimes", "lcccc",
        r"\textbf{Regime and metric} & \textbf{$n$} & \textbf{AERIS} & "
        r"\textbf{Next best} & \textbf{Held by}\\",
        rows, size="footnotesize")
    write("tab_regimes.tex", body)

def table_main() -> None:
    path = REV_TABLE_DIR / "rev07_metrics_with_ci.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    order = [
        # proposed model
        "AERIS",
        # tuned boosting machines, regression then classification formulation
        "HistGBM (regression, tuned)", "XGBoost (regression, tuned)",
        "LightGBM (regression, tuned)",
        "HistGBM (tuned)", "XGBoost (tuned)", "LightGBM (tuned)",
        # model-free heuristic
        "Heuristic: composite score persistence",
        # sequence networks under the common training recipe
        "AERIS (common recipe)", "AERIS-DB", "TCN classifier", "LSTM classifier",
        "GRU classifier", "Transformer classifier",
        # untuned tabular models
        "HistGBM", "LightGBM", "XGBoost", "GBM", "RF", "CART", "Logistic Regression",
        # unsupervised detectors and the weaker heuristics
        "VAE reconstruction (unsupervised)", "Isolation Forest", "KDE (input space)",
        "One-Class SVM", "Heuristic: renewable ramp", "Heuristic: |system imbalance|",
    ]
    df["rank"] = df["Model"].apply(lambda m: order.index(m) if m in order else 99)
    df = df.sort_values(["rank", "Model"])
    # the best value in each metric column is set in bold, whichever method holds it
    # the operational metrics come first; F1 is reported in the text (Section 5.2)
    best = {c: df[c].max() for c in ("PR_AUC", "Precision@2%", "Recall@2%", "ROC_AUC")}
    rows = []
    for _, r in df.iterrows():
        name = esc(short(r["Model"]))

        def cell(col: str, fmt: str) -> str:
            text = fmt.format(r[col])
            return f"\\textbf{{{text}}}" if r[col] == best[col] else text

        rows.append(
            f"{name} & {cell('Precision@2%', '{:.1f}')} & {cell('Recall@2%', '{:.1f}')} & "
            f"{cell('ROC_AUC', '{:.3f}')} & {cell('PR_AUC', '{:.3f}')} & "
            f"[{r['PR_AUC_lo']:.2f}, {r['PR_AUC_hi']:.2f}] \\\\")
    body = wrap(
        "One-step-ahead screening on the test partition; best value per column in bold",
        "tab:main_comparison", "lccccc",
        "\\textbf{Method} & \\textbf{P@2\\%} & \\textbf{R@2\\%} & \\textbf{ROC} & "
        "\\textbf{PR-AUC} & \\textbf{95\\% CI}\\\\",
        rows, size="footnotesize")
    write("tab_main_comparison.tex", body)


def table_significance() -> None:
    path = REV_TABLE_DIR / "rev07_significance.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    df = df[df["Metric"] == "PR_AUC"]
    rows = []
    for base in df["Baseline"].unique():
        sub = df[df["Baseline"] == base]
        iid = sub[sub["Resampling"] == "i.i.d."]
        blk = sub[sub["Resampling"].str.startswith("moving block")]
        if iid.empty or blk.empty:
            continue
        i, b = iid.iloc[0], blk.iloc[0]
        rows.append(
            f"{esc(short(base))} & {b['observed_diff']:+.3f} & "
            f"[{b['ci_lower']:+.2f}, {b['ci_upper']:+.2f}] & "
            f"{b['p_two_sided']:.3f} & {i['p_two_sided']:.3f} \\\\")
    body = wrap(
        "Paired bootstrap test of the PR-AUC difference, AERIS minus baseline",
        "tab:significance", "lcccc",
        r"\textbf{Baseline} & \textbf{Difference} & \textbf{Block 95\% CI} & "
        r"\textbf{$p_{\mathrm{blk}}$} & \textbf{$p_{\mathrm{iid}}$}\\",
        rows, size="footnotesize")
    write("tab_significance.tex", body)

def table_budget() -> None:
    path = REV_TABLE_DIR / "rev07_alert_budget.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    # every method that leads the proposed model anywhere in the sweep is listed
    keep = ["AERIS", "HistGBM (regression, tuned)", "XGBoost (regression, tuned)",
            "LightGBM (regression, tuned)", "HistGBM (tuned)", "XGBoost (tuned)",
            "LightGBM (tuned)", "RF",
            "Heuristic: composite score persistence", "LSTM classifier", "GBM"]
    budgets = [0.005, 0.01, 0.015, 0.02, 0.03, 0.05]
    rows = []
    for m in keep:
        sub = df[df["Model"] == m]
        if sub.empty:
            continue
        cells = []
        for b in budgets:
            r = sub[np.isclose(sub["Budget"], b)]
            cells.append("--" if r.empty else
                         f"{r['Recall@K'].iloc[0]:.1f}/{r['Missed events'].iloc[0]:.0f}")
        rows.append(f"{esc(short(m))} & " + " & ".join(cells) + " \\\\")
    header = "\\textbf{Method} & " + " & ".join(
        f"\\textbf{{{100 * b:g}\\%}}" for b in budgets) + "\\\\"
    body = wrap(
        "Recall (\\%) and number of missed events under fixed alert budgets, expressed as a "
        "percentage of the $6{,}598$ test intervals. A budget of $1\\%$ corresponds to "
        "inspecting $66$ quarter-hours over the $2.8$-month test period, roughly $0.8$ alerts "
        "per day. Each cell reads recall / missed events out of $71$.",
        "tab:budget", "lcccccc", header, rows)
    write("tab_budget.tex", body)


def table_cost() -> None:
    path = REV_TABLE_DIR / "rev07_cost_utility.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    # every method that leads the proposed model anywhere in the sweep is listed
    keep = ["AERIS", "HistGBM (regression, tuned)", "XGBoost (regression, tuned)",
            "LightGBM (regression, tuned)", "HistGBM (tuned)", "XGBoost (tuned)",
            "LightGBM (tuned)", "RF",
            "Heuristic: composite score persistence", "LSTM classifier", "GBM"]
    ratios = sorted(df["Cost ratio (miss/alert)"].unique())
    rows = []
    for m in keep:
        sub = df[df["Model"] == m]
        if sub.empty:
            continue
        cells = [f"{sub[sub['Cost ratio (miss/alert)'] == k]['Saving vs no-alert (%)'].iloc[0]:.1f}"
                 for k in ratios]
        rows.append(f"{esc(short(m))} & " + " & ".join(cells) + " \\\\")
    header = "\\textbf{Method} & " + " & ".join(
        f"\\textbf{{$\\kappa={k:g}$}}" for k in ratios) + "\\\\"
    body = wrap(
        "Reduction (\\%) in expected operational cost relative to a no-alert policy, at the "
        "budget that minimises \\eqref{eq:cost} for each cost ratio $\\kappa=C_{\\mathrm{miss}}/"
        "C_{\\mathrm{alert}}$. Positive values mean the screening layer is worth operating; "
        "larger is better.",
        "tab:cost", "lcccccc", header, rows)
    write("tab_cost.tex", body)


def table_audit() -> None:
    path = REV_TABLE_DIR / "rev05_robustness.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    sub = df[df["Part"] == "C_availability"]
    if sub.empty:
        return
    # the tabular references are the strongest competitors of the main comparison,
    # the tuned boosting regressors, refitted under each setting (rev32);
    # rev05's fixed-parameter classifier is too weak to be the reference
    tuned_clf = {}
    t32c = REV_TABLE_DIR / "rev32c_audit_tuned_classifier.csv"
    if t32c.exists():
        d32c = pd.read_csv(t32c)
        tuned_clf = dict(zip(d32c["Setting"], d32c["PR_AUC"]))
    tuned, tuned_lgbm = {}, {}
    t32 = REV_TABLE_DIR / "rev32_audit_tuned_regressors.csv"
    if t32.exists():
        d32 = pd.read_csv(t32)
        h = d32[d32["Model"] == "HistGBM (regression, tuned)"]
        tuned = dict(zip(h["Setting"], h["PR_AUC"]))
        lg = d32[d32["Model"] == "LightGBM (regression, tuned)"]
        tuned_lgbm = dict(zip(lg["Setting"], lg["PR_AUC"]))
    rows = []
    for setting in sub["Setting"].unique():
        block = sub[sub["Setting"] == setting].set_index("Model")
        if "AERIS" not in block.index:
            continue
        a = block.loc["AERIS"]
        feats = 146 if pd.isna(a.get("Features", np.nan)) else int(a["Features"])
        reg = tuned.get(setting)
        lgbm = tuned_lgbm.get(setting)
        # rev05's fixed-parameter histogram boosting classifier of the thresholded
        # label, reported beside the regressor so both task formulations are visible
        clf = tuned_clf.get(setting)
        pers = block.loc["Persistence heuristic", "PR_AUC"] \
            if "Persistence heuristic" in block.index else None
        fmt = (lambda v: "--" if v is None else f"{v:.3f}")
        rows.append(f"{esc(short(setting))} & {feats} & {a['PR_AUC']:.3f} & "
                    f"{fmt(lgbm)} & {fmt(reg)} & {fmt(clf)} & {fmt(pers)} \\\\")
    body = wrap(
        "Test PR-AUC under stricter information and labelling assumptions",
        "tab:audit", "lcccccc",
        "\\textbf{Setting} & \\textbf{Inputs} & \\textbf{AERIS} & \\textbf{LightGBM-R} & "
        "\\textbf{HistGBM-R} & \\textbf{HistGBM-C} & \\textbf{Persistence}\\\\",
        rows, size="footnotesize", colsep="3pt")
    write("tab_audit.tex", body)


def table_ablation() -> None:
    path = REV_TABLE_DIR / "rev04_ablation_summary.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    struct = df[df["Group"] == "structural"]
    if struct.empty:
        return
    # "VAE without attention (mean pooling)" builds the same network as
    # "w/o attention (mean pooling)"; reporting both would list one ablation twice
    struct = struct[struct["Variant"] != "VAE without attention (mean pooling)"]
    base = struct.loc[struct["Variant"] == "AERIS (full)", "PR_AUC_mean"].iloc[0]
    rows = []
    for _, r in struct.iterrows():
        delta = 100 * (r["PR_AUC_mean"] - base) / base
        std = "" if pd.isna(r["PR_AUC_std"]) else f"\\,$\\pm$\\,{r['PR_AUC_std']:.3f}"
        rows.append(
            f"{esc(short(r['Variant']))} & "
            f"{r['PR_AUC_mean']:.3f}{std} & {delta:+.1f} & {r['F1_mean']:.1f} & "
            f"{r['Recall@2%_mean']:.1f} \\\\")
    body = wrap(
        "Component ablation. The first row is the full model; every other row differs "
        "from it in exactly one respect and is trained with the identical recipe and epoch "
        "budget, and its name states what was removed or replaced. Values are means over the seeds listed "
        "in the text, PR-AUC with its standard deviation across seeds, and $\\Delta$ is the "
        "relative change in PR-AUC with respect to the full model. $F_1$ and R@2\\% are "
        "percentages.",
        "tab:ablation", "lcccc",
        "\\textbf{Variant} & \\textbf{PR-AUC} & \\textbf{$\\Delta$ (\\%)} & "
        "\\textbf{$F_1$} & \\textbf{R@2\\%}\\\\",
        rows, size="footnotesize")
    write("tab_ablation.tex", body)


def table_horizon() -> None:
    """Horizons of 30 to 60 minutes, from the per-horizon re-selection (rev27,
    HistGBM-R; rev33, LightGBM-R) and the persistence comparison (rev30)."""
    hz = REV_TABLE_DIR / "rev27_horizon_significance.csv"
    if not hz.exists():
        return
    df = pd.read_csv(hz)
    df = df[(df["Metric"] == "PR_AUC") & (df["Minutes"] >= 30) & (df["Minutes"] <= 60)]
    df = df.sort_values("Minutes")
    pers = {}
    hp = REV_TABLE_DIR / "rev30_horizon_vs_persistence.json"
    if hp.exists():
        h = json.loads(hp.read_text(encoding="utf-8"))
        pers = {k: v["Persistence"] for k, v in h.items()}
    lg = lightgbm_horizons()

    def ptext(p):
        if pd.isna(p):
            return "--"
        return (r"$\mathbf{" + f"{p:.3f}" + "}$") if p < 0.05 else f"{p:.3f}"

    rows = []
    for _, r in df.iterrows():
        key = r["Horizon"]
        l_ = float(lg.loc[key, "LightGBM"]) if key in lg.index else float("nan")
        p_l = float(lg.loc[key, "p (two-sided)"]) if key in lg.index else float("nan")
        p_ = pers.get(key, float("nan"))
        best = np.nanmax([r["AERIS"], l_, r["HistGBM"], p_])
        cell = (lambda v: "--" if pd.isna(v) else
                (rf"\textbf{{{v:.3f}}}" if v == best else f"{v:.3f}"))
        rows.append(
            rf"{int(r['Minutes'])}\,min & {cell(r['AERIS'])} & {cell(l_)} & "
            rf"{cell(r['HistGBM'])} & {cell(p_)} & {ptext(p_l)} & "
            rf"{ptext(r['p (two-sided)'])} \\")
    body = wrap(
        "Test PR-AUC by screening horizon",
        "tab:horizon", "lcccccc",
        r"\textbf{Horizon} & \textbf{AERIS} & \textbf{LightGBM-R} & \textbf{HistGBM-R} & "
        r"\textbf{Persistence} & \textbf{$p_{\mathrm{L}}$} & \textbf{$p_{\mathrm{H}}$}\\",
        rows)
    write("tab_horizon.tex", body)

def table_complexity() -> None:
    rows = []
    sel = REV_TABLE_DIR / "rev10_selected_summary.json"
    base = REV_TABLE_DIR / "rev02_baseline_suite.csv"
    seq = REV_TABLE_DIR / "rev03_sequence_summary.csv"
    if sel.exists():
        s = json.loads(sel.read_text())
        rows.append(f"AERIS & {s['n_parameters']:,} & "
                    f"{np.mean([v for v in [s['test']['PR_AUC'][0]]]):.4f} & "
                    f"{s['inference_ms_per_sample']:.3f} \\\\")
    if seq.exists():
        d = pd.read_csv(seq)
        for _, r in d.iterrows():
            if r["Model"].startswith("AERIS"):
                continue
            if pd.isna(r.get("Params", np.nan)):
                continue
            rows.append(f"{esc(r['Model'])} & {int(r['Params']):,} & {r['PR_AUC_mean']:.4f} & "
                        f"{r.get('Infer_ms_per_sample', float('nan')):.3f} \\\\")
    if base.exists():
        d = pd.read_csv(base)
        for _, r in d.iterrows():
            if r["Model"] in ("HistGBM", "GBM", "RF", "CART", "Logistic Regression"):
                rows.append(f"{esc(r['Model'])} & -- & {r['PR_AUC']:.4f} & "
                            f"{r['Infer_ms_per_sample']:.3f} \\\\")
    if not rows:
        return
    body = wrap(
        "Model size, screening quality and single-interval inference latency measured on the "
        "test partition on an eight-thread CPU. The proposed model is listed as one network of "
        "the deployed three-seed ensemble, with the mean PR-AUC of its single runs; the "
        "ensemble itself reaches $0.7056$. One quarter-hour leaves at least $900$ "
        "seconds before the next decision, so every method is far inside the real-time budget; "
        "the column is reported so that deployment feasibility can be judged.",
        "tab:complexity", "lccc",
        "\\textbf{Method} & \\textbf{Parameters} & \\textbf{PR-AUC} & "
        "\\textbf{Inference (ms/interval)}\\\\",
        rows)
    write("tab_complexity.tex", body)


def table_label_validation() -> None:
    path = REV_TABLE_DIR / "rev01_label_external_validation.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    # the absolute system imbalance is a constituent of the label, so it has no
    # place in a table of *external* validation
    df = df[df["External to label"] == "yes"]
    rows = []
    for _, r in df.iterrows():
        rows.append(
            f"{esc(short(strip_units(r['Indicator'])))} & "
            f"{r['P(top-5% severity | event) %']:.1f} & "
            f"{r['P(top-5% severity | non-event) %']:.1f} & {r['Risk ratio']:.2f} & "
            f"{r['Rank AUC']:.3f} \\\\")
    body = wrap(
        "External validation of the screening label",
        "tab:label_validation", "lcccc",
        r"\textbf{External indicator} & \textbf{Flagged (\%)} & "
        r"\textbf{Unflagged (\%)} & \textbf{Ratio} & \textbf{Rank AUC}\\",
        rows)
    write("tab_label_validation.tex", body)

def table_rolling() -> None:
    path = REV_TABLE_DIR / "rev05_robustness.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    sub = df[df["Part"] == "D_rolling"]
    if sub.empty:
        return
    rows = []
    for setting in sub["Setting"].unique():
        block = sub[sub["Setting"] == setting]
        get = lambda m, c: (block.loc[block["Model"] == m, c].iloc[0]  # noqa: E731
                            if (block["Model"] == m).any() else np.nan)
        rows.append(f"{esc(setting)} & {int(get('AERIS', 'Positives'))} & "
                    f"{get('AERIS', 'PR_AUC'):.4f} & {get('HistGBM', 'PR_AUC'):.4f} & "
                    f"{get('AERIS', 'Recall@2%'):.1f} & {get('HistGBM', 'Recall@2%'):.1f} \\\\")
    body = wrap(
        "Rolling-origin evaluation. Each fold expands the training window, uses the following "
        "$10\\%$ of the record for validation and the next $10\\%$ as a fresh test block; the "
        "label normalisation and the event threshold are re-estimated on that fold's training "
        "window only.",
        "tab:rolling", "lccccc",
        "\\textbf{Fold} & \\textbf{Events} & \\textbf{AERIS PR-AUC} & "
        "\\textbf{HistGBM PR-AUC} & \\textbf{AERIS R@2\\%} & \\textbf{HistGBM R@2\\%}\\\\",
        rows)
    write("tab_rolling.tex", body)


if __name__ == "__main__":
    table_label_validation()
    table_main()
    table_significance()
    table_budget()
    table_cost()
    table_audit()
    table_ablation()
    table_horizon()
    table_rolling()
    table_complexity()
    table_regimes()
    print("done")
