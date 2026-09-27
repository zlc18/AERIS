"""rev01 - Construction, causality and external validation of the screening label.

Answers Reviewer 1 comment 1 and Reviewer 2 comment 1:
  (a) writes out the explicit composite-score formula, weights and normalisation;
  (b) compares full-record normalisation (original submission) with
      training-partition-only normalisation and threshold (causal variant);
  (c) reports label prevalence for q in {0.95, 0.975, 0.98, 0.99};
  (d) validates the proxy label against *external* operational outcomes that are
      not constituents of the score: imbalance price, price spread, marginal
      incremental/decremental activation prices, area control error and Elia's
      non-linear alpha component.

Outputs: outputs/revision/tables/rev01_*.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.revision.rev_common import (  # noqa: E402
    EVENT_COMPONENTS, REV_TABLE_DIR, build_labels, load_feature_table, split_bounds,
)

# Operational severity indicators. 'external' = not a constituent of the score.
SEVERITY_INDICATORS = {
    "imbalanceprice_abs": ("Absolute imbalance price (EUR/MWh)", True),
    "price_spread": ("Marginal price spread (EUR/MWh)", True),
    "marginalincrementalprice": ("Marginal incremental price (EUR/MWh)", True),
    "marginaldecrementalprice": ("Marginal decremental price (EUR/MWh)", True),
    "alpha": ("Elia non-linear alpha component (EUR/MWh)", True),
    "ace_abs": ("Absolute area control error (MW)", True),
    "systemimbalance_abs": ("Absolute system imbalance (MW)", False),
}


def rank_auc(values: np.ndarray, labels: np.ndarray) -> float:
    """P(indicator on event > indicator on non-event) + 0.5 P(tie)."""
    pos, neg = values[labels == 1], values[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    u = stats.mannwhitneyu(pos, neg, alternative="two-sided").statistic
    return float(u / (len(pos) * len(neg)))


def main() -> None:
    df = load_feature_table()
    df["ace_abs"] = df["ace"].abs()
    n = len(df)
    train_end, valid_end = split_bounds(n)

    meta: dict[str, object] = {
        "n_rows": int(n),
        "start": str(df["datetime"].min()),
        "end": str(df["datetime"].max()),
        "resolution_minutes": 15,
        "weights": EVENT_COMPONENTS,
        "train_rows": int(train_end),
        "valid_rows": int(valid_end - train_end),
        "test_rows": int(n - valid_end),
        "train_period": [str(df["datetime"].iloc[0]), str(df["datetime"].iloc[train_end - 1])],
        "valid_period": [str(df["datetime"].iloc[train_end]), str(df["datetime"].iloc[valid_end - 1])],
        "test_period": [str(df["datetime"].iloc[valid_end]), str(df["datetime"].iloc[-1])],
    }

    # ---- (b) full-record vs training-only normalisation -------------------
    rows = []
    label_cache: dict[tuple[str, float], pd.DataFrame] = {}
    for tag, norm, thr_src in [
        ("Full-record (original submission)", "full", "full"),
        ("Training partition only (causal)", "train", "train"),
    ]:
        for q in (0.95, 0.975, 0.98, 0.99):
            frame, thr = build_labels(df, quantile=q, horizon=1,
                                      normalisation=norm, threshold_source=thr_src)
            label_cache[(norm, q)] = frame
            te = frame.iloc[valid_end:]
            rows.append({
                "Normalisation": tag,
                "Quantile": q,
                "Threshold": thr,
                "Positive rate overall (%)": 100 * frame["future_event_label"].mean(),
                "Positive rate train (%)": 100 * frame["future_event_label"].iloc[:train_end].mean(),
                "Positive rate valid (%)": 100 * frame["future_event_label"].iloc[train_end:valid_end].mean(),
                "Positive rate test (%)": 100 * te["future_event_label"].mean(),
                "Test positives": int(te["future_event_label"].sum()),
            })
    prevalence = pd.DataFrame(rows)
    prevalence.to_csv(REV_TABLE_DIR / "rev01_label_prevalence.csv", index=False, encoding="utf-8-sig")

    # Agreement between the two labelling variants at q = 0.98.
    a = label_cache[("full", 0.98)]["future_event_label"].to_numpy()
    b = label_cache[("train", 0.98)]["future_event_label"].to_numpy()
    agree = {
        "overall_agreement_pct": float(100 * (a == b).mean()),
        "jaccard_pct": float(100 * np.logical_and(a, b).sum() / max(1, np.logical_or(a, b).sum())),
        "test_agreement_pct": float(100 * (a[valid_end:] == b[valid_end:]).mean()),
        "test_jaccard_pct": float(100 * np.logical_and(a[valid_end:], b[valid_end:]).sum()
                                  / max(1, np.logical_or(a[valid_end:], b[valid_end:]).sum())),
    }
    meta["label_variant_agreement"] = agree

    # ---- (d) external validation of the proxy label ----------------------
    frame = label_cache[("train", 0.98)]
    lab = frame["future_event_label"].to_numpy()
    # severity is evaluated at t+1, i.e. in the interval the label refers to
    shifted = df.iloc[1:1 + len(frame)].reset_index(drop=True)

    val_rows = []
    for col, (pretty, external) in SEVERITY_INDICATORS.items():
        v = shifted[col].to_numpy(dtype=float)
        pos, neg = v[lab == 1], v[lab == 0]
        u = stats.mannwhitneyu(pos, neg, alternative="two-sided")
        auc = rank_auc(v, lab)
        q95 = float(np.quantile(v, 0.95))
        p_stress_event = float((pos >= q95).mean())
        p_stress_base = float((neg >= q95).mean())
        val_rows.append({
            "Indicator": pretty,
            "Column": col,
            "External to label": "yes" if external else "no (constituent)",
            "Median | event": float(np.median(pos)),
            "Median | non-event": float(np.median(neg)),
            "Mean | event": float(np.mean(pos)),
            "Mean | non-event": float(np.mean(neg)),
            "Median ratio": float(np.median(pos) / max(np.median(neg), 1e-9)),
            "Rank AUC": auc,
            "Mann-Whitney p (two-sided)": float(u.pvalue),
            "P(top-5% severity | event) %": 100 * p_stress_event,
            "P(top-5% severity | non-event) %": 100 * p_stress_base,
            "Risk ratio": float(p_stress_event / max(p_stress_base, 1e-9)),
        })

    # combined "any severe external outcome" indicator
    ext_cols = [c for c, (_, e) in SEVERITY_INDICATORS.items() if e]
    sev = np.zeros(len(shifted), dtype=bool)
    for c in ext_cols:
        v = shifted[c].to_numpy(dtype=float)
        sev |= v >= np.quantile(v, 0.95)
    p_any_event = float(sev[lab == 1].mean())
    p_any_base = float(sev[lab == 0].mean())
    val_rows.append({
        "Indicator": "Any external indicator in its own top 5%",
        "Column": "composite_external_stress",
        "External to label": "yes",
        "Median | event": np.nan, "Median | non-event": np.nan,
        "Mean | event": p_any_event, "Mean | non-event": p_any_base,
        "Median ratio": np.nan, "Rank AUC": rank_auc(sev.astype(float), lab),
        "Mann-Whitney p (two-sided)": float(
            stats.mannwhitneyu(sev[lab == 1].astype(float), sev[lab == 0].astype(float),
                               alternative="two-sided").pvalue),
        "P(top-5% severity | event) %": 100 * p_any_event,
        "P(top-5% severity | non-event) %": 100 * p_any_base,
        "Risk ratio": float(p_any_event / max(p_any_base, 1e-9)),
    })
    validation = pd.DataFrame(val_rows)
    validation.to_csv(REV_TABLE_DIR / "rev01_label_external_validation.csv",
                      index=False, encoding="utf-8-sig")

    # ---- weight sensitivity: the same external check with equal weights ---
    from scripts.revision.rev_common import EQUAL_WEIGHTS
    equal_frame, _ = build_labels(df, quantile=0.98, horizon=1,
                                  normalisation="train", threshold_source="train",
                                  weights=EQUAL_WEIGHTS)
    lab_eq = equal_frame["future_event_label"].to_numpy()
    weight_rows = []
    for col, (pretty, external) in SEVERITY_INDICATORS.items():
        v = shifted[col].to_numpy(dtype=float)
        q95 = float(np.quantile(v, 0.95))
        row = {"Indicator": pretty, "External to label": "yes" if external else "no"}
        for tag, labels in (("original weights", lab), ("equal weights", lab_eq)):
            pos, neg = v[labels == 1], v[labels == 0]
            row[f"Risk ratio ({tag})"] = float(
                (pos >= q95).mean() / max((neg >= q95).mean(), 1e-9))
            row[f"Rank AUC ({tag})"] = rank_auc(v, labels)
        weight_rows.append(row)
    weight_df = pd.DataFrame(weight_rows)
    weight_df.to_csv(REV_TABLE_DIR / "rev01_weight_sensitivity.csv",
                     index=False, encoding="utf-8-sig")
    meta["equal_weight_label_agreement_pct"] = float(100 * (lab == lab_eq).mean())
    meta["equal_weight_jaccard_pct"] = float(
        100 * np.logical_and(lab, lab_eq).sum() / max(1, np.logical_or(lab, lab_eq).sum()))

    # ---- per-partition counts, on the frame the models actually see -------
    # The modelling frame drops rows with missing inputs (the week-ahead forecast
    # is missing on 21 scattered days) and one row for the regression target,
    # so the partition boundaries differ from those of the raw record. Reporting
    # the raw record here would contradict every experiment.
    seq_len = 8
    modelling = frame.copy()
    modelling["regulation_deviation_target"] = modelling["systemimbalance_abs"].shift(-1)
    modelling = modelling.dropna().reset_index(drop=True)
    m_train_end, m_valid_end = split_bounds(len(modelling))
    meta["modelling_frame_rows"] = int(len(modelling))
    meta["raw_record_rows"] = int(len(df))

    # both labelling conventions are reported, because the experiments use the
    # whole-record convention of the first version and the causal convention is
    # the robustness variant
    canonical = label_cache[("full", 0.98)].copy()
    canonical["regulation_deviation_target"] = canonical["systemimbalance_abs"].shift(-1)
    canonical = canonical.dropna().reset_index(drop=True)

    part_rows = []
    for name, lo, hi in [("Training", 0, m_train_end),
                         ("Validation", m_train_end, m_valid_end),
                         ("Test", m_valid_end, len(modelling))]:
        sub = modelling.iloc[lo:hi]
        labels = sub["future_event_label"].to_numpy()[seq_len - 1:]
        canon = canonical.iloc[lo:hi]["future_event_label"].to_numpy()[seq_len - 1:]
        part_rows.append({
            "Partition": name,
            "Start": str(sub["datetime"].iloc[0]),
            "End": str(sub["datetime"].iloc[-1]),
            "Intervals": len(sub),
            "Windows (seq=8)": len(labels),
            "Events, whole-record labelling": int(canon.sum()),
            "Event rate, whole-record (%)": 100 * float(canon.mean()),
            "Events, training-only labelling": int(labels.sum()),
            "Event rate, training-only (%)": 100 * float(labels.mean()),
        })
    partitions = pd.DataFrame(part_rows)
    partitions.to_csv(REV_TABLE_DIR / "rev01_partition_summary.csv", index=False, encoding="utf-8-sig")

    (REV_TABLE_DIR / "rev01_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    pd.set_option("display.width", 200)
    print("=== partitions ==="); print(partitions.to_string(index=False))
    print("\n=== prevalence ==="); print(prevalence.to_string(index=False))
    print("\n=== label variant agreement ==="); print(json.dumps(agree, indent=2))
    print("\n=== external validation ===")
    print(validation.drop(columns=["Column"]).to_string(index=False))
    print("\n=== weight sensitivity ===")
    print(weight_df.to_string(index=False))
    print(f"label agreement with equal weights: {meta['equal_weight_label_agreement_pct']:.2f}% "
          f"(Jaccard {meta['equal_weight_jaccard_pct']:.1f}%)")


if __name__ == "__main__":
    main()
