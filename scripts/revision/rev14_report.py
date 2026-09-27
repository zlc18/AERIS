"""rev14 - Print the numbers the Results section quotes, in one place.

Running this after the experiment pipeline gives a compact, authoritative
summary of every figure quoted in the manuscript text, so the narrative can be
checked against the data in one pass.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.revision.rev_common import REV_TABLE_DIR  # noqa: E402

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def show(path: Path, cols=None, sort=None, head=None) -> None:
    if not path.exists():
        print(f"  [missing] {path.name}")
        return
    df = pd.read_csv(path)
    if sort:
        df = df.sort_values(sort, ascending=False)
    if cols:
        cols = [c for c in cols if c in df.columns]
        df = df[cols]
    if head:
        df = df.head(head)
    print(df.to_string(index=False))


def show_json(path: Path) -> None:
    if not path.exists():
        print(f"  [missing] {path.name}")
        return
    print(json.dumps(json.loads(path.read_text(encoding="utf-8")), indent=2)[:2500])


def main() -> None:
    section("1  Main comparison with block-bootstrap intervals")
    show(REV_TABLE_DIR / "rev07_metrics_with_ci.csv",
         ["Model", "PR_AUC", "PR_AUC_lo", "PR_AUC_hi", "F1", "Precision@1%", "Recall@1%",
          "Precision@2%", "Recall@2%", "ROC_AUC", "Brier"], sort="PR_AUC")

    section("2  Selected AERIS configuration and seed spread")
    show_json(REV_TABLE_DIR / "rev10_selected_summary.json")
    show(REV_TABLE_DIR / "rev10_selected_per_seed.csv")

    section("3  Tuned tabular baselines")
    show(REV_TABLE_DIR / "rev09_tuned_baselines.csv",
         ["Model", "Valid PR_AUC", "PR_AUC", "F1", "Precision@2%", "Recall@2%", "Best params"])

    section("4  Sequence architectures under the common recipe")
    show(REV_TABLE_DIR / "rev03_sequence_summary.csv",
         ["Model", "Seeds", "PR_AUC_mean", "PR_AUC_std", "F1_mean", "Recall@2%_mean",
          "Params", "Train_s_mean", "Infer_ms_per_sample"], sort="PR_AUC_mean")

    section("5  Paired significance (AERIS minus baseline)")
    path = REV_TABLE_DIR / "rev07_significance.csv"
    if path.exists():
        df = pd.read_csv(path)
        for metric in ["PR_AUC", "Recall@2%"]:
            print(f"\n-- {metric} --")
            sub = df[df["Metric"] == metric][
                ["Baseline", "Resampling", "observed_diff", "ci_lower", "ci_upper",
                 "p_one_sided", "p_two_sided"]]
            print(sub.to_string(index=False))

    section("6  Alert budgets")
    path = REV_TABLE_DIR / "rev07_alert_budget.csv"
    if path.exists():
        df = pd.read_csv(path)
        keep = df[np.isclose(df["Budget"].values[:, None],
                             [0.005, 0.01, 0.015, 0.02, 0.05]).any(axis=1)]
        pivot = keep.pivot_table(index="Model", columns="Budget",
                                 values=["Recall@K", "Precision@K", "Missed events"])
        print(pivot.round(2).to_string())

    section("7  Cost-sensitive utility (% saving vs no-alert)")
    path = REV_TABLE_DIR / "rev07_cost_utility.csv"
    if path.exists():
        df = pd.read_csv(path)
        print(df.pivot_table(index="Model", columns="Cost ratio (miss/alert)",
                             values="Saving vs no-alert (%)").round(1).to_string())
        print("\nbest budget (%) by cost ratio")
        print(df.pivot_table(index="Model", columns="Cost ratio (miss/alert)",
                             values="Best budget (%)").round(2).to_string())

    section("8  Onset subset (current state not yet extreme)")
    show_json(REV_TABLE_DIR / "rev07_onset_meta.json")
    show(REV_TABLE_DIR / "rev07_onset_subset.csv", sort="PR_AUC")

    section("9  Availability audit, horizons, thresholds, rolling origin, perturbation")
    path = REV_TABLE_DIR / "rev05_robustness.csv"
    if path.exists():
        df = pd.read_csv(path)
        for part in sorted(df["Part"].unique()):
            print(f"\n-- {part} --")
            cols = [c for c in ["Setting", "Model", "Features", "Positives", "PositiveRate",
                                "PR_AUC", "PR_AUC_std", "F1", "Precision@2%", "Recall@2%"]
                    if c in df.columns]
            print(df[df["Part"] == part][cols].to_string(index=False))

    section("10  Component ablation")
    show(REV_TABLE_DIR / "rev04_ablation_summary.csv",
         ["Variant", "Group", "Removed component", "PR_AUC_mean", "PR_AUC_std", "F1_mean",
          "Recall@2%_mean", "Params"])

    section("11  Latent density confidence layer")
    show(REV_TABLE_DIR / "rev06_density_separation.csv")
    show(REV_TABLE_DIR / "rev06_alert_severity_split.csv")
    show_json(REV_TABLE_DIR / "rev06_summary.json")
    path = REV_TABLE_DIR / "rev06_alpha_sweep.csv"
    if path.exists():
        df = pd.read_csv(path)
        print(df[df["alpha"].isin([0.0, 0.2, 0.5, 1.0, 1.5, 2.0])].to_string(index=False))

    section("12  Label validation and partitions")
    show(REV_TABLE_DIR / "rev01_partition_summary.csv")
    show(REV_TABLE_DIR / "rev01_label_prevalence.csv")
    show(REV_TABLE_DIR / "rev01_label_external_validation.csv",
         ["Indicator", "External to label", "Risk ratio", "Rank AUC",
          "Mann-Whitney p (two-sided)"])
    show_json(REV_TABLE_DIR / "rev01b_persistence.json")


if __name__ == "__main__":
    main()
