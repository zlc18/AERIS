"""rev26 - Calibrate the deployed scores on the validation partition.

The deployed AERIS score is an average of percentile ranks, which is a valid
ranking but is not a probability: read as one it is badly miscalibrated by
construction, and the Brier score and calibration error computed on it are
meaningless. Reviewer 1 asks for calibration to be part of the evaluation, so
every method that is to be read probabilistically is mapped through an isotonic
regression fitted on the validation partition only.

Calibration is monotone, so it cannot change any ranking metric; it only makes
the probability interpretation and the Brier / calibration statistics
comparable across methods.

Usage: python scripts/revision/rev26_calibrate.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.revision.rev_common import REV_TABLE_DIR, load_all_scores, slug  # noqa: E402
from scripts.revision.rev_eval import metric_value  # noqa: E402

SCORE_DIR = Path(__file__).resolve().parents[2] / "outputs" / "revision" / "scores"


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return float("nan")
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, len(edges) - 2)
    ece = 0.0
    for b in range(len(edges) - 1):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(ece)


def main() -> None:
    valid, test = load_all_scores()
    y_va = np.load(SCORE_DIR / "y_valid.npy").astype(int)
    y_te = np.load(SCORE_DIR / "y_test.npy").astype(int)

    rows = []
    for name in sorted(test):
        if name not in valid:
            continue
        v, t = valid[name], test[name]
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(v, y_va)
        t_cal = iso.predict(t)
        np.save(SCORE_DIR / f"calibrated__{slug(name)}.npy", t_cal)

        raw_ok = t.min() >= 0 and t.max() <= 1
        rows.append({
            "Model": name,
            "PR_AUC": metric_value(y_te, t, "PR_AUC"),
            "Brier (raw)": float(np.mean((t - y_te) ** 2)) if raw_ok else np.nan,
            "ECE (raw)": expected_calibration_error(y_te, t) if raw_ok else np.nan,
            "Brier (calibrated)": float(np.mean((t_cal - y_te) ** 2)),
            "ECE (calibrated)": expected_calibration_error(y_te, t_cal),
            "PR_AUC after calibration": metric_value(y_te, t_cal, "PR_AUC"),
        })

    frame = pd.DataFrame(rows).sort_values("Brier (calibrated)")
    frame.to_csv(REV_TABLE_DIR / "rev26_calibration.csv", index=False, encoding="utf-8-sig")
    print(frame.to_string(index=False))

    if "AERIS" in set(frame["Model"]):
        a = frame[frame["Model"] == "AERIS"].iloc[0]
        others = frame[frame["Model"] != "AERIS"]
        for col in ("Brier (calibrated)", "ECE (calibrated)"):
            best = others[col].min()
            who = others.loc[others[col].idxmin(), "Model"]
            print(f"\n  {col}: AERIS {a[col]:.5f} vs best other {best:.5f} ({who}) "
                  f"-> {'lead' if a[col] < best else 'behind'} "
                  f"{100 * (best - a[col]) / best:+.1f}%")


if __name__ == "__main__":
    main()
