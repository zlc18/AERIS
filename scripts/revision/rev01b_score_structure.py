"""rev01b - Internal structure of the composite event score.

Reviewer 1 warns that the constituents of the composite score also appear among
the model inputs.  This script quantifies how much of the score each component
actually drives, how strongly the constituents are correlated with one another,
and how predictable the label is from simple persistence - the last point is
what makes the model-free persistence heuristic in rev02 the reference any
learned model has to beat.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.revision.rev_common import (EVENT_COMPONENTS, REV_TABLE_DIR,  # noqa: E402
                                         build_event_score, load_feature_table, split_bounds)


def main() -> None:
    df = load_feature_table()
    train_end, _ = split_bounds(len(df))
    cols = list(EVENT_COMPONENTS)
    z = pd.DataFrame({c: (df[c] - df[c].iloc[:train_end].mean()) / df[c].iloc[:train_end].std()
                      for c in cols})
    score = build_event_score(df, "train", train_end)

    z.corr().round(4).to_csv(REV_TABLE_DIR / "rev01b_constituent_correlation_pearson.csv",
                             encoding="utf-8-sig")
    z.corr(method="spearman").round(4).to_csv(
        REV_TABLE_DIR / "rev01b_constituent_correlation_spearman.csv", encoding="utf-8-sig")

    rows = []
    for c, w in EVENT_COMPONENTS.items():
        contribution = w * z[c]
        rows.append({
            "Component": c,
            "Weight": w,
            "Correlation with composite score": float(np.corrcoef(contribution, score)[0, 1]),
            "Share of score variance (%)": 100 * float(np.cov(contribution, score)[0, 1] / np.var(score)),
        })
    contrib = pd.DataFrame(rows)
    contrib.to_csv(REV_TABLE_DIR / "rev01b_component_contribution.csv",
                   index=False, encoding="utf-8-sig")

    threshold = float(score.iloc[:train_end].quantile(0.98))
    label = (score.shift(-1) >= threshold).astype(int)
    persistence = {
        "lag1_autocorrelation_of_score": float(pd.Series(score).autocorr(1)),
        "lag4_autocorrelation_of_score": float(pd.Series(score).autocorr(4)),
        "P(event at t+1 | event at t)": float(label[label.shift(1) == 1].mean()),
        "P(event at t+1 | no event at t)": float(label[label.shift(1) == 0].mean()),
        "base_rate": float(label.mean()),
    }
    (REV_TABLE_DIR / "rev01b_persistence.json").write_text(
        json.dumps(persistence, indent=2), encoding="utf-8")

    print(z.corr().round(3).to_string())
    print()
    print(contrib.round(4).to_string(index=False))
    print()
    print(json.dumps(persistence, indent=2))


if __name__ == "__main__":
    main()
