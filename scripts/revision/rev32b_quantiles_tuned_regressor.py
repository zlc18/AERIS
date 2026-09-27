"""rev32b - Event-quantile sensitivity against the tuned boosting regressor.

rev05 part B compared AERIS with a fixed-parameter histogram boosting classifier
at q in {0.95, 0.975, 0.98, 0.99}.  This repeats the tabular side with the tuned
boosting regressors of the main comparison, deployed as there (rev32.fit_predict);
AERIS results are taken from rev05.

Usage: python scripts/revision/rev32b_quantiles_tuned_regressor.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.revision.rev_common import (REV_TABLE_DIR, build_matrices,  # noqa: E402
                                         summarise_sequence)
from scripts.revision.rev_eval import metric_value  # noqa: E402
from scripts.revision.rev32_audit_tuned_regressors import (factories,  # noqa: E402
                                                           fit_predict)

QUANTILES = [0.95, 0.975, 0.98, 0.99]


def main() -> None:
    rob = pd.read_csv(REV_TABLE_DIR / "rev05_robustness.csv")
    b = rob[rob["Part"] == "B_threshold"]
    rows = []
    for q in QUANTILES:
        tr, va, te, y_tr, y_va, y_te, meta = build_matrices(
            quantile=q, horizon=1, seq_len=8, normalisation="full", threshold_source="full")
        trf, vaf, tef = summarise_sequence(tr), summarise_sequence(va), summarise_sequence(te)
        sub = b[b["Setting"] == f"q={q}"].set_index("Model")["PR_AUC"]
        row = {"Quantile": q, "Events": int(y_te.sum())}
        for name in factories():
            ts = fit_predict(name, trf, meta["aux_train"], vaf, tef)
            row[name] = metric_value(y_te, ts, "PR_AUC")
        row.update({"AERIS": sub.get("AERIS"), "Persistence": sub.get("Persistence heuristic")})
        rows.append(row)
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(REV_TABLE_DIR / "rev32b_quantiles_tuned_regressor.csv",
              index=False, encoding="utf-8-sig")
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
