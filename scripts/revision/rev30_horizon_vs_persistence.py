"""rev30 - The proposed model against the model-free persistence rule at every horizon.

At 60 minutes ahead the persistence rule, not the boosting regressor, is the
strongest alternative.  Test the proposed model against it with the same paired
moving-block bootstrap used everywhere else."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.revision.rev_common import REV_DIR, REV_TABLE_DIR, build_matrices  # noqa: E402
from scripts.revision.rev_eval import (make_index_sets, metric_value,  # noqa: E402
                                       paired_bootstrap_diff)

SCORES = REV_DIR / "scores"
OUT = REV_TABLE_DIR / "rev30_horizon_vs_persistence.json"

results = {}
for horizon in (1, 2, 3, 4, 5, 6, 8):
    y = np.load(SCORES / f"horizon{horizon}_y.npy").astype(int)
    aeris = np.load(SCORES / f"horizon{horizon}_AERIS.npy")
    hist = np.load(SCORES / f"horizon{horizon}_HistGBM.npy")
    built = build_matrices(quantile=0.98, horizon=horizon, seq_len=8,
                           normalisation="full", threshold_source="full")
    meta = built[6]
    persistence = meta["test_frame"]["event_proxy_score"].to_numpy(dtype=float)
    assert len(persistence) == len(y), (len(persistence), len(y))

    idx = make_index_sets(len(y), block=96, rounds=2000, seed=7)
    entry = {
        "minutes": 15 * horizon,
        "AERIS": metric_value(y, aeris, "PR_AUC"),
        "HistGBM": metric_value(y, hist, "PR_AUC"),
        "Persistence": metric_value(y, persistence, "PR_AUC"),
    }
    best_alt = max(("HistGBM", entry["HistGBM"]), ("Persistence", entry["Persistence"]),
                   key=lambda t: t[1])
    entry["strongest_alternative"] = best_alt[0]
    d = paired_bootstrap_diff(y, aeris, persistence, "PR_AUC", None, None,
                              block=96, index_sets=idx)
    entry["vs_persistence"] = {k: float(v) for k, v in d.items()}
    entry["relative_vs_persistence_pct"] = 100 * (entry["AERIS"] - entry["Persistence"]) / entry["Persistence"]
    entry["relative_vs_best_alt_pct"] = 100 * (entry["AERIS"] - best_alt[1]) / best_alt[1]
    results[f"t+{horizon}"] = entry
    print(f"t+{horizon} ({15*horizon:3d} min)  AERIS {entry['AERIS']:.4f}  "
          f"HistGBM {entry['HistGBM']:.4f}  Persistence {entry['Persistence']:.4f}  "
          f"| strongest alternative: {best_alt[0]} ({best_alt[1]:.4f}), "
          f"AERIS {entry['relative_vs_best_alt_pct']:+.1f}%  "
          f"| vs persistence: diff {d['observed_diff']:+.4f} "
          f"[{d['ci_lower']:+.3f}, {d['ci_upper']:+.3f}] p={d['p_two_sided']:.3f}")

OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
print("\nwrote", OUT)
