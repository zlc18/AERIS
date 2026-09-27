"""rev31 - Sensitivity of the paired bootstrap to the moving-block length.

Section 5.4 states that the conclusions do not change for block lengths in
{1, 32, 96, 192} intervals.  This script produces the evidence for that
statement: for every baseline it recomputes the paired bootstrap of the proposed
model's PR-AUC difference at each block length and reports which baselines are
resolved at the 5% level under each.

Usage: python scripts/revision/rev31_block_sensitivity.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.revision.rev_common import REV_TABLE_DIR, load_all_scores  # noqa: E402
from scripts.revision.rev_eval import make_index_sets, paired_bootstrap_diff  # noqa: E402

SCORE_DIR = Path(__file__).resolve().parents[2] / "outputs" / "revision" / "scores"
PROPOSED = "AERIS"
BLOCKS = [1, 32, 96, 192]
ROUNDS = 2000


def main() -> None:
    y = np.load(SCORE_DIR / "y_test.npy").astype(int)
    _, test = load_all_scores()
    baselines = [m for m in test if m != PROPOSED]
    print(f"{len(baselines)} baselines, block lengths {BLOCKS}, {ROUNDS} resamples each")

    rows = []
    for block in BLOCKS:
        idx = make_index_sets(len(y), block=block, rounds=ROUNDS, seed=7)
        for base in baselines:
            d = paired_bootstrap_diff(y, test[PROPOSED], test[base], "PR_AUC",
                                      None, None, block=block, index_sets=idx)
            rows.append({"Block length": block, "Baseline": base,
                         "Difference": d["observed_diff"],
                         "CI lower": d["ci_lower"], "CI upper": d["ci_upper"],
                         "p (two-sided)": d["p_two_sided"]})
        n_sig = sum(1 for r in rows if r["Block length"] == block
                    and r["p (two-sided)"] < 0.05 and r["Difference"] > 0)
        print(f"  L={block:4d}: {n_sig} baselines beaten at p<0.05", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(REV_TABLE_DIR / "rev31_block_sensitivity.csv",
              index=False, encoding="utf-8-sig")

    wins = {b: sorted(df[(df["Block length"] == b) & (df["p (two-sided)"] < 0.05)
                         & (df["Difference"] > 0)]["Baseline"]) for b in BLOCKS}
    reference = wins[96]
    summary = {
        "block_lengths": BLOCKS,
        "rounds": ROUNDS,
        "significant_wins": {str(b): wins[b] for b in BLOCKS},
        "counts": {str(b): len(wins[b]) for b in BLOCKS},
        # the claim in the text: the set of resolved comparisons is stable over
        # the block lengths the paper actually relies on (32, 96, 192)
        "stable_over_32_96_192": all(wins[b] == reference for b in (32, 96, 192)),
        "iid_extra_wins": sorted(set(wins[1]) - set(reference)),
        "iid_missing_wins": sorted(set(reference) - set(wins[1])),
    }
    (REV_TABLE_DIR / "rev31_block_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print("\ncounts by block length:", summary["counts"])
    print("stable over L in {32, 96, 192}:", summary["stable_over_32_96_192"])
    if summary["iid_extra_wins"]:
        print("resolved by the i.i.d. scheme but not by the block scheme:",
              summary["iid_extra_wins"])


if __name__ == "__main__":
    main()
