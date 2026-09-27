"""rev27 - Is the longer-horizon advantage statistically resolvable?

rev21 shows the proposed model overtaking the strongest boosting regressor once
the screening horizon passes one interval. A difference in a point estimate is
not yet a result: this script retrains both methods at each horizon with the
configuration that horizon's validation partition selected, keeps the scores,
and tests every difference with the same paired moving-block bootstrap used for
the head-line comparison.

It also reports the operational reading: the share of the alert budget that is
still actionable, given that a reserve or demand-response decision cannot be
executed inside a single quarter-hour.

Usage: python scripts/revision/rev27_horizon_significance.py [--seeds 42,7,13]
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.revision.rev09b_regression_baselines import strip  # noqa: E402
from scripts.revision.rev_common import (REV_TABLE_DIR, build_matrices,  # noqa: E402
                                         summarise_sequence)
from scripts.revision.rev_eval import metric_value, paired_bootstrap_diff  # noqa: E402
from scripts.revision.rev_models import (ModelConfig, ScreeningNet, TrainConfig,  # noqa: E402
                                         _percentile_rank, predict, set_seed, train)

SCORE_DIR = Path(__file__).resolve().parents[2] / "outputs" / "revision" / "scores"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HORIZONS = (1, 2, 4, 8)


def merge_write(frame: pd.DataFrame, path, horizons) -> pd.DataFrame:
    """Replace only the rows of the horizons run now; keep every other horizon."""
    done = {f"t+{h}" for h in horizons}
    if path.exists():
        old = pd.read_csv(path)
        old = old[~old["Horizon"].isin(done)]
        frame = pd.concat([old, frame[frame["Horizon"].isin(done)]], ignore_index=True)
    order = frame["Horizon"].str.slice(2).astype(int)
    frame = frame.assign(_o=order).sort_values("_o", kind="stable").drop(columns="_o")
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return frame


def rank_mean(scores):
    return np.mean([_percentile_rank(np.asarray(s, float)) for s in scores], axis=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,7,13")
    ap.add_argument("--horizons", default=",".join(map(str, HORIZONS)))
    args = ap.parse_args()
    horizons = [int(h) for h in args.horizons.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]

    prior = json.loads((REV_TABLE_DIR / "rev10_selected_summary.json").read_text(
        encoding="utf-8"))
    cfg = prior["selected_config"]
    selected_inputs = prior.get("selected_inputs")

    choices_path = REV_TABLE_DIR / "rev21_horizon_choices.csv"
    choices = pd.read_csv(choices_path) if choices_path.exists() else pd.DataFrame()

    def aeris_settings(horizon: int) -> tuple[float, str]:
        if choices.empty:
            return cfg.get("aux_weight", 1.0), cfg.get("score_mode", "rank_blend")
        row = choices[(choices["Horizon"] == f"t+{horizon}") & (choices["Model"] == "AERIS")]
        if row.empty:
            return cfg.get("aux_weight", 1.0), cfg.get("score_mode", "rank_blend")
        text = row.iloc[0]["Selected"]
        aux = float(text.split("aux_weight=")[1].split(",")[0])
        mode = text.split("score=")[1].strip()
        return aux, mode

    reference = build_matrices(quantile=0.98, horizon=1, seq_len=cfg["seq_len"],
                               normalisation="full", threshold_source="full")
    drop = ()
    if selected_inputs:
        keep = set(selected_inputs)
        drop = tuple(c for c in reference[6]["feature_names"] if c not in keep)

    trials = pd.read_csv(REV_TABLE_DIR / "rev09b_search_trials.csv")
    hist_pool = [json.loads(p) for p in trials[trials["Model"] == "HistGBM"]
                 .sort_values("Valid PR_AUC", ascending=False).head(4)["Params"]]

    mcfg = ModelConfig(encoder="tcn", pooling="attention", use_gate=True, bottleneck="vae",
                       hidden_dim=cfg["hidden_dim"], latent_dim=cfg["latent_dim"],
                       attention_heads=cfg["attention_heads"], dropout=cfg["dropout"],
                       head_skip=cfg.get("head_skip", False))

    rows = []
    for horizon in horizons:
        aux_weight, mode = aeris_settings(horizon)
        tr, va, te, y_tr, y_va, y_te, meta = build_matrices(
            quantile=0.98, horizon=horizon, seq_len=cfg["seq_len"], normalisation="full",
            threshold_source="full", clip=cfg.get("input_clip"), drop_features=drop)
        tr_f, va_f, te_f, _, _, _, meta_f = build_matrices(
            quantile=0.98, horizon=horizon, seq_len=cfg["seq_len"], normalisation="full",
            threshold_source="full")
        y_va_i, y_te_i = y_va.astype(int), y_te.astype(int)

        tcfg = TrainConfig(max_epochs=cfg["max_epochs"], lr=cfg["lr"], device=DEVICE,
                           batch_size=cfg["batch_size"], ranking_weight=cfg["ranking_weight"],
                           recon_weight=cfg["recon_weight"], kl_weight=cfg["kl_weight"],
                           aux_weight=aux_weight, aux_loss=cfg.get("aux_loss", "mse"),
                           select_mode=mode)
        vs_list, ts_list = [], []
        for seed in seeds:
            set_seed(seed)
            model = ScreeningNet(tr.shape[-1], mcfg)
            model, _ = train(model, tr, y_tr, va, y_va, replace(tcfg, seed=seed),
                             aux_tr=meta["aux_train"])
            vs_list.append(predict(model, va, DEVICE, mode=mode))
            ts_list.append(predict(model, te, DEVICE, mode=mode))
        single = int(np.argmax([average_precision_score(y_va_i, s) for s in vs_list]))
        use_ens = average_precision_score(y_va_i, rank_mean(vs_list)) > \
            average_precision_score(y_va_i, vs_list[single])
        a_te = rank_mean(ts_list) if use_ens else ts_list[single]

        best_b = None
        for params in hist_pool:
            reg = HistGradientBoostingRegressor(**strip(params))
            reg.fit(summarise_sequence(tr_f), meta_f["aux_train"])
            vb = reg.predict(summarise_sequence(va_f))
            v = float(average_precision_score(y_va_i, vb))
            if best_b is None or v > best_b[0]:
                best_b = (v, reg.predict(summarise_sequence(te_f)))
        b_te = best_b[1]

        np.save(SCORE_DIR / f"horizon{horizon}_AERIS.npy", a_te)
        np.save(SCORE_DIR / f"horizon{horizon}_HistGBM.npy", b_te)
        np.save(SCORE_DIR / f"horizon{horizon}_y.npy", y_te_i)

        for metric in ("PR_AUC", "Recall@2%"):
            d = paired_bootstrap_diff(y_te_i, a_te, b_te, metric, block=96, rounds=2000)
            rows.append({
                "Horizon": f"t+{horizon}", "Minutes": 15 * horizon, "Metric": metric,
                "AERIS": metric_value(y_te_i, a_te, metric),
                "HistGBM": metric_value(y_te_i, b_te, metric),
                "Difference": d["observed_diff"],
                "Relative (%)": 100 * d["observed_diff"] / abs(metric_value(y_te_i, b_te, metric)),
                "CI lower": d["ci_lower"], "CI upper": d["ci_upper"],
                "p (two-sided)": d["p_two_sided"],
                "AERIS deployment": "ensemble" if use_ens else "single",
            })
        frame = merge_write(pd.DataFrame(rows),
                            REV_TABLE_DIR / "rev27_horizon_significance.csv", horizons)
        print(f"\n--- t+{horizon} ({15 * horizon} min), aux={aux_weight}, score={mode} ---")
        print(frame[frame["Horizon"] == f"t+{horizon}"][
            ["Metric", "AERIS", "HistGBM", "Relative (%)", "CI lower", "CI upper",
             "p (two-sided)"]].to_string(index=False), flush=True)

    print("\n" + pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
