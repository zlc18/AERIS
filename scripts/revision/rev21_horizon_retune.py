"""rev21 - Multi-horizon comparison with per-horizon model selection.

rev20 fixes every configuration at the value chosen for t+1 and reuses it at all
horizons. That is symmetric across families but it is not how any of these models
would be deployed at a longer lead time, and it penalises whichever family is
more sensitive to the horizon. Here each family re-selects its configuration on
the validation partition separately at every horizon, from its own candidate
pool, with the test partition untouched.

Candidate pools
  AERIS    the auxiliary weight and the ranking score, the two settings that
           depend most directly on how predictable the target still is
  HistGBM  the top configurations of its own regression search

Usage: python scripts/revision/rev21_horizon_retune.py [--seeds 42,7]
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
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.revision.rev09b_regression_baselines import strip  # noqa: E402
from scripts.revision.rev_common import (REV_TABLE_DIR, build_matrices,  # noqa: E402
                                         summarise_sequence)
from scripts.revision.rev_eval import metric_value, select_threshold  # noqa: E402
from scripts.revision.rev_models import (ModelConfig, ScreeningNet, TrainConfig,  # noqa: E402
                                         predict, set_seed, train)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HORIZONS = (1, 2, 4, 8)
AUX_CANDIDATES = (0.3, 1.0)
SCORE_MODES = ("cls", "aux", "rank_blend")
N_BOOSTING_CANDIDATES = 4


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,7")
    ap.add_argument("--horizons", default=",".join(map(str, HORIZONS)))
    ap.add_argument("--aux-weights", default=None,
                    help="evaluate only these auxiliary weights (default: the full grid)")
    args = ap.parse_args()
    horizons = [int(h) for h in args.horizons.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]

    summary = json.loads((REV_TABLE_DIR / "rev10_selected_summary.json").read_text(
        encoding="utf-8"))
    cfg = summary["selected_config"]
    selected_inputs = summary.get("selected_inputs")
    mcfg = ModelConfig(encoder="tcn", pooling="attention", use_gate=True, bottleneck="vae",
                       hidden_dim=cfg["hidden_dim"], latent_dim=cfg["latent_dim"],
                       attention_heads=cfg["attention_heads"], dropout=cfg["dropout"],
                       head_skip=cfg.get("head_skip", False))
    base_t = TrainConfig(max_epochs=cfg["max_epochs"], lr=cfg["lr"], device=DEVICE,
                         batch_size=cfg["batch_size"], ranking_weight=cfg["ranking_weight"],
                         recon_weight=cfg["recon_weight"], kl_weight=cfg["kl_weight"],
                         aux_loss=cfg.get("aux_loss", "mse"))

    reference = build_matrices(quantile=0.98, horizon=1, seq_len=cfg["seq_len"],
                               normalisation="full", threshold_source="full")
    drop = ()
    if selected_inputs:
        keep = set(selected_inputs)
        drop = tuple(c for c in reference[6]["feature_names"] if c not in keep)

    trials = pd.read_csv(REV_TABLE_DIR / "rev09b_search_trials.csv")
    hist_candidates = [json.loads(p) for p in trials[trials["Model"] == "HistGBM"]
                       .sort_values("Valid PR_AUC", ascending=False)
                       .head(N_BOOSTING_CANDIDATES)["Params"]]
    from sklearn.ensemble import HistGradientBoostingRegressor

    rows, choices = [], []
    for horizon in horizons:
        tr, va, te, y_tr, y_va, y_te, meta = build_matrices(
            quantile=0.98, horizon=horizon, seq_len=cfg["seq_len"], normalisation="full",
            threshold_source="full", clip=cfg.get("input_clip"), drop_features=drop)
        tr_f, va_f, te_f, _, _, _, meta_f = build_matrices(
            quantile=0.98, horizon=horizon, seq_len=cfg["seq_len"], normalisation="full",
            threshold_source="full")
        y_va_i, y_te_i = y_va.astype(int), y_te.astype(int)

        # ---- proposed model -------------------------------------------
        best = None
        # the grid always contains the weight selected one step ahead, so that the
        # deployed configuration is among the candidates at every horizon
        grid = sorted(set(AUX_CANDIDATES) | {float(cfg.get("aux_weight", 1.0))})
        if args.aux_weights:
            grid = [float(w) for w in args.aux_weights.split(",")]
        for aux_weight in grid:
            per_seed = []
            for seed in seeds:
                set_seed(seed)
                model = ScreeningNet(tr.shape[-1], mcfg)
                model, _ = train(model, tr, y_tr, va, y_va,
                                 replace(base_t, seed=seed, aux_weight=aux_weight,
                                         select_mode="rank_blend"),
                                 aux_tr=meta["aux_train"])
                per_seed.append(model)
            for mode in SCORE_MODES:
                v = float(np.mean([average_precision_score(
                    y_va_i, predict(m, va, DEVICE, mode=mode)) for m in per_seed]))
                if best is None or v > best[0]:
                    ts = np.mean([predict(m, te, DEVICE, mode=mode) for m in per_seed], axis=0)
                    vs = np.mean([predict(m, va, DEVICE, mode=mode) for m in per_seed], axis=0)
                    best = (v, aux_weight, mode, vs, ts)
            print(f"  [t+{horizon}] AERIS aux={aux_weight} evaluated", flush=True)
        v, aux_weight, mode, vs, ts = best
        thr = select_threshold(y_va, vs)
        rows.append({"Horizon": f"t+{horizon}", "Minutes": 15 * horizon, "Model": "AERIS",
                     "Events": int(y_te.sum()), "Valid PR_AUC": v,
                     "Test PR_AUC": metric_value(y_te_i, ts, "PR_AUC"),
                     "Test Recall@2%": metric_value(y_te_i, ts, "Recall@2"),
                     "Test F1": metric_value(y_te_i, ts, "F1", thr)})
        choices.append({"Horizon": f"t+{horizon}", "Model": "AERIS",
                        "Selected": f"aux_weight={aux_weight}, score={mode}"})

        # ---- boosting reference ---------------------------------------
        best_b = None
        for params in hist_candidates:
            reg = HistGradientBoostingRegressor(**strip(params))
            reg.fit(summarise_sequence(tr_f), meta_f["aux_train"])
            vb = reg.predict(summarise_sequence(va_f))
            v = float(average_precision_score(y_va_i, vb))
            if best_b is None or v > best_b[0]:
                best_b = (v, params, vb, reg.predict(summarise_sequence(te_f)))
        v, params, vb, tb = best_b
        thr_b = select_threshold(y_va, vb)
        rows.append({"Horizon": f"t+{horizon}", "Minutes": 15 * horizon,
                     "Model": "HistGBM (regression, tuned)", "Events": int(y_te.sum()),
                     "Valid PR_AUC": v,
                     "Test PR_AUC": metric_value(y_te_i, tb, "PR_AUC"),
                     "Test Recall@2%": metric_value(y_te_i, tb, "Recall@2"),
                     "Test F1": metric_value(y_te_i, tb, "F1", thr_b)})
        choices.append({"Horizon": f"t+{horizon}", "Model": "HistGBM",
                        "Selected": json.dumps({k: v2 for k, v2 in params.items()
                                                if k in ("max_iter", "learning_rate",
                                                         "max_depth", "max_leaf_nodes")})})

        persistence = meta["test_frame"]["event_proxy_score"].to_numpy(dtype=float)
        rows.append({"Horizon": f"t+{horizon}", "Minutes": 15 * horizon,
                     "Model": "Persistence heuristic", "Events": int(y_te.sum()),
                     "Valid PR_AUC": np.nan,
                     "Test PR_AUC": metric_value(y_te_i, persistence, "PR_AUC"),
                     "Test Recall@2%": metric_value(y_te_i, persistence, "Recall@2"),
                     "Test F1": np.nan})

        frame = merge_write(pd.DataFrame(rows), REV_TABLE_DIR / "rev21_horizon_retuned.csv",
                            horizons)
        merge_write(pd.DataFrame(choices), REV_TABLE_DIR / "rev21_horizon_choices.csv",
                    horizons)
        print(f"\n--- t+{horizon} ({15 * horizon} min) ---")
        print(frame[frame["Horizon"] == f"t+{horizon}"][
            ["Model", "Valid PR_AUC", "Test PR_AUC", "Test Recall@2%"]].to_string(index=False),
            flush=True)

    print("\n" + pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
