"""rev10b - Finalise the AERIS configuration.

Takes the winning configuration of the validation-only random search (rev10),
decides the one remaining preprocessing choice --- whether the standardised
inputs are winsorised --- on the validation partition, retrains the result over
several seeds and writes the deployed scores, latents and checkpoint.

The winsorising question arises because several inputs are extremely heavy
tailed (the wind-to-solar ratio spans nine orders of magnitude at night). Trees
are invariant to monotone transforms of an input and are therefore unaffected;
gradient-based models are not. Treating it as a hyper-parameter selected on
validation keeps the protocol identical to the one used for the baselines.

Usage: python scripts/revision/rev10b_finalise_aeris.py [--seeds 42,7,13,2024,99]
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
from scripts.revision.rev_common import REV_TABLE_DIR, build_matrices, save_scores  # noqa: E402
from scripts.revision.rev_eval import metric_value, select_threshold  # noqa: E402
from scripts.revision.rev_models import (ModelConfig, ScreeningNet, TrainConfig,  # noqa: E402
                                         inference_latency_ms, predict, set_seed, train)

SCORE_DIR = Path(__file__).resolve().parents[2] / "outputs" / "revision" / "scores"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_OPTIONS = [None, 5.0]
# Weight of the auxiliary head that regresses the continuous future composite
# score, and the score the ranking is read from. Both are decided on validation.
AUX_WEIGHTS = [0.0, 0.3, 1.0]
SCORE_MODES = ["cls", "aux", "rank_blend"]


def build(cfg: dict) -> tuple[ModelConfig, TrainConfig]:
    m = ModelConfig(encoder="tcn", pooling="attention", use_gate=True, bottleneck="vae",
                    hidden_dim=cfg["hidden_dim"], latent_dim=cfg["latent_dim"],
                    attention_heads=cfg["attention_heads"], dropout=cfg["dropout"])
    t = TrainConfig(max_epochs=cfg["max_epochs"], lr=cfg["lr"], device=DEVICE,
                    batch_size=cfg["batch_size"], ranking_weight=cfg["ranking_weight"],
                    recon_weight=cfg["recon_weight"], kl_weight=cfg["kl_weight"])
    return m, t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,7,13,2024,99")
    ap.add_argument("--select-seeds", type=int, default=2)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    trials = pd.read_csv(REV_TABLE_DIR / "rev10_search_trials.csv")
    best = trials.sort_values("Valid PR_AUC", ascending=False).iloc[0]
    cfg = json.loads(best["Params"])
    print(f"winning search configuration (validation PR-AUC {best['Valid PR_AUC']:.4f}):")
    print(f"  {cfg}")
    mcfg, tcfg = build(cfg)

    cache: dict = {}

    def data(clip):
        if clip not in cache:
            cache[clip] = build_matrices(quantile=0.98, horizon=1, seq_len=cfg["seq_len"],
                                         normalisation="full", threshold_source="full",
                                         clip=clip)
        return cache[clip]

    rows = []
    for clip in CLIP_OPTIONS:
        for aux_weight in AUX_WEIGHTS:
            tr, va, te, y_tr, y_va, y_te, m_data = data(clip)
            # One training run per (winsorising, auxiliary weight); the three
            # ranking scores are then read from the same checkpoint. The
            # checkpoint itself is selected on the blended score whenever the
            # auxiliary head is active, so that neither head is favoured.
            per_mode: dict[str, list[float]] = {m: [] for m in SCORE_MODES}
            select_on = "cls" if aux_weight == 0.0 else "rank_blend"
            for seed in seeds[:args.select_seeds]:
                set_seed(seed)
                m = ScreeningNet(tr.shape[-1], mcfg)
                m, _ = train(m, tr, y_tr, va, y_va,
                             replace(tcfg, seed=seed, aux_weight=aux_weight,
                                     select_mode=select_on),
                             aux_tr=m_data["aux_train"])
                for mode in SCORE_MODES:
                    if aux_weight == 0.0 and mode != "cls":
                        continue
                    per_mode[mode].append(float(average_precision_score(
                        y_va.astype(int), predict(m, va, DEVICE, mode=mode))))
            for mode, vals in per_mode.items():
                if not vals:
                    continue
                rows.append({
                    "Winsorising": "none" if clip is None else f"+/-{clip:g}",
                    "Auxiliary weight": aux_weight,
                    "Ranking score": mode,
                    "Valid PR_AUC mean": float(np.mean(vals)),
                    "Valid PR_AUC std": float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan,
                    "Seeds": len(vals),
                    "_clip": clip,
                })
                print(f"  clip={clip} aux={aux_weight} score={mode}: "
                      f"validation PR-AUC {np.mean(vals):.4f}", flush=True)

    choice = pd.DataFrame(rows)
    choice.drop(columns=["_clip"]).to_csv(REV_TABLE_DIR / "rev10b_formulation_choice.csv",
                                          index=False, encoding="utf-8-sig")
    best_row = choice.sort_values("Valid PR_AUC mean", ascending=False).iloc[0]
    best_clip = best_row["_clip"]
    best_aux = float(best_row["Auxiliary weight"])
    best_mode = str(best_row["Ranking score"])
    cfg.update({"input_clip": best_clip, "aux_weight": best_aux, "score_mode": best_mode})
    print(f"  selected: winsorising={best_clip}, auxiliary weight={best_aux}, "
          f"ranking score={best_mode} (validation PR-AUC {best_row['Valid PR_AUC mean']:.4f})")

    tr, va, te, y_tr, y_va, y_te, m_data = data(best_clip)
    tcfg = replace(tcfg, aux_weight=best_aux, select_mode=best_mode)
    seed_rows, latency = [], float("nan")
    for seed in seeds:
        set_seed(seed)
        model = ScreeningNet(tr.shape[-1], mcfg)
        model, info = train(model, tr, y_tr, va, y_va, replace(tcfg, seed=seed),
                            aux_tr=m_data["aux_train"])
        vs = predict(model, va, DEVICE, mode=best_mode)
        ts = predict(model, te, DEVICE, mode=best_mode)
        thr = select_threshold(y_va, vs)
        seed_rows.append({
            "Seed": seed,
            "PR_AUC": metric_value(y_te, ts, "PR_AUC"),
            "ROC_AUC": metric_value(y_te, ts, "ROC_AUC"),
            "F1": metric_value(y_te, ts, "F1", thr),
            "Precision@1%": metric_value(y_te, ts, "Precision@1"),
            "Recall@1%": metric_value(y_te, ts, "Recall@1"),
            "Precision@2%": metric_value(y_te, ts, "Precision@2"),
            "Recall@2%": metric_value(y_te, ts, "Recall@2"),
            "Accuracy": metric_value(y_te, ts, "Accuracy", thr),
            "Valid PR_AUC": info["best_valid_pr_auc"],
            "Params": info["n_parameters"],
            "Train_s": info["train_seconds"],
        })
        print(f"  seed {seed:5d}  test PR-AUC={seed_rows[-1]['PR_AUC']:.4f}  "
              f"F1={seed_rows[-1]['F1']:.2f}  R@2%={seed_rows[-1]['Recall@2%']:.1f}", flush=True)
        if seed == seeds[0]:
            save_scores("AERIS", vs, ts)
            _, lat_tr = predict(model, tr, DEVICE, return_latent=True, mode=best_mode)
            _, lat_va = predict(model, va, DEVICE, return_latent=True, mode=best_mode)
            _, lat_te = predict(model, te, DEVICE, return_latent=True, mode=best_mode)
            np.save(SCORE_DIR / "latent_train_AERIS.npy", lat_tr)
            np.save(SCORE_DIR / "latent_valid_AERIS.npy", lat_va)
            np.save(SCORE_DIR / "latent_test_AERIS.npy", lat_te)
            np.save(SCORE_DIR / "y_test.npy", y_te)
            np.save(SCORE_DIR / "y_valid.npy", y_va)
            np.save(SCORE_DIR / "y_train.npy", y_tr)
            torch.save(model.state_dict(), SCORE_DIR / "AERIS_selected.pt")
            latency = inference_latency_ms(model, te, DEVICE)

    detail = pd.DataFrame(seed_rows)
    detail.to_csv(REV_TABLE_DIR / "rev10_selected_per_seed.csv", index=False, encoding="utf-8-sig")
    summary = {
        "selected_config": cfg,
        "search_validation_pr_auc": float(best["Valid PR_AUC"]),
        "inference_ms_per_sample": float(latency),
        "n_parameters": int(detail["Params"].iloc[0]),
        "seeds": seeds,
        "test": {c: [float(detail[c].mean()), float(detail[c].std(ddof=1))]
                 for c in ["PR_AUC", "ROC_AUC", "F1", "Precision@1%", "Recall@1%",
                           "Precision@2%", "Recall@2%", "Accuracy"]},
    }
    (REV_TABLE_DIR / "rev10_selected_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print("\n" + detail.to_string(index=False))
    print("\n" + json.dumps(summary["test"], indent=2))


if __name__ == "__main__":
    main()
