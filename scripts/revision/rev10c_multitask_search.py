"""rev10c - Validation-only search for AERIS in the multi-task formulation.

The search in rev10 optimised the model for the pure classification formulation.
Section 2.4b shows that the binary label is a thresholded version of a fully
observed continuous quantity, and that learning that quantity improves every
model family. This script therefore searches the proposed architecture in the
formulation it will actually be deployed in: classification plus an auxiliary
regression of the future composite score, with the ranking score, the auxiliary
loss and the input winsorising all decided on the validation partition.

The boosting baselines receive the matching search in rev09b, so the two
families get the same opportunity and the same budget.

Usage: python scripts/revision/rev10c_multitask_search.py [--trials 14] [--seeds 42,7,13]
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

SPACE = {
    "seq_len": [4, 8],
    "hidden_dim": [96, 128, 192],
    "latent_dim": [4, 8, 16, 32],
    "attention_heads": [4, 8],
    "dropout": [0.05, 0.10, 0.20],
    "lr": [5e-4, 1e-3],
    "max_epochs": [12, 20],
    "batch_size": [128, 256],
    "ranking_weight": [0.0, 0.03, 0.10],
    "recon_weight": [0.05, 0.15],
    "kl_weight": [0.003, 0.01],
    "aux_weight": [0.3, 1.0, 3.0],
    "aux_loss": ["mse", "huber", "tail"],
    "input_clip": [None, 5.0],
    "score_mode": ["aux", "rank_blend"],
    "head_skip": [False, True],
}

INT_KEYS = {"seq_len", "hidden_dim", "latent_dim", "attention_heads", "max_epochs", "batch_size"}

# A sensible starting point evaluated as the first trial, so the search can only
# improve on a documented configuration.
SEED_CONFIG = {
    "seq_len": 8, "hidden_dim": 128, "latent_dim": 16, "attention_heads": 8,
    "dropout": 0.10, "lr": 5e-4, "max_epochs": 20, "batch_size": 128,
    "ranking_weight": 0.03, "recon_weight": 0.15, "kl_weight": 0.01,
    "aux_weight": 1.0, "aux_loss": "mse", "input_clip": 5.0, "score_mode": "rank_blend",
    "head_skip": True,
}


def sample(rng) -> dict:
    cfg = {}
    for key, values in SPACE.items():
        pick = values[int(rng.integers(len(values)))]
        cfg[key] = int(pick) if key in INT_KEYS else pick
    while cfg["hidden_dim"] % cfg["attention_heads"]:
        cfg["attention_heads"] = int(SPACE["attention_heads"][int(rng.integers(2))])
    return cfg


def build(cfg: dict) -> tuple[ModelConfig, TrainConfig]:
    m = ModelConfig(encoder="tcn", pooling="attention", use_gate=True, bottleneck="vae",
                    hidden_dim=cfg["hidden_dim"], latent_dim=cfg["latent_dim"],
                    attention_heads=cfg["attention_heads"], dropout=cfg["dropout"],
                    head_skip=cfg["head_skip"])
    t = TrainConfig(max_epochs=cfg["max_epochs"], lr=cfg["lr"], device=DEVICE,
                    batch_size=cfg["batch_size"], ranking_weight=cfg["ranking_weight"],
                    recon_weight=cfg["recon_weight"], kl_weight=cfg["kl_weight"],
                    aux_weight=cfg["aux_weight"], aux_loss=cfg["aux_loss"],
                    select_mode=cfg["score_mode"])
    return m, t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=14)
    ap.add_argument("--seeds", default="42,7,13")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    cache: dict = {}

    def data(seq_len: int, clip):
        key = (seq_len, clip)
        if key not in cache:
            cache[key] = build_matrices(quantile=0.98, horizon=1, seq_len=seq_len,
                                        normalisation="full", threshold_source="full",
                                        clip=clip)
        return cache[key]

    rng = np.random.default_rng(2026)
    rows, best = [], None
    for k in range(args.trials):
        cfg = dict(SEED_CONFIG) if k == 0 else sample(rng)
        tr, va, te, y_tr, y_va, y_te, meta = data(cfg["seq_len"], cfg["input_clip"])
        mcfg, tcfg = build(cfg)
        set_seed(42)
        model = ScreeningNet(tr.shape[-1], mcfg)
        model, info = train(model, tr, y_tr, va, y_va, replace(tcfg, seed=42),
                            aux_tr=meta["aux_train"])
        vpr = float(average_precision_score(
            y_va.astype(int), predict(model, va, DEVICE, mode=cfg["score_mode"])))
        rows.append({"Trial": k, "Valid PR_AUC": vpr, "Seconds": info["train_seconds"],
                     "Params": json.dumps(cfg)})
        print(f"  trial {k:2d}  valid PR-AUC={vpr:.4f}  ({info['train_seconds']:.0f}s)  {cfg}",
              flush=True)
        if best is None or vpr > best[0]:
            best = (vpr, cfg)
        pd.DataFrame(rows).to_csv(REV_TABLE_DIR / "rev10c_search_trials.csv",
                                  index=False, encoding="utf-8-sig")

    vpr, cfg = best
    print(f"\nbest validation PR-AUC {vpr:.4f}\n  {cfg}", flush=True)

    tr, va, te, y_tr, y_va, y_te, meta = data(cfg["seq_len"], cfg["input_clip"])
    mcfg, tcfg = build(cfg)
    seed_rows, latency = [], float("nan")
    for seed in seeds:
        set_seed(seed)
        model = ScreeningNet(tr.shape[-1], mcfg)
        model, info = train(model, tr, y_tr, va, y_va, replace(tcfg, seed=seed),
                            aux_tr=meta["aux_train"])
        vs = predict(model, va, DEVICE, mode=cfg["score_mode"])
        ts = predict(model, te, DEVICE, mode=cfg["score_mode"])
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
        # keep every seed's scores so that a seed ensemble can be assembled later
        # without retraining, and so the seed spread is auditable
        np.save(SCORE_DIR / f"aeris_seed{seed}_valid.npy", vs)
        np.save(SCORE_DIR / f"aeris_seed{seed}_test.npy", ts)
        if seed == seeds[0]:
            save_scores("AERIS", vs, ts)
            _, lat_tr = predict(model, tr, DEVICE, return_latent=True, mode=cfg["score_mode"])
            _, lat_va = predict(model, va, DEVICE, return_latent=True, mode=cfg["score_mode"])
            _, lat_te = predict(model, te, DEVICE, return_latent=True, mode=cfg["score_mode"])
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
        "search_validation_pr_auc": vpr,
        "search_trials": len(rows),
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
