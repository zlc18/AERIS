"""rev10 - Validation-based random search for AERIS.

The proposed model receives a random search of the same size as the one given
to the tuned boosting baselines in rev09, scored exclusively on validation
PR-AUC.  The winning configuration is then retrained over several seeds and its
test scores are written to the shared score directory, so that the head-line
comparison puts a tuned proposed model against tuned baselines.

Usage: python scripts/revision/rev10_aeris_search.py [--trials 24] [--seeds 42,7,13,2024,99]
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
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
SCORE_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SPACE = {
    "seq_len": [4, 8, 12, 16],
    "hidden_dim": [64, 96, 128],
    "latent_dim": [2, 4, 8, 16],
    "attention_heads": [2, 4, 8],
    "dropout": [0.05, 0.10, 0.20],
    "lr": [3e-4, 5e-4, 1e-3],
    "max_epochs": [8, 12, 20],
    "ranking_weight": [0.0, 0.03, 0.10, 0.30],
    "recon_weight": [0.05, 0.15, 0.30],
    "kl_weight": [0.003, 0.01, 0.03],
    "batch_size": [64, 128, 256],
}

# The configuration reported in the first version of this work is evaluated as the
# incumbent trial, so the search can only improve on a documented starting point.
INCUMBENT = {
    "seq_len": 8, "hidden_dim": 96, "latent_dim": 4, "attention_heads": 4,
    "dropout": 0.10, "lr": 5e-4, "max_epochs": 12, "ranking_weight": 0.03,
    "recon_weight": 0.15, "kl_weight": 0.01, "batch_size": 64,
}


def sample(rng) -> dict:
    cfg = {k: rng.choice(v) for k, v in SPACE.items()}
    cfg = {k: (int(v) if k in ("seq_len", "hidden_dim", "latent_dim", "attention_heads",
                               "max_epochs", "batch_size") else float(v))
           for k, v in cfg.items()}
    # multi-head attention requires hidden_dim divisible by the head count
    while cfg["hidden_dim"] % cfg["attention_heads"]:
        cfg["attention_heads"] = int(rng.choice(SPACE["attention_heads"]))
    return cfg


def build(cfg: dict) -> tuple[ModelConfig, TrainConfig]:
    m = ModelConfig(encoder="tcn", pooling="attention", use_gate=True, bottleneck="vae",
                    hidden_dim=cfg["hidden_dim"], latent_dim=cfg["latent_dim"],
                    attention_heads=cfg["attention_heads"], dropout=cfg["dropout"])
    t = TrainConfig(max_epochs=cfg["max_epochs"], lr=cfg["lr"], device=DEVICE,
                    batch_size=cfg["batch_size"],
                    ranking_weight=cfg["ranking_weight"], recon_weight=cfg["recon_weight"],
                    kl_weight=cfg["kl_weight"])
    return m, t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=24)
    ap.add_argument("--seeds", default="42,7,13,2024,99")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    cache: dict[tuple[int, float | None], tuple] = {}

    def data(seq_len: int, clip: float | None = None):
        key = (seq_len, clip)
        if key not in cache:
            cache[key] = build_matrices(quantile=0.98, horizon=1, seq_len=seq_len,
                                        normalisation="full", threshold_source="full",
                                        clip=clip)
        return cache[key]

    rng = np.random.default_rng(42)
    rows, best = [], None
    for k in range(args.trials):
        cfg = dict(INCUMBENT) if k == 0 else sample(rng)
        tr, va, te, y_tr, y_va, y_te, _ = data(cfg["seq_len"])
        mcfg, tcfg = build(cfg)
        set_seed(42)
        model = ScreeningNet(tr.shape[-1], mcfg)
        model, info = train(model, tr, y_tr, va, y_va, replace(tcfg, seed=42))
        vs = predict(model, va, DEVICE)
        vpr = float(average_precision_score(y_va.astype(int), vs))
        rows.append({"Trial": k, "Valid PR_AUC": vpr, "Seconds": info["train_seconds"],
                     "Params": json.dumps(cfg)})
        print(f"  trial {k:2d}  valid PR-AUC={vpr:.4f}  ({info['train_seconds']:.0f}s)  {cfg}",
              flush=True)
        if best is None or vpr > best[0]:
            best = (vpr, cfg)
        pd.DataFrame(rows).to_csv(REV_TABLE_DIR / "rev10_search_trials.csv",
                                  index=False, encoding="utf-8-sig")

    vpr, cfg = best
    print(f"\nbest validation PR-AUC {vpr:.4f} with {cfg}")

    # Winsorising the standardised inputs is a preprocessing choice; it is decided
    # on the validation partition like every other hyper-parameter.
    mcfg, tcfg = build(cfg)
    clip_rows = []
    for clip in (None, 5.0):
        tr, va, te, y_tr, y_va, y_te, _ = data(cfg["seq_len"], clip)
        scores = []
        for seed in seeds[:2]:
            set_seed(seed)
            m = ScreeningNet(tr.shape[-1], mcfg)
            m, _ = train(m, tr, y_tr, va, y_va, replace(tcfg, seed=seed))
            scores.append(float(average_precision_score(
                y_va.astype(int), predict(m, va, DEVICE))))
        clip_rows.append({"Clip": "none" if clip is None else f"+/-{clip:g}",
                          "Valid PR_AUC mean": float(np.mean(scores)),
                          "Seeds": len(scores)})
        print(f"  preprocessing clip={clip}: mean validation PR-AUC "
              f"{np.mean(scores):.4f}", flush=True)
    pd.DataFrame(clip_rows).to_csv(REV_TABLE_DIR / "rev10_preprocessing_choice.csv",
                                   index=False, encoding="utf-8-sig")
    best_clip = None if clip_rows[0]["Valid PR_AUC mean"] >= clip_rows[1]["Valid PR_AUC mean"] else 5.0
    cfg["input_clip"] = best_clip
    print(f"  selected preprocessing: clip={best_clip}")

    tr, va, te, y_tr, y_va, y_te, _ = data(cfg["seq_len"], best_clip)

    seed_rows = []
    for seed in seeds:
        set_seed(seed)
        model = ScreeningNet(tr.shape[-1], mcfg)
        model, info = train(model, tr, y_tr, va, y_va, replace(tcfg, seed=seed))
        vs, ts = predict(model, va, DEVICE), predict(model, te, DEVICE)
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
              f"F1={seed_rows[-1]['F1']:.2f}", flush=True)
        if seed == seeds[0]:
            save_scores("AERIS", vs, ts)
            _, lat_tr = predict(model, tr, DEVICE, return_latent=True)
            _, lat_va = predict(model, va, DEVICE, return_latent=True)
            _, lat_te = predict(model, te, DEVICE, return_latent=True)
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
    summary = {"selected_config": cfg, "validation_pr_auc": vpr,
               "inference_ms_per_sample": float(latency),
               "n_parameters": int(detail["Params"].iloc[0]),
               "test": {c: [float(detail[c].mean()), float(detail[c].std(ddof=1))]
                        for c in ["PR_AUC", "ROC_AUC", "F1", "Precision@1%", "Recall@1%",
                                  "Precision@2%", "Recall@2%", "Accuracy"]}}
    (REV_TABLE_DIR / "rev10_selected_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print("\n" + detail.to_string(index=False))
    print("\n" + json.dumps(summary["test"], indent=2))


if __name__ == "__main__":
    main()
