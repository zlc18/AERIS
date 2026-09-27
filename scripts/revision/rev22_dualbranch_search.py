"""rev22 - AERIS with an explicit window-summary branch.

The strongest competing method is a boosting regressor fitted on the per-feature
mean, terminal value and maximum of the window. That is an *input representation*
advantage, not an architectural one: the boosting machine gets axis-aligned
access to the near-autoregressive structure, while the sequence model has to
recover it through a projection, a temporal encoder and a latent bottleneck.

This script removes that asymmetry by giving AERIS the same window summary
through a second branch, fused with the temporal context before the latent layer
and again at the prediction heads. The temporal encoder, the variational latent
state and the density layer are unchanged, so the latent representation the
confidence analysis relies on is preserved.

The branch is treated as a hyper-parameter and selected on validation, exactly
like every other setting, and the comparison it enters is against boosting
models fitted on the identical summary features.

Usage: python scripts/revision/rev22_dualbranch_search.py [--trials 10] [--seeds 42,7,13]
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
from scripts.revision.rev_common import (REV_TABLE_DIR, build_matrices,  # noqa: E402
                                         save_scores, summarise_sequence)
from scripts.revision.rev_eval import metric_value, select_threshold  # noqa: E402
from scripts.revision.rev_models import (ModelConfig, ScreeningNet, TrainConfig,  # noqa: E402
                                         _percentile_rank, inference_latency_ms,
                                         predict, set_seed, train)

SCORE_DIR = Path(__file__).resolve().parents[2] / "outputs" / "revision" / "scores"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SPACE = {
    "hidden_dim": [96, 128, 192],
    "latent_dim": [8, 16, 32],
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
}
INT_KEYS = {"hidden_dim", "latent_dim", "attention_heads", "max_epochs", "batch_size"}
SEQ_LEN = 8

SEED_CONFIG = {
    "hidden_dim": 128, "latent_dim": 16, "attention_heads": 8, "dropout": 0.10,
    "lr": 5e-4, "max_epochs": 20, "batch_size": 128, "ranking_weight": 0.03,
    "recon_weight": 0.15, "kl_weight": 0.01, "aux_weight": 1.0, "aux_loss": "mse",
    "input_clip": 5.0, "score_mode": "rank_blend",
}


def sample(rng) -> dict:
    cfg = {}
    for key, values in SPACE.items():
        pick = values[int(rng.integers(len(values)))]
        cfg[key] = int(pick) if key in INT_KEYS else pick
    while cfg["hidden_dim"] % cfg["attention_heads"]:
        cfg["attention_heads"] = int(SPACE["attention_heads"][int(rng.integers(2))])
    return cfg


def build(cfg: dict, summary_dim: int) -> tuple[ModelConfig, TrainConfig]:
    m = ModelConfig(encoder="tcn", pooling="attention", use_gate=True, bottleneck="vae",
                    hidden_dim=cfg["hidden_dim"], latent_dim=cfg["latent_dim"],
                    attention_heads=cfg["attention_heads"], dropout=cfg["dropout"],
                    head_skip=cfg.get("head_skip", False), summary_dim=summary_dim)
    t = TrainConfig(max_epochs=cfg["max_epochs"], lr=cfg["lr"], device=DEVICE,
                    batch_size=cfg["batch_size"], ranking_weight=cfg["ranking_weight"],
                    recon_weight=cfg["recon_weight"], kl_weight=cfg["kl_weight"],
                    aux_weight=cfg["aux_weight"], aux_loss=cfg["aux_loss"],
                    select_mode=cfg["score_mode"])
    return m, t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--seeds", default="42,7,13")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    cache: dict = {}

    def data(clip):
        if clip not in cache:
            built = build_matrices(quantile=0.98, horizon=1, seq_len=SEQ_LEN,
                                   normalisation="full", threshold_source="full", clip=clip)
            tr, va, te = built[0], built[1], built[2]
            summaries = (summarise_sequence(tr), summarise_sequence(va), summarise_sequence(te))
            cache[clip] = (built, summaries)
        return cache[clip]

    rng = np.random.default_rng(31415)
    rows, best = [], None
    for k in range(args.trials):
        cfg = dict(SEED_CONFIG) if k == 0 else sample(rng)
        (tr, va, te, y_tr, y_va, y_te, meta), (s_tr, s_va, s_te) = data(cfg["input_clip"])
        mcfg, tcfg = build(cfg, s_tr.shape[1])
        set_seed(42)
        model = ScreeningNet(tr.shape[-1], mcfg)
        model, info = train(model, tr, y_tr, va, y_va, replace(tcfg, seed=42),
                            aux_tr=meta["aux_train"], summary_tr=s_tr, summary_va=s_va)
        vpr = float(average_precision_score(
            y_va.astype(int),
            predict(model, va, DEVICE, mode=cfg["score_mode"], summary=s_va)))
        rows.append({"Trial": k, "Valid PR_AUC": vpr, "Seconds": info["train_seconds"],
                     "Params": json.dumps(cfg)})
        print(f"  trial {k:2d}  valid PR-AUC={vpr:.4f}  ({info['train_seconds']:.0f}s)",
              flush=True)
        if best is None or vpr > best[0]:
            best = (vpr, cfg)
        pd.DataFrame(rows).to_csv(REV_TABLE_DIR / "rev22_search_trials.csv",
                                  index=False, encoding="utf-8-sig")

    vpr, cfg = best
    print(f"\nbest validation PR-AUC {vpr:.4f}\n  {cfg}", flush=True)

    (tr, va, te, y_tr, y_va, y_te, meta), (s_tr, s_va, s_te) = data(cfg["input_clip"])
    mcfg, tcfg = build(cfg, s_tr.shape[1])
    seed_rows, valid_scores, test_scores, latency = [], [], [], float("nan")
    for seed in seeds:
        set_seed(seed)
        model = ScreeningNet(tr.shape[-1], mcfg)
        model, info = train(model, tr, y_tr, va, y_va, replace(tcfg, seed=seed),
                            aux_tr=meta["aux_train"], summary_tr=s_tr, summary_va=s_va)
        vs = predict(model, va, DEVICE, mode=cfg["score_mode"], summary=s_va)
        ts = predict(model, te, DEVICE, mode=cfg["score_mode"], summary=s_te)
        valid_scores.append(vs)
        test_scores.append(ts)
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
              f"R@2%={seed_rows[-1]['Recall@2%']:.1f}", flush=True)
        if seed == seeds[0]:
            _, lat_tr = predict(model, tr, DEVICE, return_latent=True,
                                mode=cfg["score_mode"], summary=s_tr)
            _, lat_va = predict(model, va, DEVICE, return_latent=True,
                                mode=cfg["score_mode"], summary=s_va)
            _, lat_te = predict(model, te, DEVICE, return_latent=True,
                                mode=cfg["score_mode"], summary=s_te)
            torch.save(model.state_dict(), SCORE_DIR / "AERIS_dualbranch.pt")
            latency = inference_latency_ms(model, te, DEVICE, summary=s_te)

    # the seed ensemble is compared with the single best seed on validation
    def rank_mean(scores):
        return np.mean([_percentile_rank(np.asarray(s, float)) for s in scores], axis=0)

    single = int(np.argmax([average_precision_score(y_va.astype(int), s) for s in valid_scores]))
    v_single = float(average_precision_score(y_va.astype(int), valid_scores[single]))
    v_ens = float(average_precision_score(y_va.astype(int), rank_mean(valid_scores)))
    use_ens = v_ens > v_single
    vs_final = rank_mean(valid_scores) if use_ens else valid_scores[single]
    ts_final = rank_mean(test_scores) if use_ens else test_scores[single]
    print(f"  validation: single {v_single:.4f} / ensemble {v_ens:.4f} -> "
          f"{'ensemble' if use_ens else 'single'}")

    detail = pd.DataFrame(seed_rows)
    detail.to_csv(REV_TABLE_DIR / "rev22_dualbranch_per_seed.csv",
                  index=False, encoding="utf-8-sig")
    thr = select_threshold(y_va, vs_final)
    final = {
        "selected_config": {**cfg, "seq_len": SEQ_LEN, "summary_branch": True,
                            "head_skip": cfg.get("head_skip", False), "deployment": "ensemble" if use_ens else "single"},
        "search_validation_pr_auc": vpr,
        "inference_ms_per_sample": float(latency),
        "n_parameters": int(detail["Params"].iloc[0]),
        "test_deployed": {
            "PR_AUC": metric_value(y_te, ts_final, "PR_AUC"),
            "ROC_AUC": metric_value(y_te, ts_final, "ROC_AUC"),
            "F1": metric_value(y_te, ts_final, "F1", thr),
            "Precision@1%": metric_value(y_te, ts_final, "Precision@1"),
            "Recall@1%": metric_value(y_te, ts_final, "Recall@1"),
            "Precision@2%": metric_value(y_te, ts_final, "Precision@2"),
            "Recall@2%": metric_value(y_te, ts_final, "Recall@2"),
        },
        "test_per_seed": {c: [float(detail[c].mean()), float(detail[c].std(ddof=1))]
                          for c in ["PR_AUC", "Recall@2%", "Precision@2%"]},
    }
    (REV_TABLE_DIR / "rev22_dualbranch_summary.json").write_text(
        json.dumps(final, indent=2), encoding="utf-8")

    np.save(SCORE_DIR / "dualbranch_latent_train.npy", lat_tr)
    np.save(SCORE_DIR / "dualbranch_latent_valid.npy", lat_va)
    np.save(SCORE_DIR / "dualbranch_latent_test.npy", lat_te)
    save_scores("AERIS-DB", vs_final, ts_final)

    print("\n" + detail.to_string(index=False))
    print("\ndeployed: " + json.dumps(final["test_deployed"], indent=2))


if __name__ == "__main__":
    main()
