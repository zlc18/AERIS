"""rev04 - Component and hyper-parameter ablation of AERIS.

Answers Reviewer 1 comment 3.  Every variant differs from the canonical model
in exactly one respect, and all of them share the training recipe defined in
rev_models.TrainConfig, so the comparison attributes performance to the
component rather than to the optimisation budget.

Usage: python scripts/revision/rev04_ablation.py [--seeds 42,7,13]
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.revision.rev_common import REV_TABLE_DIR, build_matrices  # noqa: E402
from scripts.revision.rev_eval import metric_value, select_threshold  # noqa: E402
from scripts.revision.rev_models import (ModelConfig, ScreeningNet, TrainConfig,  # noqa: E402
                                         predict, set_seed, train)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def deployed_configuration() -> tuple[ModelConfig, TrainConfig, int, float | None, str]:
    """Ablate the configuration that is actually deployed, when one has been
    selected (rev10b); otherwise fall back to the configuration of the first
    version of this work."""
    path = REV_TABLE_DIR / "rev10_selected_summary.json"
    if not path.exists():
        return (ModelConfig(encoder="tcn", pooling="attention", use_gate=True, bottleneck="vae",
                            hidden_dim=96, latent_dim=4, attention_heads=4, dropout=0.10),
                TrainConfig(max_epochs=12, device=DEVICE, ranking_weight=0.03, use_focal=True),
                8, None, "cls")
    cfg = json.loads(path.read_text(encoding="utf-8"))["selected_config"]
    m = ModelConfig(encoder="tcn", pooling="attention", use_gate=True, bottleneck="vae",
                    hidden_dim=cfg["hidden_dim"], latent_dim=cfg["latent_dim"],
                    attention_heads=cfg["attention_heads"], dropout=cfg["dropout"],
                    head_skip=cfg.get("head_skip", False))
    t = TrainConfig(max_epochs=cfg["max_epochs"], lr=cfg["lr"], device=DEVICE,
                    batch_size=cfg["batch_size"], ranking_weight=cfg["ranking_weight"],
                    recon_weight=cfg["recon_weight"], kl_weight=cfg["kl_weight"],
                    aux_weight=cfg.get("aux_weight", 0.0),
                    select_mode=cfg.get("score_mode", "cls"))
    return m, t, cfg["seq_len"], cfg.get("input_clip"), cfg.get("score_mode", "cls")


BASE_M, BASE_T, BASE_SEQ, BASE_CLIP, BASE_MODE = deployed_configuration()

# (label, model overrides, train overrides, sequence length, ablated component)
STRUCTURAL = [
    ("AERIS (full)", {}, {}, BASE_SEQ, "-"),
    ("w/o attention (mean pooling)", {"pooling": "mean"}, {}, BASE_SEQ, "attention"),
    ("w/o context gate", {"use_gate": False}, {}, BASE_SEQ, "gate"),
    ("w/o variational latent (deterministic AE)", {"bottleneck": "ae"}, {}, BASE_SEQ, "KL / sampling"),
    ("w/o latent bottleneck and reconstruction", {"bottleneck": "none"}, {}, BASE_SEQ, "VAE branch"),
    ("w/o reconstruction term", {}, {"recon_weight": 0.0}, BASE_SEQ, "reconstruction loss"),
    ("w/o focal loss (weighted BCE)", {}, {"use_focal": False}, BASE_SEQ, "focal loss"),
    ("w/o ranking loss", {}, {"ranking_weight": 0.0}, BASE_SEQ, "ranking loss"),
    # ("VAE without attention (mean pooling)", ...) was removed: the deployed
     # configuration already uses the variational bottleneck, so it built the
     # same network as "w/o attention (mean pooling)" above.
    ("Strictly causal TCN blocks", {"encoder": "causal_tcn"}, {}, BASE_SEQ, "non-causal right context"),
    ("w/o auxiliary score regression", {}, {"aux_weight": 0.0, "select_mode": "cls"}, BASE_SEQ,
     "auxiliary regression head"),
    ("Full-window reconstruction target", {"recon_target": "window", "seq_len": BASE_SEQ}, {},
     BASE_SEQ, "window-mean reconstruction replaced"),
]

LATENT_DIMS = [2, 4, 8, 16]
SEQ_LENS = [4, 8, 16, 24]
HIDDEN_DIMS = sorted({48, 96, 192, BASE_M.hidden_dim})


def run_one(mcfg: ModelConfig, tcfg: TrainConfig, seq_len: int, seed: int, cache: dict):
    if seq_len not in cache:
        cache[seq_len] = build_matrices(quantile=0.98, horizon=1, seq_len=seq_len,
                                        normalisation="full", threshold_source="full",
                                        clip=BASE_CLIP)
    tr, va, te, y_tr, y_va, y_te, meta = cache[seq_len]
    set_seed(seed)
    model = ScreeningNet(tr.shape[-1], mcfg)
    model, info = train(model, tr, y_tr, va, y_va, replace(tcfg, seed=seed),
                        aux_tr=meta["aux_train"])
    mode = tcfg.select_mode
    vs, ts = predict(model, va, DEVICE, mode=mode), predict(model, te, DEVICE, mode=mode)
    thr = select_threshold(y_va, vs)
    return {
        "PR_AUC": metric_value(y_te, ts, "PR_AUC"),
        "ROC_AUC": metric_value(y_te, ts, "ROC_AUC"),
        "F1": metric_value(y_te, ts, "F1", thr),
        "Precision@2%": metric_value(y_te, ts, "Precision@2"),
        "Recall@2%": metric_value(y_te, ts, "Recall@2"),
        "Params": info["n_parameters"],
        "Train_s": info["train_seconds"],
    }


def aggregate(label, group, records, extra=None):
    frame = pd.DataFrame(records)
    row = {"Variant": label, "Group": group}
    if extra:
        row.update(extra)
    for col in ["PR_AUC", "F1", "Precision@2%", "Recall@2%", "ROC_AUC"]:
        row[f"{col}_mean"] = frame[col].mean()
        row[f"{col}_std"] = frame[col].std(ddof=1) if len(frame) > 1 else np.nan
    row["Params"] = int(frame["Params"].iloc[0])
    row["Train_s_mean"] = frame["Train_s"].mean()
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,7,13")
    ap.add_argument("--sweep-seeds", default="42,7")
    ap.add_argument("--epochs", type=int, default=12)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    sweep_seeds = [int(s) for s in args.sweep_seeds.split(",")]
    base_t = replace(BASE_T, max_epochs=args.epochs)

    cache: dict = {}
    rows, detail = [], []

    for label, m_over, t_over, seq_len, component in STRUCTURAL:
        recs = []
        for seed in seeds:
            r = run_one(replace(BASE_M, **m_over), replace(base_t, **t_over), seq_len, seed, cache)
            recs.append(r)
            detail.append({"Variant": label, "Seed": seed, **r})
            print(f"  {label:44s} seed={seed:5d} PR-AUC={r['PR_AUC']:.4f} F1={r['F1']:.2f}", flush=True)
        rows.append(aggregate(label, "structural", recs, {"Removed component": component}))

    for d in LATENT_DIMS:
        recs = []
        for seed in sweep_seeds:
            r = run_one(replace(BASE_M, latent_dim=d), base_t, BASE_SEQ, seed, cache)
            recs.append(r); detail.append({"Variant": f"latent={d}", "Seed": seed, **r})
            print(f"  latent_dim={d:<3d}                                 seed={seed:5d} "
                  f"PR-AUC={r['PR_AUC']:.4f}", flush=True)
        rows.append(aggregate(f"Latent dimension = {d}", "latent_dim", recs, {"Removed component": "-"}))

    for L in SEQ_LENS:
        recs = []
        for seed in sweep_seeds:
            r = run_one(BASE_M, base_t, L, seed, cache)
            recs.append(r); detail.append({"Variant": f"seq={L}", "Seed": seed, **r})
            print(f"  seq_len={L:<3d}                                    seed={seed:5d} "
                  f"PR-AUC={r['PR_AUC']:.4f}", flush=True)
        rows.append(aggregate(f"Sequence length = {L}", "sequence_length", recs, {"Removed component": "-"}))

    for h in HIDDEN_DIMS:
        recs = []
        for seed in sweep_seeds:
            r = run_one(replace(BASE_M, hidden_dim=h), base_t, BASE_SEQ, seed, cache)
            recs.append(r); detail.append({"Variant": f"hidden={h}", "Seed": seed, **r})
            print(f"  hidden_dim={h:<4d}                                 seed={seed:5d} "
                  f"PR-AUC={r['PR_AUC']:.4f}", flush=True)
        rows.append(aggregate(f"Hidden dimension = {h}", "hidden_dim", recs, {"Removed component": "-"}))

    pd.DataFrame(detail).to_csv(REV_TABLE_DIR / "rev04_ablation_detail.csv",
                                index=False, encoding="utf-8-sig")
    summary = pd.DataFrame(rows)
    summary.to_csv(REV_TABLE_DIR / "rev04_ablation_summary.csv", index=False, encoding="utf-8-sig")
    print("\n" + summary.to_string(index=False))


if __name__ == "__main__":
    main()
