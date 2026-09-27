"""rev03 - Sequence-learning baselines and the proposed model under one protocol.

Answers Reviewer 1 comment 4 and Reviewer 2 comment 3: LSTM, GRU, Transformer
and a plain TCN classifier are trained with the identical optimiser, schedule,
loss weighting, checkpoint rule and threshold selection as AERIS, so that the
comparison isolates the architecture rather than the training recipe.  An
unsupervised sequence VAE scored by reconstruction error is added as the
representation-learning-without-supervision reference.

Usage:  python scripts/revision/rev03_sequence_models.py [--seeds 42,7,13,2024,99]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.revision.rev_common import REV_TABLE_DIR, build_matrices, save_scores  # noqa: E402
from scripts.revision.rev_eval import metric_value, select_threshold  # noqa: E402
from scripts.revision.rev_models import (ModelConfig, ScreeningNet, TrainConfig,  # noqa: E402
                                         inference_latency_ms, predict, set_seed, train)

SCORE_DIR = Path(__file__).resolve().parents[2] / "outputs" / "revision" / "scores"
SCORE_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ARCHITECTURES = {
    "AERIS (common recipe)": ModelConfig(encoder="tcn", pooling="attention", use_gate=True, bottleneck="vae"),
    "TCN classifier": ModelConfig(encoder="tcn", pooling="mean", use_gate=False, bottleneck="none"),
    "LSTM classifier": ModelConfig(encoder="lstm", pooling="last", use_gate=False, bottleneck="none"),
    "GRU classifier": ModelConfig(encoder="gru", pooling="last", use_gate=False, bottleneck="none"),
    "Transformer classifier": ModelConfig(encoder="transformer", pooling="mean",
                                          use_gate=False, bottleneck="none"),
}


def train_unsupervised_vae(tr: np.ndarray, y_tr: np.ndarray, va: np.ndarray, te: np.ndarray,
                           input_dim: int, seed: int = 42, epochs: int = 12):
    """Sequence VAE fitted on non-event training windows; score = reconstruction error."""
    set_seed(seed)
    cfg = ModelConfig(encoder="tcn", pooling="attention", use_gate=True, bottleneck="vae")
    model = ScreeningNet(input_dim, cfg).to(DEVICE)
    normal = tr[y_tr == 0]
    loader = DataLoader(TensorDataset(torch.tensor(normal, dtype=torch.float32)),
                        batch_size=128, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=2e-4)
    mse = nn.MSELoss()
    for _ in range(epochs):
        model.train()
        for (bx,) in loader:
            bx = bx.to(DEVICE)
            opt.zero_grad()
            _, recon, tgt, mu, logvar, _, _ = model(bx)
            loss = mse(recon, tgt) + 0.01 * (-0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp()))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    def recon_error(x: np.ndarray) -> np.ndarray:
        out = []
        model.eval()
        with torch.no_grad():
            for (bx,) in DataLoader(TensorDataset(torch.tensor(x, dtype=torch.float32)),
                                    batch_size=512, shuffle=False):
                bx = bx.to(DEVICE)
                _, recon, tgt, _, _, _, _ = model(bx)
                out.append(((recon - tgt) ** 2).mean(dim=1).cpu().numpy())
        return np.concatenate(out)

    return recon_error(va), recon_error(te)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="42,7,13,2024,99")
    parser.add_argument("--epochs", type=int, default=12)
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    torch.set_num_threads(max(1, (torch.get_num_threads() or 4)))
    tr, va, te, y_tr, y_va, y_te, meta = build_matrices(
        quantile=0.98, horizon=1, seq_len=8, normalisation="full", threshold_source="full")
    print(f"device={DEVICE}  features={meta['n_features']}  "
          f"train={len(tr)} valid={len(va)} test={len(te)}  positives(test)={int(y_te.sum())}")

    rows, per_seed = [], []
    for name, mcfg in ARCHITECTURES.items():
        seed_metrics = []
        for seed in seeds:
            set_seed(seed)
            model = ScreeningNet(tr.shape[-1], mcfg)
            tcfg = TrainConfig(max_epochs=args.epochs, seed=seed, device=DEVICE,
                               ranking_weight=0.03)
            model, info = train(model, tr, y_tr, va, y_va, tcfg)
            vs = predict(model, va, DEVICE)
            ts = predict(model, te, DEVICE)
            thr = select_threshold(y_va, vs)
            m = {
                "Model": name, "Seed": seed,
                "PR_AUC": metric_value(y_te, ts, "PR_AUC"),
                "ROC_AUC": metric_value(y_te, ts, "ROC_AUC"),
                "F1": metric_value(y_te, ts, "F1", thr),
                "Precision": metric_value(y_te, ts, "Precision", thr),
                "Recall": metric_value(y_te, ts, "Recall", thr),
                "Accuracy": metric_value(y_te, ts, "Accuracy", thr),
                "Precision@1%": metric_value(y_te, ts, "Precision@1"),
                "Recall@1%": metric_value(y_te, ts, "Recall@1"),
                "Precision@2%": metric_value(y_te, ts, "Precision@2"),
                "Recall@2%": metric_value(y_te, ts, "Recall@2"),
                "Valid_PR_AUC": info["best_valid_pr_auc"],
                "Params": info["n_parameters"],
                "Train_s": info["train_seconds"],
            }
            seed_metrics.append(m)
            per_seed.append(m)
            print(f"  {name:24s} seed={seed:5d}  PR-AUC={m['PR_AUC']:.4f}  F1={m['F1']:.2f}  "
                  f"R@2%={m['Recall@2%']:.1f}  ({info['train_seconds']:.0f}s)", flush=True)
            if seed == seeds[0]:
                save_scores(name, vs, ts)
                lat = inference_latency_ms(model, te, DEVICE)
                m["Infer_ms_per_sample"] = lat
                # latents and the deployed checkpoint come from the selected
                # configuration in rev10, so they are not overwritten here

        frame = pd.DataFrame(seed_metrics)
        agg = {"Model": name, "Seeds": len(seeds)}
        for col in ["PR_AUC", "ROC_AUC", "F1", "Precision@2%", "Recall@2%", "Accuracy"]:
            agg[f"{col}_mean"] = frame[col].mean()
            agg[f"{col}_std"] = frame[col].std(ddof=1)
        agg["Params"] = int(frame["Params"].iloc[0])
        agg["Train_s_mean"] = frame["Train_s"].mean()
        agg["Infer_ms_per_sample"] = seed_metrics[0].get("Infer_ms_per_sample", np.nan)
        rows.append(agg)

    # unsupervised sequence VAE reference
    print("  training unsupervised sequence VAE ...", flush=True)
    vs, ts = train_unsupervised_vae(tr, y_tr, va, te, tr.shape[-1], seed=seeds[0], epochs=args.epochs)
    thr = select_threshold(y_va, vs)
    save_scores("VAE reconstruction (unsupervised)", vs, ts)
    rows.append({
        "Model": "VAE reconstruction (unsupervised)", "Seeds": 1,
        "PR_AUC_mean": metric_value(y_te, ts, "PR_AUC"), "PR_AUC_std": np.nan,
        "ROC_AUC_mean": metric_value(y_te, ts, "ROC_AUC"), "ROC_AUC_std": np.nan,
        "F1_mean": metric_value(y_te, ts, "F1", thr), "F1_std": np.nan,
        "Precision@2%_mean": metric_value(y_te, ts, "Precision@2"), "Precision@2%_std": np.nan,
        "Recall@2%_mean": metric_value(y_te, ts, "Recall@2"), "Recall@2%_std": np.nan,
        "Accuracy_mean": metric_value(y_te, ts, "Accuracy", thr), "Accuracy_std": np.nan,
    })

    pd.DataFrame(per_seed).to_csv(REV_TABLE_DIR / "rev03_sequence_per_seed.csv",
                                  index=False, encoding="utf-8-sig")
    summary = pd.DataFrame(rows).sort_values("PR_AUC_mean", ascending=False)
    summary.to_csv(REV_TABLE_DIR / "rev03_sequence_summary.csv", index=False, encoding="utf-8-sig")
    (REV_TABLE_DIR / "rev03_meta.json").write_text(
        json.dumps({"device": DEVICE, "seeds": seeds, "epochs": args.epochs}, indent=2),
        encoding="utf-8")
    print("\n" + summary.to_string(index=False))


if __name__ == "__main__":
    main()
