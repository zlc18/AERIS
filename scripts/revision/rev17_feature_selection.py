"""rev17 - Input selection for the sequence model, decided on validation.

Boosting machines select features implicitly; a neural network does not, and
with 146 inputs and only about 500 positive training windows the proposed model
may be carrying inputs that only add variance. This script ranks the inputs by
gain importance computed on the *training* partition alone, retrains the
deployed configuration on the top-k inputs for several k, and keeps the value of
k that wins on validation. If the full input set wins, nothing changes.

Usage: python scripts/revision/rev17_feature_selection.py [--seeds 42,7,13]
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
                                         inference_latency_ms, predict, set_seed, train)

SCORE_DIR = Path(__file__).resolve().parents[2] / "outputs" / "revision" / "scores"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOP_K = [32, 64, 96]


def importance_ranking(trf: np.ndarray, a_tr: np.ndarray, names: list[str]) -> list[str]:
    """Gain importance per input, aggregated over the mean/last/max blocks.
    Fitted on the training partition only."""
    n = len(names)
    try:
        from lightgbm import LGBMRegressor
        model = LGBMRegressor(n_estimators=300, num_leaves=31, learning_rate=0.05,
                              random_state=42, n_jobs=-1, verbose=-1)
        model.fit(trf, a_tr)
        gain = np.asarray(model.booster_.feature_importance(importance_type="gain"),
                          dtype=float)
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.inspection import permutation_importance
        model = HistGradientBoostingRegressor(max_iter=200, random_state=42).fit(trf, a_tr)
        sub = np.random.default_rng(42).choice(len(trf), size=min(4000, len(trf)), replace=False)
        gain = permutation_importance(model, trf[sub], a_tr[sub], n_repeats=3,
                                      random_state=42).importances_mean
    per_input = gain.reshape(3, n).sum(axis=0)
    order = np.argsort(per_input)[::-1]
    return [names[i] for i in order]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,7,13")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    summary_path = REV_TABLE_DIR / "rev10_selected_summary.json"
    if not summary_path.exists():
        raise SystemExit("run the AERIS search first")
    cfg = json.loads(summary_path.read_text(encoding="utf-8"))["selected_config"]
    # the input-selected networks are trained without the head skip connection,
    # whatever rev10c chose on the full input set; the record below states this
    mcfg = ModelConfig(encoder="tcn", pooling="attention", use_gate=True, bottleneck="vae",
                       hidden_dim=cfg["hidden_dim"], latent_dim=cfg["latent_dim"],
                       attention_heads=cfg["attention_heads"], dropout=cfg["dropout"],
                       head_skip=False)
    tcfg = TrainConfig(max_epochs=cfg["max_epochs"], lr=cfg["lr"], device=DEVICE,
                       batch_size=cfg["batch_size"], ranking_weight=cfg["ranking_weight"],
                       recon_weight=cfg["recon_weight"], kl_weight=cfg["kl_weight"],
                       aux_weight=cfg.get("aux_weight", 0.0),
                       aux_loss=cfg.get("aux_loss", "mse"),
                       select_mode=cfg.get("score_mode", "cls"))
    mode = cfg.get("score_mode", "cls")
    seq_len, clip = cfg["seq_len"], cfg.get("input_clip")

    full = build_matrices(quantile=0.98, horizon=1, seq_len=seq_len,
                          normalisation="full", threshold_source="full", clip=clip)
    names = full[6]["feature_names"]
    ranking = importance_ranking(summarise_sequence(full[0]), full[6]["aux_train"], names)
    pd.DataFrame({"Rank": np.arange(1, len(ranking) + 1), "Input": ranking}).to_csv(
        REV_TABLE_DIR / "rev17_input_ranking.csv", index=False, encoding="utf-8-sig")
    print(f"  top inputs: {', '.join(ranking[:8])}", flush=True)

    rows, cache = [], {len(names): full}

    def evaluate(k: int, seed: int):
        if k not in cache:
            drop = tuple(c for c in names if c not in set(ranking[:k]))
            cache[k] = build_matrices(quantile=0.98, horizon=1, seq_len=seq_len,
                                      normalisation="full", threshold_source="full",
                                      clip=clip, drop_features=drop)
        tr, va, te, y_tr, y_va, y_te, meta = cache[k]
        set_seed(seed)
        model = ScreeningNet(tr.shape[-1], mcfg)
        model, info = train(model, tr, y_tr, va, y_va, replace(tcfg, seed=seed),
                            aux_tr=meta["aux_train"])
        vs = predict(model, va, DEVICE, mode=mode)
        ts = predict(model, te, DEVICE, mode=mode)
        return model, info, vs, ts, y_va, y_te

    for k in TOP_K + [len(names)]:
        _, info, vs, ts, y_va, y_te = evaluate(k, seeds[0])
        vpr = float(average_precision_score(y_va.astype(int), vs))
        rows.append({"Inputs": k, "Valid PR_AUC": vpr,
                     "Test PR_AUC": metric_value(y_te, ts, "PR_AUC"),
                     "Test Recall@2%": metric_value(y_te, ts, "Recall@2"),
                     "Parameters": info["n_parameters"],
                     "Train_s": info["train_seconds"]})
        print(f"  top-{k:<4d} inputs: validation PR-AUC {vpr:.4f}  "
              f"(test {rows[-1]['Test PR_AUC']:.4f})", flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(REV_TABLE_DIR / "rev17_input_selection.csv", index=False, encoding="utf-8-sig")
    best_k = int(frame.sort_values("Valid PR_AUC", ascending=False).iloc[0]["Inputs"])
    print(f"  selected input set: top-{best_k}")

    if best_k == len(names):
        print("  the full input set wins on validation; deployed scores unchanged")
        return

    seed_rows = []
    for seed in seeds:
        model, info, vs, ts, y_va, y_te = evaluate(best_k, seed)
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
        np.save(SCORE_DIR / f"aeris_seed{seed}_valid.npy", vs)
        np.save(SCORE_DIR / f"aeris_seed{seed}_test.npy", ts)
        print(f"  seed {seed:5d}  test PR-AUC={seed_rows[-1]['PR_AUC']:.4f}", flush=True)
        if seed == seeds[0]:
            save_scores("AERIS", vs, ts)
            tr, va, te, *_ , meta = cache[best_k]
            _, lat_tr = predict(model, tr, DEVICE, return_latent=True, mode=mode)
            _, lat_va = predict(model, va, DEVICE, return_latent=True, mode=mode)
            _, lat_te = predict(model, te, DEVICE, return_latent=True, mode=mode)
            np.save(SCORE_DIR / "latent_train_AERIS.npy", lat_tr)
            np.save(SCORE_DIR / "latent_valid_AERIS.npy", lat_va)
            np.save(SCORE_DIR / "latent_test_AERIS.npy", lat_te)
            torch.save(model.state_dict(), SCORE_DIR / "AERIS_selected.pt")
            latency = inference_latency_ms(model, te, DEVICE)

    detail = pd.DataFrame(seed_rows)
    detail.to_csv(REV_TABLE_DIR / "rev10_selected_per_seed.csv", index=False, encoding="utf-8-sig")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["selected_config"]["n_inputs"] = best_k
    summary["selected_config"]["head_skip"] = False
    summary["selected_inputs"] = ranking[:best_k]
    summary["inference_ms_per_sample"] = float(latency)
    summary["n_parameters"] = int(detail["Params"].iloc[0])
    summary["test"] = {c: [float(detail[c].mean()), float(detail[c].std(ddof=1))]
                       for c in ["PR_AUC", "ROC_AUC", "F1", "Precision@1%", "Recall@1%",
                                 "Precision@2%", "Recall@2%", "Accuracy"]}
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n" + detail.to_string(index=False))


if __name__ == "__main__":
    main()
