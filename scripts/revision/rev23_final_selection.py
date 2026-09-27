"""rev23 - Final architecture selection on the validation partition.

The component ablation (rev04) suggested that several switches of the submitted
architecture do not earn their place: removing the variational latent, replacing
attention pooling by a mean, and making the temporal blocks strictly causal all
scored at least as well as the full model. Those were test-set numbers and
cannot be used to choose anything. This script turns them into a proper
selection: a greedy coordinate search over the architectural switches, scored
only on the validation partition, seeds averaged.

The bottleneck is restricted to a low-dimensional latent (variational or
deterministic) so that the density-based confidence layer remains available
whatever the search decides.

Usage: python scripts/revision/rev23_final_selection.py [--seeds 42,7]
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
                                         summarise_sequence)
from scripts.revision.rev_eval import metric_value, select_threshold  # noqa: E402
from scripts.revision.rev_models import (ModelConfig, ScreeningNet, TrainConfig,  # noqa: E402
                                         _percentile_rank, inference_latency_ms,
                                         predict, set_seed, train)

SCORE_DIR = Path(__file__).resolve().parents[2] / "outputs" / "revision" / "scores"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# switches explored one at a time, in this order
COORDINATES = [
    ("bottleneck", ["vae", "ae"]),
    ("pooling", ["attention", "mean"]),
    ("encoder", ["tcn", "causal_tcn"]),
    ("use_gate", [True, False]),
    ("summary_branch", [True, False]),
]


def rank_mean(scores):
    return np.mean([_percentile_rank(np.asarray(s, float)) for s in scores], axis=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,7")
    ap.add_argument("--final-seeds", default="42,7,13")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    final_seeds = [int(s) for s in args.final_seeds.split(",")]

    prior = json.loads((REV_TABLE_DIR / "rev10_selected_summary.json").read_text(
        encoding="utf-8"))
    cfg = prior["selected_config"]
    selected_inputs = prior.get("selected_inputs")

    reference = build_matrices(quantile=0.98, horizon=1, seq_len=cfg["seq_len"],
                               normalisation="full", threshold_source="full")
    drop = ()
    if selected_inputs:
        keep = set(selected_inputs)
        drop = tuple(c for c in reference[6]["feature_names"] if c not in keep)

    built = build_matrices(quantile=0.98, horizon=1, seq_len=cfg["seq_len"],
                           normalisation="full", threshold_source="full",
                           clip=cfg.get("input_clip"), drop_features=drop)
    tr, va, te, y_tr, y_va, y_te, meta = built
    # the summary branch always sees the full input set, exactly as the boosting
    # baselines do, so the two families get the same view of the window
    full = build_matrices(quantile=0.98, horizon=1, seq_len=cfg["seq_len"],
                          normalisation="full", threshold_source="full",
                          clip=cfg.get("input_clip"))
    s_tr, s_va, s_te = (summarise_sequence(x) for x in (full[0], full[1], full[2]))
    y_va_i = y_va.astype(int)
    print(f"  sequence branch: {tr.shape[-1]} inputs   summary branch: {s_tr.shape[1]} columns",
          flush=True)

    state = {"bottleneck": "vae", "pooling": "attention", "encoder": "tcn",
             "use_gate": True, "summary_branch": True}

    def evaluate(switches: dict, seed_list):
        mcfg = ModelConfig(
            encoder=switches["encoder"], pooling=switches["pooling"],
            use_gate=switches["use_gate"], bottleneck=switches["bottleneck"],
            hidden_dim=cfg["hidden_dim"], latent_dim=cfg["latent_dim"],
            attention_heads=cfg["attention_heads"], dropout=cfg["dropout"],
            head_skip=cfg.get("head_skip", False),
            summary_dim=s_tr.shape[1] if switches["summary_branch"] else 0)
        tcfg = TrainConfig(max_epochs=cfg["max_epochs"], lr=cfg["lr"], device=DEVICE,
                           batch_size=cfg["batch_size"], ranking_weight=cfg["ranking_weight"],
                           recon_weight=cfg["recon_weight"], kl_weight=cfg["kl_weight"],
                           aux_weight=cfg.get("aux_weight", 1.0),
                           aux_loss=cfg.get("aux_loss", "mse"),
                           select_mode=cfg.get("score_mode", "rank_blend"))
        use_summary = switches["summary_branch"]
        vs_list, ts_list, models = [], [], []
        for seed in seed_list:
            set_seed(seed)
            model = ScreeningNet(tr.shape[-1], mcfg)
            model, info = train(model, tr, y_tr, va, y_va, replace(tcfg, seed=seed),
                                aux_tr=meta["aux_train"],
                                summary_tr=s_tr if use_summary else None,
                                summary_va=s_va if use_summary else None)
            mode = tcfg.select_mode
            vs_list.append(predict(model, va, DEVICE, mode=mode,
                                   summary=s_va if use_summary else None))
            ts_list.append(predict(model, te, DEVICE, mode=mode,
                                   summary=s_te if use_summary else None))
            models.append((model, info))
        v = float(np.mean([average_precision_score(y_va_i, s) for s in vs_list]))
        return v, vs_list, ts_list, models, mcfg, tcfg, use_summary

    history = []
    current_v, *_ = evaluate(state, seeds)
    print(f"  start {state} -> validation {current_v:.4f}", flush=True)
    history.append({"Step": "start", **state, "Valid PR_AUC": current_v})

    for name, options in COORDINATES:
        best_option, best_v = state[name], current_v
        for option in options:
            if option == state[name]:
                continue
            trial = dict(state)
            trial[name] = option
            v, *_ = evaluate(trial, seeds)
            history.append({"Step": f"{name}={option}", **trial, "Valid PR_AUC": v})
            print(f"    {name}={option}: validation {v:.4f}"
                  f"{'  <- keep' if v > best_v else ''}", flush=True)
            if v > best_v:
                best_option, best_v = option, v
        state[name] = best_option
        current_v = best_v
        pd.DataFrame(history).to_csv(REV_TABLE_DIR / "rev23_selection_path.csv",
                                     index=False, encoding="utf-8-sig")

    print(f"\n  selected architecture: {state}  (validation {current_v:.4f})", flush=True)

    v, vs_list, ts_list, models, mcfg, tcfg, use_summary = evaluate(state, final_seeds)
    single = int(np.argmax([average_precision_score(y_va_i, s) for s in vs_list]))
    v_single = float(average_precision_score(y_va_i, vs_list[single]))
    v_ens = float(average_precision_score(y_va_i, rank_mean(vs_list)))
    use_ens = v_ens > v_single
    vs_final = rank_mean(vs_list) if use_ens else vs_list[single]
    ts_final = rank_mean(ts_list) if use_ens else ts_list[single]
    print(f"  validation: single {v_single:.4f} / ensemble {v_ens:.4f} -> "
          f"{'ensemble' if use_ens else 'single'}", flush=True)

    thr = select_threshold(y_va, vs_final)
    per_seed = pd.DataFrame([{
        "Seed": s,
        "PR_AUC": metric_value(y_te, t, "PR_AUC"),
        "Recall@2%": metric_value(y_te, t, "Recall@2"),
        "Precision@2%": metric_value(y_te, t, "Precision@2"),
    } for s, t in zip(final_seeds, ts_list)])
    per_seed.to_csv(REV_TABLE_DIR / "rev23_final_per_seed.csv",
                    index=False, encoding="utf-8-sig")

    model, info = models[0]
    _, lat_tr = predict(model, tr, DEVICE, return_latent=True, mode=tcfg.select_mode,
                        summary=s_tr if use_summary else None)
    _, lat_va = predict(model, va, DEVICE, return_latent=True, mode=tcfg.select_mode,
                        summary=s_va if use_summary else None)
    _, lat_te = predict(model, te, DEVICE, return_latent=True, mode=tcfg.select_mode,
                        summary=s_te if use_summary else None)
    # this is a candidate, not the deployed model: it is kept in its own files and
    # rev28 decides on validation whether it replaces the deployed AERIS scores
    np.save(SCORE_DIR / "rev23_latent_train.npy", lat_tr)
    np.save(SCORE_DIR / "rev23_latent_valid.npy", lat_va)
    np.save(SCORE_DIR / "rev23_latent_test.npy", lat_te)
    np.save(SCORE_DIR / "rev23_valid.npy", vs_final)
    np.save(SCORE_DIR / "rev23_test.npy", ts_final)
    torch.save(model.state_dict(), SCORE_DIR / "AERIS_final.pt")

    summary = {
        "architecture": state,
        "hyper_parameters": cfg,
        "sequence_inputs": int(tr.shape[-1]),
        "summary_columns": int(s_tr.shape[1]) if use_summary else 0,
        "validation_pr_auc": current_v,
        "final_validation_pr_auc": v_ens if use_ens else v_single,
        "deployment": "ensemble" if use_ens else "single",
        "n_parameters": int(info["n_parameters"]),
        "inference_ms_per_sample": float(inference_latency_ms(
            model, te, DEVICE, summary=s_te if use_summary else None)),
        "test_deployed": {
            "PR_AUC": metric_value(y_te, ts_final, "PR_AUC"),
            "ROC_AUC": metric_value(y_te, ts_final, "ROC_AUC"),
            "F1": metric_value(y_te, ts_final, "F1", thr),
            "Precision@1%": metric_value(y_te, ts_final, "Precision@1"),
            "Recall@1%": metric_value(y_te, ts_final, "Recall@1"),
            "Precision@2%": metric_value(y_te, ts_final, "Precision@2"),
            "Recall@2%": metric_value(y_te, ts_final, "Recall@2"),
        },
        "test_per_seed_mean_std": {
            c: [float(per_seed[c].mean()), float(per_seed[c].std(ddof=1))]
            for c in ["PR_AUC", "Recall@2%", "Precision@2%"]},
    }
    (REV_TABLE_DIR / "rev23_final_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print("\n" + json.dumps(summary["test_deployed"], indent=2))


if __name__ == "__main__":
    main()
