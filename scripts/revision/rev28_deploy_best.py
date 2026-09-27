"""rev28 - Deploy whichever candidate the validation partition prefers.

Two candidates reached the end of the selection programme: the model kept after
input selection and seed ensembling (rev17 / rev16), and the model chosen by the
greedy architectural search (rev23). The greedy search evaluated each switch
with two seeds and no ensembling, which is a noisier estimate than the final
three-seed ensemble, and the two procedures disagree. The rule is unchanged --
the validation partition decides -- and this script applies it, restores the
winner's scores and regenerates its latent states so that the density analysis
describes the model that is actually deployed.

Usage: python scripts/revision/rev28_deploy_best.py
"""
from __future__ import annotations

import glob
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.revision.rev_common import (REV_TABLE_DIR, build_matrices,  # noqa: E402
                                         save_scores)
from scripts.revision.rev_eval import metric_value  # noqa: E402
from scripts.revision.rev_models import (ModelConfig, ScreeningNet, TrainConfig,  # noqa: E402
                                         _percentile_rank, inference_latency_ms,
                                         predict, set_seed, train)

SCORE_DIR = Path(__file__).resolve().parents[2] / "outputs" / "revision" / "scores"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main() -> None:
    y_va = np.load(SCORE_DIR / "y_valid.npy").astype(int)
    y_te = np.load(SCORE_DIR / "y_test.npy").astype(int)

    seed_files = sorted(glob.glob(str(SCORE_DIR / "aeris_seed*_valid.npy")))
    if not seed_files:
        raise SystemExit("no per-seed scores to compare against")
    vs_list = [np.load(p) for p in seed_files]
    ts_list = [np.load(p.replace("_valid", "_test")) for p in seed_files]
    ens_v = np.mean([_percentile_rank(s) for s in vs_list], axis=0)
    ens_t = np.mean([_percentile_rank(s) for s in ts_list], axis=0)

    candidates = {"input-selected ensemble (rev16/rev17)": (ens_v, ens_t)}
    if (SCORE_DIR / "rev23_valid.npy").exists():
        candidates["greedy architecture search (rev23)"] = (
            np.load(SCORE_DIR / "rev23_valid.npy"), np.load(SCORE_DIR / "rev23_test.npy"))
    scored = {k: float(average_precision_score(y_va, v)) for k, (v, _) in candidates.items()}
    for name, v in scored.items():
        print(f"  {name:42s} validation PR-AUC {v:.4f}  "
              f"test {metric_value(y_te, candidates[name][1], 'PR_AUC'):.4f}")
    winner = max(scored, key=scored.get)
    print(f"  -> deploying: {winner}")
    if winner.startswith("greedy"):
        # the latents, checkpoint and record below describe the rev16/rev17 model
        raise SystemExit("rev23 won on validation: deploy AERIS_final.pt and the rev23 "
                         "latents instead of retraining the rev17 configuration")

    vs, ts = candidates[winner]
    save_scores("AERIS", vs, ts)

    # regenerate the latent states of the deployed model for the density layer
    cfg = json.loads((REV_TABLE_DIR / "rev10_selected_summary.json").read_text(
        encoding="utf-8"))
    selected_inputs = cfg.get("selected_inputs")
    conf = cfg["selected_config"]
    reference = build_matrices(quantile=0.98, horizon=1, seq_len=conf["seq_len"],
                               normalisation="full", threshold_source="full")
    drop = ()
    if selected_inputs:
        keep = set(selected_inputs)
        drop = tuple(c for c in reference[6]["feature_names"] if c not in keep)
    tr, va, te, y_tr, y_va_f, y_te_f, meta = build_matrices(
        quantile=0.98, horizon=1, seq_len=conf["seq_len"], normalisation="full",
        threshold_source="full", clip=conf.get("input_clip"), drop_features=drop)

    mcfg = ModelConfig(encoder="tcn", pooling="attention", use_gate=True, bottleneck="vae",
                       hidden_dim=conf["hidden_dim"], latent_dim=conf["latent_dim"],
                       attention_heads=conf["attention_heads"], dropout=conf["dropout"],
                       head_skip=conf.get("head_skip", False))
    tcfg = TrainConfig(max_epochs=conf["max_epochs"], lr=conf["lr"], device=DEVICE,
                       batch_size=conf["batch_size"], ranking_weight=conf["ranking_weight"],
                       recon_weight=conf["recon_weight"], kl_weight=conf["kl_weight"],
                       aux_weight=conf.get("aux_weight", 1.0),
                       aux_loss=conf.get("aux_loss", "mse"),
                       select_mode=conf.get("score_mode", "rank_blend"))
    set_seed(42)
    model = ScreeningNet(tr.shape[-1], mcfg)
    model, info = train(model, tr, y_tr, va, y_va_f, replace(tcfg, seed=42),
                        aux_tr=meta["aux_train"])
    mode = tcfg.select_mode
    _, lat_tr = predict(model, tr, DEVICE, return_latent=True, mode=mode)
    _, lat_va = predict(model, va, DEVICE, return_latent=True, mode=mode)
    _, lat_te = predict(model, te, DEVICE, return_latent=True, mode=mode)
    np.save(SCORE_DIR / "latent_train_AERIS.npy", lat_tr)
    np.save(SCORE_DIR / "latent_valid_AERIS.npy", lat_va)
    np.save(SCORE_DIR / "latent_test_AERIS.npy", lat_te)
    np.save(SCORE_DIR / "y_train.npy", y_tr)
    torch.save(model.state_dict(), SCORE_DIR / "AERIS_deployed.pt")

    summary = {
        "deployed": winner,
        "architecture": {"encoder": "tcn", "pooling": "attention", "gate": True,
                         "bottleneck": "vae", "summary_branch": False,
                         "head_skip": conf.get("head_skip", False)},
        "hyper_parameters": conf,
        "sequence_inputs": int(tr.shape[-1]),
        "seeds_ensembled": len(seed_files),
        "validation_pr_auc": scored[winner],
        "n_parameters": int(info["n_parameters"]),
        "inference_ms_per_sample": float(inference_latency_ms(model, te, DEVICE)),
        "test": {m: metric_value(y_te, ts, m) for m in
                 ["PR_AUC", "ROC_AUC", "Precision@1", "Recall@1",
                  "Precision@2", "Recall@2"]},
        "candidates_validation_pr_auc": scored,
    }
    (REV_TABLE_DIR / "rev28_deployed.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print("\n" + json.dumps(summary["test"], indent=2))


if __name__ == "__main__":
    main()
