"""rev05 - Robustness, leakage audit, horizon extension and rolling-origin evaluation.

Answers Reviewer 1 comments 1 (threshold robustness), 2 (information
availability and longer horizons) and 5 (rolling-origin evaluation).

Parts
  A  screening horizons t+1, t+2, t+4, t+8
  B  event thresholds q in {0.95, 0.975, 0.98, 0.99}
  C  information-availability audit: removing the composite score and its
     constituents from the inputs, applying a one-interval publication lag to
     every input, and rebuilding the label with training-only normalisation
  D  rolling-origin evaluation over four expanding-window folds
  E  input perturbation (measurement noise, renewable-scale change)

Usage: python scripts/revision/rev05_robustness.py [--parts ABCDE] [--seeds 42,7]
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.revision.rev_common import (EQUAL_WEIGHTS, REV_TABLE_DIR,  # noqa: E402
                                         build_matrices, build_matrices_fold,
                                         summarise_sequence)
from scripts.revision.rev_eval import metric_value, select_threshold  # noqa: E402
from scripts.revision.rev_models import (ModelConfig, ScreeningNet, TrainConfig,  # noqa: E402
                                         predict, set_seed, train)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Every robustness experiment uses the deployed configuration, so that the audits
# describe the model the paper actually reports.
from scripts.revision.rev04_ablation import (BASE_CLIP, BASE_M, BASE_MODE,  # noqa: E402
                                             BASE_SEQ, BASE_T)
SCORE_COLS = ("event_proxy_score", "intermittency_index_zscore", "renewable_ramp_ratio_zscore",
              "renewable_recent_error_ratio_zscore", "systemimbalance_abs_zscore")
CONSTITUENTS = ("intermittency_index", "renewable_ramp_ratio",
                "renewable_recent_error_ratio", "systemimbalance_abs")


def fit_aeris(tr, va, te, y_tr, y_va, y_te, seeds, mcfg=None, tcfg=None, aux_tr=None):
    out = []
    for seed in seeds:
        set_seed(seed)
        model = ScreeningNet(tr.shape[-1], mcfg or BASE_M)
        model, _ = train(model, tr, y_tr, va, y_va, replace(tcfg or BASE_T, seed=seed),
                         aux_tr=aux_tr)
        vs = predict(model, va, DEVICE, mode=BASE_MODE)
        ts = predict(model, te, DEVICE, mode=BASE_MODE)
        thr = select_threshold(y_va, vs)
        out.append({
            "PR_AUC": metric_value(y_te, ts, "PR_AUC"),
            "F1": metric_value(y_te, ts, "F1", thr),
            "Recall@2%": metric_value(y_te, ts, "Recall@2"),
            "Precision@2%": metric_value(y_te, ts, "Precision@2"),
            "scores": ts, "model": model, "valid_scores": vs,
        })
    return out


def fit_gbm(tr, va, te, y_tr, y_va, y_te, kind="hist"):
    """Strong tabular reference. `kind='hist'` is the histogram boosting machine that
    turned out to be the strongest non-sequential competitor; `kind='gbm'` is the
    gradient boosting machine used in the first version of this work."""
    from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
    if kind == "hist":
        clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_depth=6,
                                             class_weight="balanced", random_state=42)
    else:
        clf = GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                         random_state=42)
    trf, vaf, tef = summarise_sequence(tr), summarise_sequence(va), summarise_sequence(te)
    clf.fit(trf, y_tr.astype(int))
    vs, ts = clf.predict_proba(vaf)[:, 1], clf.predict_proba(tef)[:, 1]
    thr = select_threshold(y_va, vs)
    return {
        "PR_AUC": metric_value(y_te, ts, "PR_AUC"),
        "F1": metric_value(y_te, ts, "F1", thr),
        "Recall@2%": metric_value(y_te, ts, "Recall@2"),
        "Precision@2%": metric_value(y_te, ts, "Precision@2"),
    }


def heuristic_reference(meta, y_te, lag: int = 0):
    """Persistence of the composite score, aligned with the test windows.
    `lag` reproduces the publication-delay audit for the model-free rule as well."""
    frame = meta["test_frame"]
    s = frame["event_proxy_score"].shift(lag).bfill().to_numpy(dtype=float)
    return {
        "PR_AUC": metric_value(y_te, s, "PR_AUC"),
        "F1": np.nan,
        "Recall@2%": metric_value(y_te, s, "Recall@2"),
        "Precision@2%": metric_value(y_te, s, "Precision@2"),
    }


def fit_lstm(tr, va, te, y_tr, y_va, y_te, seed=42):
    cfg = ModelConfig(encoder="lstm", pooling="last", use_gate=False, bottleneck="none")
    set_seed(seed)
    model = ScreeningNet(tr.shape[-1], cfg)
    model, _ = train(model, tr, y_tr, va, y_va, replace(BASE_T, seed=seed))
    vs, ts = predict(model, va, DEVICE), predict(model, te, DEVICE)
    thr = select_threshold(y_va, vs)
    return {
        "PR_AUC": metric_value(y_te, ts, "PR_AUC"),
        "F1": metric_value(y_te, ts, "F1", thr),
        "Recall@2%": metric_value(y_te, ts, "Recall@2"),
        "Precision@2%": metric_value(y_te, ts, "Precision@2"),
    }


def summarise(runs, **extra):
    row = dict(extra)
    for key in ["PR_AUC", "F1", "Recall@2%", "Precision@2%"]:
        vals = [r[key] for r in runs]
        row[key] = float(np.mean(vals))
        row[f"{key}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan
    return row


def part_a(seeds, rows):
    for horizon in (1, 2, 4, 8):
        data = build_matrices(quantile=0.98, horizon=horizon, seq_len=BASE_SEQ,
                              normalisation="full", threshold_source="full", clip=BASE_CLIP)
        tr, va, te, y_tr, y_va, y_te, _ = data
        runs = fit_aeris(tr, va, te, y_tr, y_va, y_te, seeds, aux_tr=data[6]["aux_train"])
        rows.append(summarise(runs, Part="A_horizon", Setting=f"t+{horizon} ({15*horizon} min)",
                              Model="AERIS", Positives=int(y_te.sum())))
        rows.append({"Part": "A_horizon", "Setting": f"t+{horizon} ({15*horizon} min)",
                     "Model": "HistGBM", "Positives": int(y_te.sum()),
                     **fit_gbm(tr, va, te, y_tr, y_va, y_te)})
        rows.append({"Part": "A_horizon", "Setting": f"t+{horizon} ({15*horizon} min)",
                     "Model": "Persistence heuristic", "Positives": int(y_te.sum()),
                     **heuristic_reference(data[6], y_te)})
        rows.append({"Part": "A_horizon", "Setting": f"t+{horizon} ({15*horizon} min)",
                     "Model": "LSTM classifier", "Positives": int(y_te.sum()),
                     **fit_lstm(tr, va, te, y_tr, y_va, y_te)})
        print(f"  [A] horizon t+{horizon} done", flush=True)


def part_b(seeds, rows):
    for q in (0.95, 0.975, 0.98, 0.99):
        tr, va, te, y_tr, y_va, y_te, meta_b = build_matrices(
            quantile=q, horizon=1, seq_len=BASE_SEQ, normalisation="full",
            threshold_source="full", clip=BASE_CLIP)
        runs = fit_aeris(tr, va, te, y_tr, y_va, y_te, seeds, aux_tr=meta_b["aux_train"])
        rows.append(summarise(runs, Part="B_threshold", Setting=f"q={q}", Model="AERIS",
                              Positives=int(y_te.sum()),
                              PositiveRate=100 * float(y_te.mean())))
        rows.append({"Part": "B_threshold", "Setting": f"q={q}", "Model": "HistGBM",
                     "Positives": int(y_te.sum()), "PositiveRate": 100 * float(y_te.mean()),
                     **fit_gbm(tr, va, te, y_tr, y_va, y_te)})
        rows.append({"Part": "B_threshold", "Setting": f"q={q}", "Model": "Persistence heuristic",
                     "Positives": int(y_te.sum()), "PositiveRate": 100 * float(y_te.mean()),
                     **heuristic_reference(meta_b, y_te)})
        print(f"  [B] quantile {q} done", flush=True)


def part_c(seeds, rows):
    variants = [
        ("Canonical inputs (146 features)", dict(drop_features=(), lag_features=0,
                                                 normalisation="full", threshold_source="full")),
        ("Composite score and z-scores removed (141)",
         dict(drop_features=SCORE_COLS, lag_features=0,
              normalisation="full", threshold_source="full")),
        ("Composite score, z-scores and raw constituents removed (137)",
         dict(drop_features=SCORE_COLS + CONSTITUENTS, lag_features=0,
              normalisation="full", threshold_source="full")),
        ("One-interval publication lag on all inputs",
         dict(drop_features=(), lag_features=1,
              normalisation="full", threshold_source="full")),
        ("Training-only label normalisation and threshold",
         dict(drop_features=(), lag_features=0,
              normalisation="train", threshold_source="train")),
        ("Training-only label + publication lag",
         dict(drop_features=(), lag_features=1,
              normalisation="train", threshold_source="train")),
        ("Equal weights in the composite score",
         dict(drop_features=(), lag_features=0,
              normalisation="full", threshold_source="full", weights=EQUAL_WEIGHTS)),
    ]
    for label, kwargs in variants:
        tr, va, te, y_tr, y_va, y_te, meta = build_matrices(quantile=0.98, horizon=1,
                                                            seq_len=BASE_SEQ, clip=BASE_CLIP, **kwargs)
        runs = fit_aeris(tr, va, te, y_tr, y_va, y_te, seeds, aux_tr=meta["aux_train"])
        rows.append(summarise(runs, Part="C_availability", Setting=label, Model="AERIS",
                              Features=meta["n_features"], Positives=int(y_te.sum())))
        rows.append({"Part": "C_availability", "Setting": label, "Model": "HistGBM",
                     "Features": meta["n_features"], "Positives": int(y_te.sum()),
                     **fit_gbm(tr, va, te, y_tr, y_va, y_te)})
        rows.append({"Part": "C_availability", "Setting": label, "Model": "Persistence heuristic",
                     "Features": 1, "Positives": int(y_te.sum()),
                     **heuristic_reference(meta, y_te, lag=kwargs.get("lag_features", 0))})
        print(f"  [C] {label} done", flush=True)


def part_d(seeds, rows):
    folds = [(0.50, 0.60, 0.70), (0.60, 0.70, 0.80), (0.70, 0.80, 0.90), (0.80, 0.90, 1.00)]
    for k, (a, b, c) in enumerate(folds, start=1):
        tr, va, te, y_tr, y_va, y_te, meta = build_matrices_fold(a, b, c, quantile=0.98, seq_len=BASE_SEQ)
        if y_te.sum() == 0:
            continue
        runs = fit_aeris(tr, va, te, y_tr, y_va, y_te, seeds, aux_tr=meta.get("aux_train"))
        label = f"Fold {k}: test {meta['test_start'][:10]} to {meta['test_end'][:10]}"
        rows.append(summarise(runs, Part="D_rolling", Setting=label, Model="AERIS",
                              Positives=int(y_te.sum())))
        rows.append({"Part": "D_rolling", "Setting": label, "Model": "HistGBM",
                     "Positives": int(y_te.sum()),
                     **fit_gbm(tr, va, te, y_tr, y_va, y_te)})
        print(f"  [D] {label} done", flush=True)


def part_e(seeds, rows):
    tr, va, te, y_tr, y_va, y_te, meta = build_matrices(
        quantile=0.98, horizon=1, seq_len=BASE_SEQ, normalisation="full",
        threshold_source="full", clip=BASE_CLIP)
    names = meta["feature_names"]
    renewable_idx = [i for i, c in enumerate(names)
                     if c.startswith(("wind_", "solar_", "renewable_"))]
    runs = fit_aeris(tr, va, te, y_tr, y_va, y_te, seeds[:1], aux_tr=meta["aux_train"])
    model, vs = runs[0]["model"], runs[0]["valid_scores"]
    thr = select_threshold(y_va, vs)
    rng = np.random.default_rng(42)

    def evaluate(x, label):
        ts = predict(model, x, DEVICE, mode=BASE_MODE)
        rows.append({"Part": "E_perturbation", "Setting": label, "Model": "AERIS",
                     "PR_AUC": metric_value(y_te, ts, "PR_AUC"),
                     "F1": metric_value(y_te, ts, "F1", thr),
                     "Recall@2%": metric_value(y_te, ts, "Recall@2"),
                     "Precision@2%": metric_value(y_te, ts, "Precision@2")})

    evaluate(te, "Clean test set")
    for snr in (20, 15, 10):
        power = np.mean(te ** 2)
        noise = rng.normal(0, np.sqrt(power / 10 ** (snr / 10)), te.shape).astype(np.float32)
        evaluate(te + noise, f"Additive noise at {snr} dB SNR")
    for scale in (1.10, 1.20, 0.90):
        pert = te.copy()
        pert[:, :, renewable_idx] *= scale
        evaluate(pert, f"Renewable inputs scaled by {scale:.2f}")
    print("  [E] perturbation done", flush=True)


def part_f(seeds, rows):
    """Feature-group ablation: which information source carries the signal."""
    from scripts.revision.rev_common import build_matrices as bm
    *_, meta = bm(quantile=0.98, horizon=1, seq_len=BASE_SEQ,
                  normalisation="full", threshold_source="full", clip=BASE_CLIP)
    names = meta["feature_names"]
    groups = {
        "All inputs": names,
        "Wind only": [c for c in names if c.startswith("wind_")],
        "Solar only": [c for c in names if c.startswith("solar_")],
        "Balancing and market only": [c for c in names if c.startswith(
            ("ace", "systemimbalance", "alpha", "marginal", "imbalance", "price_"))],
        "Four label constituents only": list(CONSTITUENTS),
    }
    for label, keep in groups.items():
        drop = tuple(c for c in names if c not in keep)
        tr, va, te, y_tr, y_va, y_te, m2 = build_matrices(
            quantile=0.98, horizon=1, seq_len=BASE_SEQ, normalisation="full",
            threshold_source="full", drop_features=drop, clip=BASE_CLIP)
        runs = fit_aeris(tr, va, te, y_tr, y_va, y_te, seeds[:1], aux_tr=m2["aux_train"])
        rows.append(summarise(runs, Part="F_feature_group", Setting=label, Model="AERIS",
                              Features=m2["n_features"]))
        print(f"  [F] {label} ({m2['n_features']} inputs) done", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="ABCDEF")
    ap.add_argument("--seeds", default="42,7")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    rows: list[dict] = []
    dispatch = {"A": part_a, "B": part_b, "C": part_c, "D": part_d, "E": part_e, "F": part_f}
    for key in args.parts:
        print(f"=== part {key} ===", flush=True)
        dispatch[key](seeds, rows)
        pd.DataFrame(rows).to_csv(REV_TABLE_DIR / "rev05_robustness.csv",
                                  index=False, encoding="utf-8-sig")

    frame = pd.DataFrame(rows)
    print("\n" + frame.drop(columns=[c for c in frame.columns if c.endswith("_std")]).to_string(index=False))


if __name__ == "__main__":
    main()
