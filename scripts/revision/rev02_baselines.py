"""rev02 - Expanded non-deep baseline suite.

Answers Reviewer 1 comment 4: besides GBM / RF / CART the comparison now
contains a calibrated linear model, a modern histogram boosting machine
(plus XGBoost / LightGBM when installed), three unsupervised anomaly
detectors (Isolation Forest, One-Class SVM, latent-free KDE scoring) and
three purely operational heuristics that any control room could implement
without a model at all.

Every model sees exactly the same chronological split, the same event
definition and the same validation-selected decision threshold.  Test scores
are written to outputs/revision/scores/ so that all later analyses (metrics,
bootstrap inference, cost curves) operate on one common set of predictions.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.revision.rev_common import (REV_TABLE_DIR, build_matrices,  # noqa: E402
                                         save_scores, slug, summarise_sequence)
from scripts.revision.rev_eval import metric_value, select_threshold  # noqa: E402

SCORE_DIR = Path(__file__).resolve().parents[2] / "outputs" / "revision" / "scores"
SCORE_DIR.mkdir(parents=True, exist_ok=True)

SEQ_LEN = 8
SEED = 42


def classical_models():
    from sklearn.ensemble import (GradientBoostingClassifier, HistGradientBoostingClassifier,
                                  RandomForestClassifier)
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier

    models = [
        ("Logistic Regression", LogisticRegression(C=0.1, class_weight="balanced",
                                                   max_iter=1000, random_state=SEED)),
        ("CART", DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=SEED)),
        ("RF", RandomForestClassifier(n_estimators=200, max_depth=8, class_weight="balanced",
                                      random_state=SEED, n_jobs=-1)),
        ("GBM", GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                           random_state=SEED)),
        ("HistGBM", HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05,
                                                   max_depth=6, class_weight="balanced",
                                                   random_state=SEED)),
    ]
    try:
        from xgboost import XGBClassifier
        models.append(("XGBoost", XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, eval_metric="aucpr", tree_method="hist",
            random_state=SEED, n_jobs=-1)))
    except ImportError:
        print("  [skip] xgboost not installed")
    try:
        from lightgbm import LGBMClassifier
        models.append(("LightGBM", LGBMClassifier(
            n_estimators=400, num_leaves=31, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, class_weight="balanced", random_state=SEED,
            n_jobs=-1, verbose=-1)))
    except ImportError:
        print("  [skip] lightgbm not installed")
    return models


def fit_supervised(name, clf, tr, va, te, y_tr, y_va, pos_weight_field=None):
    t0 = time.perf_counter()
    if name == "XGBoost":
        ratio = float((1 - y_tr.mean()) / max(y_tr.mean(), 1e-6))
        clf.set_params(scale_pos_weight=ratio)
    clf.fit(tr, y_tr.astype(int))
    train_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    va_s = clf.predict_proba(va)[:, 1]
    te_s = clf.predict_proba(te)[:, 1]
    infer_ms = 1000 * (time.perf_counter() - t0) / len(te)
    return va_s, te_s, train_s, infer_ms


def anomaly_models(tr, va, te, y_tr):
    """Unsupervised detectors: fitted on the *normal* training intervals only."""
    from sklearn.decomposition import PCA
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import KernelDensity
    from sklearn.svm import OneClassSVM

    rng = np.random.default_rng(SEED)
    normal = tr[y_tr == 0]
    out = {}

    t0 = time.perf_counter()
    iso = IsolationForest(n_estimators=300, contamination=0.02, random_state=SEED, n_jobs=-1)
    iso.fit(normal)
    tr_t = time.perf_counter() - t0
    t0 = time.perf_counter()
    out["Isolation Forest"] = (-iso.score_samples(va), -iso.score_samples(te), tr_t,
                               1000 * (time.perf_counter() - t0) / (len(va) + len(te)))

    sub = normal[rng.choice(len(normal), size=min(4000, len(normal)), replace=False)]
    t0 = time.perf_counter()
    ocs = OneClassSVM(kernel="rbf", nu=0.05, gamma="scale")
    ocs.fit(sub)
    tr_t = time.perf_counter() - t0
    t0 = time.perf_counter()
    out["One-Class SVM"] = (-ocs.decision_function(va), -ocs.decision_function(te), tr_t,
                            1000 * (time.perf_counter() - t0) / (len(va) + len(te)))

    # KDE scoring directly in a PCA-reduced *input* space (no learned latent):
    # this is the ablation that isolates what the learned representation adds.
    pca = PCA(n_components=8, random_state=SEED).fit(normal)
    t0 = time.perf_counter()
    kde = KernelDensity(kernel="gaussian", bandwidth=0.5)
    kde.fit(pca.transform(sub))
    tr_t = time.perf_counter() - t0
    t0 = time.perf_counter()
    out["KDE (input space)"] = (-kde.score_samples(pca.transform(va)),
                                -kde.score_samples(pca.transform(te)), tr_t,
                                1000 * (time.perf_counter() - t0) / (len(va) + len(te)))
    return out


def heuristics(meta, va_slice, te_slice):
    """Model-free control-room rules, ranked by the raw quantity at time t."""
    frame = meta["all_frame"]
    rules = {
        "Heuristic: composite score persistence": "event_proxy_score",
        "Heuristic: renewable ramp": "renewable_ramp_ratio",
        "Heuristic: |system imbalance|": "systemimbalance_abs",
    }
    out = {}
    for name, col in rules.items():
        v = frame[col].to_numpy(dtype=float)
        out[name] = (v[va_slice], v[te_slice], 0.0, 0.0)
    return out


def main() -> None:
    tr_s, va_s, te_s, y_tr, y_va, y_te, meta = build_matrices(
        quantile=0.98, horizon=1, seq_len=SEQ_LEN,
        normalisation="full", threshold_source="full")

    # index bookkeeping so the heuristics are aligned with the sequence windows
    from scripts.revision.rev_common import build_labels, load_feature_table, split_bounds
    raw = load_feature_table()
    labelled, _ = build_labels(raw, 0.98, 1, "full", "full")
    labelled["regulation_deviation_target"] = labelled["systemimbalance_abs"].shift(-1)
    labelled = labelled.dropna().reset_index(drop=True)
    n = len(labelled)
    tr_end, va_end = split_bounds(n)
    meta["all_frame"] = labelled
    va_slice = slice(tr_end + SEQ_LEN - 1, va_end)
    te_slice = slice(va_end + SEQ_LEN - 1, n)

    tr_f = summarise_sequence(tr_s)
    va_f = summarise_sequence(va_s)
    te_f = summarise_sequence(te_s)
    print(f"flat features: {tr_f.shape[1]}  train={tr_f.shape[0]}  valid={va_f.shape[0]}  test={te_f.shape[0]}")
    print(f"test positives: {int(y_te.sum())} ({100 * y_te.mean():.3f}%)")

    np.save(SCORE_DIR / "y_test.npy", y_te)
    np.save(SCORE_DIR / "y_valid.npy", y_va)

    results = []
    collected: dict[str, tuple] = {}
    timing_path = REV_TABLE_DIR / "rev02_timing.json"
    timing: dict[str, list] = json.loads(timing_path.read_text()) if timing_path.exists() else {}

    def cached(name: str):
        tag = slug(name)
        f_te, f_va = SCORE_DIR / f"test__{tag}.npy", SCORE_DIR / f"valid__{tag}.npy"
        if f_te.exists() and f_va.exists() and name in timing:
            print(f"  [cached] {name}")
            return np.load(f_va), np.load(f_te), *timing[name]
        return None

    for name, clf in classical_models():
        hit = cached(name)
        if hit is not None:
            collected[name] = hit
            continue
        print(f"  fitting {name} ...", flush=True)
        collected[name] = fit_supervised(name, clf, tr_f, va_f, te_f, y_tr, y_va)

    anomaly_names = ["Isolation Forest", "One-Class SVM", "KDE (input space)"]
    if all(cached(n) is not None for n in anomaly_names):
        for n in anomaly_names:
            collected[n] = cached(n)
    else:
        print("  fitting anomaly detectors ...", flush=True)
        collected.update(anomaly_models(tr_f, va_f, te_f, y_tr))
    collected.update(heuristics(meta, va_slice, te_slice))

    for name, (vs, ts, train_s, infer_ms) in collected.items():
        timing[name] = [float(train_s), float(infer_ms)]
        thr = select_threshold(y_va, vs)
        row = {
            "Model": name,
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
            "Train_s": train_s,
            "Infer_ms_per_sample": infer_ms,
            "Threshold": thr,
        }
        results.append(row)
        save_scores(name, vs, ts)
        print(f"    {name:38s} PR-AUC={row['PR_AUC']:.4f}  F1={row['F1']:.2f}  R@2%={row['Recall@2%']:.1f}")

    timing_path.write_text(json.dumps(timing, indent=2), encoding="utf-8")
    df = pd.DataFrame(results).sort_values("PR_AUC", ascending=False)
    df.to_csv(REV_TABLE_DIR / "rev02_baseline_suite.csv", index=False, encoding="utf-8-sig")
    (REV_TABLE_DIR / "rev02_meta.json").write_text(json.dumps({
        "n_features": meta["n_features"], "seq_len": SEQ_LEN,
        "test_positives": int(y_te.sum()), "test_size": int(len(y_te)),
    }, indent=2), encoding="utf-8")
    print("\n" + df.to_string(index=False))


if __name__ == "__main__":
    main()
