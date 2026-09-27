"""Evaluation utilities for the revision: alert-budget metrics, calibration,
cost-sensitive utility and *time-series-aware* (moving block) bootstrap
inference.

Reviewer 1 comment 5 objects to the i.i.d. bootstrap used in the original
submission; every interval-level resampling in this module therefore uses a
moving block bootstrap that preserves the local temporal dependence of the
15-minute operational series.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score,
                             precision_score, recall_score, roc_auc_score)

DEFAULT_BLOCK = 96          # 96 x 15 min = 24 h
DEFAULT_ROUNDS = 2000


# --------------------------------------------------------------------------
# point metrics
# --------------------------------------------------------------------------
def topk_indices(scores: np.ndarray, budget: float) -> np.ndarray:
    k = max(1, int(round(len(scores) * budget)))
    return np.argsort(scores)[::-1][:k]


def recall_at_budget(y: np.ndarray, s: np.ndarray, budget: float) -> float:
    idx = topk_indices(s, budget)
    return float(y[idx].sum() / max(1, y.sum()))


def precision_at_budget(y: np.ndarray, s: np.ndarray, budget: float) -> float:
    idx = topk_indices(s, budget)
    return float(y[idx].sum() / len(idx))


def average_precision(y: np.ndarray, s: np.ndarray) -> float:
    """Average precision, numerically identical to sklearn's
    average_precision_score including its handling of tied scores, but fast enough
    to be called inside a bootstrap loop."""
    order = np.argsort(-s, kind="mergesort")
    s_sorted, y_sorted = s[order], y[order]
    positives = y_sorted.sum()
    if positives == 0:
        return 0.0
    # keep only the last index of each run of equal scores (a distinct threshold)
    cut = np.r_[np.flatnonzero(np.diff(s_sorted)), len(s_sorted) - 1]
    tp = np.cumsum(y_sorted)[cut]
    precision = tp / (cut + 1)
    recall = tp / positives
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def metric_value(y: np.ndarray, s: np.ndarray, metric: str, threshold: float | None = None) -> float:
    if metric == "PR_AUC":
        return average_precision(y, s)
    if metric == "ROC_AUC":
        return float(roc_auc_score(y, s))
    if metric == "Brier":
        return float(brier_score_loss(y, np.clip(s, 0.0, 1.0)))
    if metric.startswith("Recall@"):
        return 100 * recall_at_budget(y, s, float(metric.split("@")[1].rstrip("%")) / 100)
    if metric.startswith("Precision@"):
        return 100 * precision_at_budget(y, s, float(metric.split("@")[1].rstrip("%")) / 100)
    if metric.startswith("Miss@"):
        return 100 * (1 - recall_at_budget(y, s, float(metric.split("@")[1].rstrip("%")) / 100))
    if threshold is None:
        raise ValueError(f"{metric} needs a threshold")
    pred = (s >= threshold).astype(int)
    if metric == "F1":
        return 100 * float(f1_score(y, pred, zero_division=0))
    if metric == "Precision":
        return 100 * float(precision_score(y, pred, zero_division=0))
    if metric == "Recall":
        return 100 * float(recall_score(y, pred, zero_division=0))
    if metric == "Accuracy":
        return 100 * float((pred == y).mean())
    raise ValueError(metric)


# --------------------------------------------------------------------------
# moving block bootstrap
# --------------------------------------------------------------------------
def block_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, max(1, n - block + 1), size=n_blocks)
    idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]
    return np.clip(idx, 0, n - 1)


def make_index_sets(n: int, block: int, rounds: int, seed: int = 42) -> np.ndarray:
    """Pre-draw all bootstrap index sets once so that every model and every metric
    is evaluated on exactly the same resamples."""
    rng = np.random.default_rng(seed)
    return np.stack([block_indices(n, block, rng) for _ in range(rounds)]).astype(np.int32)


def bootstrap_ci(y: np.ndarray, s: np.ndarray, metric: str, threshold: float | None = None,
                 block: int = DEFAULT_BLOCK, rounds: int = DEFAULT_ROUNDS,
                 seed: int = 42, index_sets: np.ndarray | None = None) -> dict:
    if index_sets is None:
        index_sets = make_index_sets(len(y), block, rounds, seed)
    vals = []
    for idx in index_sets:
        yb = y[idx]
        if yb.sum() == 0:
            continue
        vals.append(metric_value(yb, s[idx], metric, threshold))
    arr = np.asarray(vals, dtype=float)
    point = metric_value(y, s, metric, threshold)
    return {
        "value": point,
        "ci_lower": float(np.quantile(arr, 0.025)),
        "ci_upper": float(np.quantile(arr, 0.975)),
        "n_effective": int(arr.size),
    }


def paired_bootstrap_diff(y: np.ndarray, s_a: np.ndarray, s_b: np.ndarray, metric: str,
                          thr_a: float | None = None, thr_b: float | None = None,
                          block: int = DEFAULT_BLOCK, rounds: int = DEFAULT_ROUNDS,
                          seed: int = 42, index_sets: np.ndarray | None = None) -> dict:
    """Paired moving-block bootstrap of metric(A) - metric(B)."""
    if index_sets is None:
        index_sets = make_index_sets(len(y), block, rounds, seed)
    diffs = []
    for idx in index_sets:
        yb = y[idx]
        if yb.sum() == 0:
            continue
        diffs.append(metric_value(yb, s_a[idx], metric, thr_a)
                     - metric_value(yb, s_b[idx], metric, thr_b))
    arr = np.asarray(diffs, dtype=float)
    p_one = float((np.sum(arr <= 0.0) + 1) / (arr.size + 1))
    return {
        "observed_diff": metric_value(y, s_a, metric, thr_a) - metric_value(y, s_b, metric, thr_b),
        "mean_diff": float(arr.mean()),
        "ci_lower": float(np.quantile(arr, 0.025)),
        "ci_upper": float(np.quantile(arr, 0.975)),
        "p_one_sided": p_one,
        "p_two_sided": float(min(1.0, 2 * min(p_one, 1 - p_one))),
        "n_bootstrap": int(arr.size),
        "block_length": block,
    }


# --------------------------------------------------------------------------
# cost-sensitive operational utility
# --------------------------------------------------------------------------
def expected_cost(y: np.ndarray, s: np.ndarray, budget: float, cost_ratio: float) -> float:
    """Cost per test interval in units of one false-alert inspection cost.

    cost_ratio = C_miss / C_alert.  Alerts cost 1 each (operator inspection /
    pre-positioned reserve), a missed extreme event costs `cost_ratio`.
    """
    idx = topk_indices(s, budget)
    flagged = np.zeros(len(y), dtype=bool)
    flagged[idx] = True
    n_alerts = int(flagged.sum())
    n_missed = int(((~flagged) & (y == 1)).sum())
    return (n_alerts + cost_ratio * n_missed) / len(y)


def best_budget_cost(y: np.ndarray, s: np.ndarray, cost_ratio: float,
                     budgets: np.ndarray) -> tuple[float, float]:
    costs = [expected_cost(y, s, b, cost_ratio) for b in budgets]
    j = int(np.argmin(costs))
    return float(budgets[j]), float(costs[j])


def select_threshold(y_valid: np.ndarray, valid_scores: np.ndarray) -> float:
    """Validation F1-optimal threshold, identical in spirit to
    FollowupExperimentRunner.select_best_threshold in the original code."""
    cands = np.unique(np.quantile(valid_scores, np.linspace(0.02, 0.98, 161)))
    best_thr, best_f1 = float(np.median(valid_scores)), -1.0
    for thr in cands:
        f1 = f1_score(y_valid.astype(int), (valid_scores >= thr).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)
    return best_thr
