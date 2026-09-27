"""Shared helpers for the 2026 revision experiments.

Everything in this module is deliberately written so that the label
construction, the chronological split and the feature bookkeeping are
explicit and auditable, because the reviewers asked for exactly that.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURE_TABLE = PROJECT_ROOT / "outputs" / "processed" / "aligned_feature_table.csv"
REV_DIR = PROJECT_ROOT / "outputs" / "revision"
REV_TABLE_DIR = REV_DIR / "tables"
REV_FIG_DIR = REV_DIR / "figures"
for _d in (REV_DIR, REV_TABLE_DIR, REV_FIG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Composite event-score constituents and their weights (unchanged from the
# original submission; written out here so the formula has one single source).
EVENT_COMPONENTS = {
    "intermittency_index": 0.35,
    "renewable_ramp_ratio": 0.30,
    "renewable_recent_error_ratio": 0.20,
    "systemimbalance_abs": 0.15,
}

# Columns that are identifiers / calendar bookkeeping rather than features.
NON_FEATURE_COLUMNS = {
    "datetime", "date", "year", "day", "regulation_deviation_target",
    "extreme_imbalance_flag",
}

TRAIN_FRACTION = 0.70
VALID_FRACTION = 0.80  # cumulative: valid = (0.70, 0.80], test = (0.80, 1.0]


def load_feature_table() -> pd.DataFrame:
    return pd.read_csv(FEATURE_TABLE, parse_dates=["datetime"])


def split_bounds(n_rows: int) -> tuple[int, int]:
    return int(n_rows * TRAIN_FRACTION), int(n_rows * VALID_FRACTION)


EQUAL_WEIGHTS = {c: 0.25 for c in EVENT_COMPONENTS}


def build_event_score(
    df: pd.DataFrame,
    normalisation: str = "train",
    train_end: int | None = None,
    weights: dict[str, float] | None = None,
) -> pd.Series:
    """Composite event score E_t.

    normalisation='full'  reproduces the original submission (z-scores over the
                          whole record);
    normalisation='train' uses training-partition moments only, which is the
                          causal variant requested by Reviewer 1.
    `weights` defaults to the weighting of the original submission; passing
    EQUAL_WEIGHTS gives the weight-sensitivity variant.
    """
    if normalisation not in {"full", "train"}:
        raise ValueError(normalisation)
    if normalisation == "train" and train_end is None:
        train_end, _ = split_bounds(len(df))

    score = pd.Series(0.0, index=df.index, dtype=float)
    z_frame = {}
    for column, weight in (weights or EVENT_COMPONENTS).items():
        reference = df[column] if normalisation == "full" else df[column].iloc[:train_end]
        mu = float(reference.mean())
        sd = float(reference.std())
        z = pd.Series(0.0, index=df.index) if (not np.isfinite(sd) or sd == 0) else (df[column] - mu) / sd
        z_frame[f"{column}_zscore"] = z
        score = score + weight * z
    score.attrs["zscores"] = z_frame
    return score


def build_labels(
    df: pd.DataFrame,
    quantile: float = 0.98,
    horizon: int = 1,
    normalisation: str = "train",
    threshold_source: str = "train",
    weights: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, float]:
    """Attach event score and forward-looking label; return (frame, threshold)."""
    train_end, _ = split_bounds(len(df))
    out = df.copy()
    score = build_event_score(out, normalisation, train_end, weights)
    # The original pipeline also exposes the four standardised constituents as
    # model inputs; they are kept here so that the canonical 146-column feature
    # set is reproduced exactly.
    out = pd.concat([out, pd.DataFrame(score.attrs["zscores"], index=out.index)], axis=1)
    out["event_proxy_score"] = score
    reference = (
        out["event_proxy_score"] if threshold_source == "full"
        else out["event_proxy_score"].iloc[:train_end]
    )
    threshold = float(reference.quantile(quantile))
    out["future_event_label"] = (
        out["event_proxy_score"].shift(-horizon) >= threshold
    ).astype(int)
    out["future_event_proxy_score"] = out["event_proxy_score"].shift(-horizon)
    out = out.iloc[:-horizon].reset_index(drop=True)
    return out, threshold


def feature_columns(df: pd.DataFrame, drop: tuple[str, ...] = ()) -> list[str]:
    excluded = set(NON_FEATURE_COLUMNS) | set(drop) | {
        "future_event_label", "future_event_proxy_score",
    }
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in numeric if c not in excluded]


REFERENCE_SEQ_LEN = 8  # window length the evaluation index set is defined by


def make_sequences(x: np.ndarray, y: np.ndarray, seq_len: int,
                   first_end: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Windows indexed by their *end* interval.

    `first_end` fixes the first end index, so that models using different window
    lengths are evaluated on exactly the same target intervals. Without it a
    4-step model would be scored on four extra leading intervals that an 8-step
    model cannot see, and the two would not be comparable.
    """
    start = seq_len - 1 if first_end is None else max(seq_len - 1, first_end)
    seqs, labels = [], []
    for idx in range(start, len(x)):
        seqs.append(x[idx - seq_len + 1: idx + 1])
        labels.append(y[idx])
    return np.asarray(seqs, dtype=np.float32), np.asarray(labels, dtype=np.float32)


def summarise_sequence(seq: np.ndarray) -> np.ndarray:
    """Flatten a window into [mean, last, max] exactly as the tabular baselines
    in the original submission do (scripts/event_screening_real_benchmarks.py)."""
    return np.concatenate([seq.mean(axis=1), seq[:, -1, :], seq.max(axis=1)], axis=1)


def build_matrices(
    quantile: float = 0.98,
    horizon: int = 1,
    seq_len: int = 8,
    normalisation: str = "train",
    threshold_source: str = "train",
    drop_features: tuple[str, ...] = (),
    lag_features: int = 0,
    df: "pd.DataFrame | None" = None,
    weights: dict[str, float] | None = None,
    clip: float | None = None,
):
    """Reproduce the original data pipeline (scripts/run_exp51_enhanced_experiments.build_data)
    with explicit control over label normalisation, feature exclusions and an
    optional publication lag applied to every feature."""
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    raw = load_feature_table() if df is None else df.copy()
    labelled, threshold = build_labels(raw, quantile, horizon, normalisation,
                                       threshold_source, weights)

    # identical to AdvancedExperimentRunner.prepare_benchmark_dataframe
    labelled["regulation_deviation_target"] = labelled["systemimbalance_abs"].shift(-1)
    labelled = labelled.dropna().reset_index(drop=True)

    cols = feature_columns(labelled, drop=drop_features)
    matrix = labelled[cols].copy()
    if lag_features > 0:
        matrix = matrix.shift(lag_features)
    target = labelled["future_event_label"].to_numpy(dtype=np.float32)

    n = len(labelled)
    tr_end, va_end = split_bounds(n)
    imputer, scaler = SimpleImputer(strategy="median"), StandardScaler()
    tr = scaler.fit_transform(imputer.fit_transform(matrix.iloc[:tr_end]))
    va = scaler.transform(imputer.transform(matrix.iloc[tr_end:va_end]))
    te = scaler.transform(imputer.transform(matrix.iloc[va_end:]))
    if clip is not None:
        # Several inputs are extremely heavy tailed (the wind-to-solar ratio spans
        # nine orders of magnitude at night), which gradient-based models tolerate
        # far worse than trees. Winsorising the standardised inputs is a
        # preprocessing choice and is therefore selected on validation like any
        # other hyper-parameter.
        tr, va, te = (np.clip(a, -clip, clip) for a in (tr, va, te))

    # All models are scored on the same target intervals regardless of their
    # window length (see make_sequences).
    first_end = max(seq_len, REFERENCE_SEQ_LEN) - 1
    tr_s, y_tr = make_sequences(tr, target[:tr_end], seq_len, first_end)
    va_s, y_va = make_sequences(va, target[tr_end:va_end], seq_len, first_end)
    te_s, y_te = make_sequences(te, target[va_end:], seq_len, first_end)

    # The continuous quantity the label thresholds. It is available wherever the
    # label is, and is exposed so that the regression formulation of the task can
    # be evaluated on equal terms for every model family.
    aux = labelled["future_event_proxy_score"].to_numpy(dtype=np.float32)
    aux_tr = aux[:tr_end][first_end:]
    aux_va = aux[tr_end:va_end][first_end:]
    aux_te = aux[va_end:][first_end:]

    meta = {
        "threshold": threshold,
        "n_features": len(cols),
        "feature_names": cols,
        "n_rows": n,
        "aux_train": aux_tr,
        "aux_valid": aux_va,
        "aux_test": aux_te,
        "first_end": first_end,
        "test_datetimes": labelled["datetime"].iloc[va_end + first_end:].reset_index(drop=True),
        "test_frame": labelled.iloc[va_end + first_end:].reset_index(drop=True),
    }
    return tr_s, va_s, te_s, y_tr, y_va, y_te, meta


def build_matrices_fold(
    train_frac: float,
    valid_frac: float,
    test_frac: float,
    quantile: float = 0.98,
    horizon: int = 1,
    seq_len: int = 8,
    drop_features: tuple[str, ...] = (),
):
    """Rolling-origin variant: an expanding training window followed by a
    contiguous validation and test block.  Label normalisation and the event
    threshold are always estimated on that fold's training window only."""
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    raw = load_feature_table()
    n_raw = len(raw)
    tr_end_raw = int(n_raw * train_frac)

    out = raw.copy()
    score = build_event_score(out, "train", tr_end_raw)
    out = pd.concat([out, pd.DataFrame(score.attrs["zscores"], index=out.index)], axis=1)
    out["event_proxy_score"] = score
    threshold = float(out["event_proxy_score"].iloc[:tr_end_raw].quantile(quantile))
    out["future_event_label"] = (out["event_proxy_score"].shift(-horizon) >= threshold).astype(int)
    out["future_event_proxy_score"] = out["event_proxy_score"].shift(-horizon)
    out = out.iloc[:-horizon].reset_index(drop=True)
    out["regulation_deviation_target"] = out["systemimbalance_abs"].shift(-1)
    out = out.dropna().reset_index(drop=True)

    n = len(out)
    tr_end = int(n * train_frac)
    va_end = int(n * valid_frac)
    te_end = int(n * test_frac)

    cols = feature_columns(out, drop=drop_features)
    matrix = out[cols]
    target = out["future_event_label"].to_numpy(dtype=np.float32)

    imputer, scaler = SimpleImputer(strategy="median"), StandardScaler()
    tr = scaler.fit_transform(imputer.fit_transform(matrix.iloc[:tr_end]))
    va = scaler.transform(imputer.transform(matrix.iloc[tr_end:va_end]))
    te = scaler.transform(imputer.transform(matrix.iloc[va_end:te_end]))

    first_end = max(seq_len, REFERENCE_SEQ_LEN) - 1
    tr_s, y_tr = make_sequences(tr, target[:tr_end], seq_len, first_end)
    va_s, y_va = make_sequences(va, target[tr_end:va_end], seq_len, first_end)
    te_s, y_te = make_sequences(te, target[va_end:te_end], seq_len, first_end)
    meta = {
        "threshold": threshold,
        "n_features": len(cols),
        "test_start": str(out["datetime"].iloc[va_end]),
        "test_end": str(out["datetime"].iloc[te_end - 1]),
    }
    return tr_s, va_s, te_s, y_tr, y_va, y_te, meta


SCORE_DIR = REV_DIR / "scores"
SCORE_DIR.mkdir(parents=True, exist_ok=True)
SCORE_INDEX = SCORE_DIR / "index.json"


def slug(name: str) -> str:
    """Filesystem-safe identifier for a model name (Windows forbids ':' and would
    otherwise silently create an alternate data stream)."""
    import re
    return re.sub(r"[^0-9A-Za-z]+", "_", name).strip("_")


def save_scores(name: str, valid_scores, test_scores) -> None:
    """Persist one model's validation and test scores and register its display name."""
    import json
    tag = slug(name)
    np.save(SCORE_DIR / f"valid__{tag}.npy", np.asarray(valid_scores, dtype=np.float64))
    np.save(SCORE_DIR / f"test__{tag}.npy", np.asarray(test_scores, dtype=np.float64))
    index = {}
    if SCORE_INDEX.exists():
        index = json.loads(SCORE_INDEX.read_text(encoding="utf-8"))
    index[tag] = name
    SCORE_INDEX.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")


def load_all_scores() -> tuple[dict, dict]:
    """Return {display name: valid scores}, {display name: test scores}."""
    import json
    index = json.loads(SCORE_INDEX.read_text(encoding="utf-8")) if SCORE_INDEX.exists() else {}
    valid, test = {}, {}
    for tag, name in index.items():
        vp, tp = SCORE_DIR / f"valid__{tag}.npy", SCORE_DIR / f"test__{tag}.npy"
        if vp.exists() and tp.exists():
            valid[name] = np.load(vp)
            test[name] = np.load(tp)
    return valid, test
