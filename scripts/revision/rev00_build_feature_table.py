"""Build the aligned quarter-hour feature table from the raw Elia open-data files.

Inputs (JSON exports of the Elia Open Data portal, placed in ``dataset/``):
  ods031.json  wind-power production, measured and forecast, per record
  ods032.json  photovoltaic production, measured and forecast, per record
  ods134.json  system imbalance, ACE and imbalance prices

Output: ``outputs/processed/aligned_feature_table.csv`` (146 columns), the input
of every revision script. Run ``python -m scripts.revision.rev00_build_feature_table``;
add ``--check`` to compare a freshly built table with an existing one.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.revision.rev_common import FEATURE_TABLE, PROJECT_ROOT

RAW_DIR = PROJECT_ROOT / "dataset"

VINTAGES = ["mostrecent", "dayahead11h", "dayahead", "weekahead"]
LEVEL_COLUMNS = ["measured"] + [
    f"{v}{kind}" for v in VINTAGES for kind in ("forecast", "confidence10", "confidence90")
] + ["monitoredcapacity"]
BALANCING_COLUMNS = ["ace", "systemimbalance", "alpha", "alpha_prime",
                     "marginalincrementalprice", "marginaldecrementalprice", "imbalanceprice"]
ROLLING_BASES = ["wind_measured", "solar_measured", "wind_mostrecent_error",
                 "solar_mostrecent_error", "renewable_measured_total",
                 "renewable_recent_error_ratio", "renewable_ramp_ratio",
                 "systemimbalance", "imbalanceprice"]
WINDOWS = {"1h": 4, "4h": 16, "1d": 96}
NATIONAL_REGION = "Belgium"


def _read(name: str) -> pd.DataFrame:
    df = pd.read_json(RAW_DIR / f"{name}.json")
    df["datetime"] = (pd.to_datetime(df["datetime"], utc=True)
                      .dt.tz_convert(None))
    return df


def aggregate_source(raw: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Sum the records of each quarter-hour; capacity-weighted load factor.

    The records passed in must be disjoint: ODS031 splits wind into disjoint
    region / grid-connection records, while ODS032 lists Belgium, its regions and
    its provinces side by side, so only its national rows are passed (see build).
    """
    raw = raw.copy()
    raw["_lf_x_cap"] = raw["loadfactor"] * raw["monitoredcapacity"]
    agg = raw.groupby("datetime")[LEVEL_COLUMNS + ["_lf_x_cap"]].sum(min_count=1)
    agg["loadfactor_weighted"] = agg["_lf_x_cap"] / agg["monitoredcapacity"].replace(0, np.nan)
    agg = agg.drop(columns="_lf_x_cap")

    agg["mostrecent_error"] = agg["measured"] - agg["mostrecentforecast"]
    for v in ["dayahead11h", "dayahead", "weekahead"]:
        agg[f"{v}_error"] = agg["measured"] - agg[f"{v}forecast"]
    for short, v in [("recent", "mostrecent"), ("dayahead", "dayahead"), ("weekahead", "weekahead")]:
        agg[f"{short}_bandwidth"] = agg[f"{v}confidence90"] - agg[f"{v}confidence10"]
    agg["ramp_abs"] = agg["measured"].diff().abs()
    cap = agg["monitoredcapacity"].replace(0, np.nan)
    agg["recent_error_ratio"] = agg["mostrecent_error"].abs() / cap
    agg["dayahead_error_ratio"] = agg["dayahead_error"].abs() / cap
    agg["ramp_ratio"] = agg["ramp_abs"] / cap
    return agg.add_prefix(f"{prefix}_")


def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    t = df["datetime"]
    df["date"] = t.dt.date.astype(str)
    df["year"] = t.dt.year
    df["month"] = t.dt.month
    df["day"] = t.dt.day
    df["day_of_year"] = t.dt.dayofyear
    df["day_of_week"] = t.dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["hour"] = t.dt.hour
    df["minute"] = t.dt.minute
    df["quarter_index"] = df["hour"] * 4 + df["minute"] // 15
    df["quarter_sin"] = np.sin(2 * np.pi * df["quarter_index"] / 96)
    df["quarter_cos"] = np.cos(2 * np.pi * df["quarter_index"] / 96)
    df["dayofweek_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dayofweek_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


def add_system_features(df: pd.DataFrame) -> pd.DataFrame:
    w_err, s_err = df["wind_mostrecent_error"].abs(), df["solar_mostrecent_error"].abs()
    w_ramp, s_ramp = df["wind_ramp_abs"].fillna(0), df["solar_ramp_abs"].fillna(0)
    df["renewable_measured_total"] = df["wind_measured"] + df["solar_measured"]
    df["renewable_capacity_total"] = df["wind_monitoredcapacity"] + df["solar_monitoredcapacity"]
    cap = df["renewable_capacity_total"].replace(0, np.nan)
    df["renewable_recent_error_abs"] = w_err + s_err
    df["renewable_dayahead_error_abs"] = (df["wind_dayahead_error"].abs()
                                          + df["solar_dayahead_error"].abs())
    df["renewable_recent_error_ratio"] = df["renewable_recent_error_abs"] / cap
    df["renewable_dayahead_error_ratio"] = df["renewable_dayahead_error_abs"] / cap
    df["renewable_ramp_abs"] = w_ramp + s_ramp
    df["renewable_ramp_ratio"] = df["renewable_ramp_abs"] / cap
    df["renewable_uncertainty_total"] = df["wind_recent_bandwidth"] + df["solar_recent_bandwidth"]
    df["wind_solar_ratio"] = df["wind_measured"] / (df["solar_measured"] + 1e-6)
    df["intermittency_index"] = (df["wind_ramp_ratio"].fillna(0) + df["solar_ramp_ratio"].fillna(0)
                                 + df["wind_recent_error_ratio"] + df["solar_recent_error_ratio"])
    df["imbalance_stress_index"] = df["systemimbalance_abs"] * df["renewable_recent_error_ratio"]
    return df


def add_rolling(df: pd.DataFrame) -> pd.DataFrame:
    cols = {}
    for base in ROLLING_BASES:
        for tag, n in WINDOWS.items():
            cols[f"{base}_mean_{tag}"] = df[base].rolling(n, min_periods=1).mean()
            cols[f"{base}_std_{tag}"] = df[base].rolling(n, min_periods=2).std()
    return pd.concat([df, pd.DataFrame(cols, index=df.index)], axis=1)


def build() -> pd.DataFrame:
    wind = aggregate_source(_read("ods031"), "wind")
    pv = _read("ods032")
    # national rows only: the regional and provincial rows are subdivisions of it
    solar = aggregate_source(pv[pv["region"] == NATIONAL_REGION], "solar")
    bal = _read("ods134").groupby("datetime")[BALANCING_COLUMNS].mean()

    df = wind.join(solar, how="inner").join(bal, how="inner").sort_index()
    df = df.reset_index()
    df["systemimbalance_abs"] = df["systemimbalance"].abs()
    df["price_spread"] = (df["marginalincrementalprice"] - df["marginaldecrementalprice"]).abs()
    df["imbalanceprice_abs"] = df["imbalanceprice"].abs()
    df = add_calendar(df)
    df = add_system_features(df)
    df = add_rolling(df)
    thr = df["systemimbalance_abs"].quantile(0.95)
    df["extreme_imbalance_flag"] = (df["systemimbalance_abs"] >= thr).astype(int)
    return df


def compare(new: pd.DataFrame, ref: pd.DataFrame, tol: float = 1e-4) -> bool:
    ok = True
    if len(new) != len(ref):
        print(f"row count differs: {len(new)} vs {len(ref)}")
        ok = False
    missing = [c for c in ref.columns if c not in new.columns]
    extra = [c for c in new.columns if c not in ref.columns]
    if missing or extra:
        print(f"missing columns: {missing}\nextra columns: {extra}")
        ok = False
    n = min(len(new), len(ref))
    for c in ref.columns:
        if c not in new.columns:
            continue
        a, b = new[c].iloc[:n], ref[c].iloc[:n]
        if c in ("datetime", "date"):
            same = (a.astype(str).values == b.astype(str).values).all()
            dev = 0.0 if same else np.inf
        else:
            a, b = a.to_numpy(float), b.to_numpy(float)
            nan_ok = np.array_equal(np.isnan(a), np.isnan(b))
            m = ~np.isnan(a) & ~np.isnan(b)
            # deviations are scaled by the column's spread: rolling standard deviations
            # of long runs of zeros differ by floating-point residue (~1e-3 MW)
            scale = max(1.0, float(np.nanstd(b)))
            dev = np.max(np.abs(a[m] - b[m])) / scale if m.any() else 0.0
            dev = dev if nan_ok else np.inf
        if dev > tol:
            print(f"  {c}: max scaled deviation {dev:.3g}")
            ok = False
    print(f"{len(ref.columns)} columns compared: {'identical within tolerance' if ok else 'DIFFERENCES FOUND'}")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=FEATURE_TABLE)
    ap.add_argument("--check", type=Path, default=None,
                    help="existing table to compare against instead of writing")
    args = ap.parse_args()

    df = build()
    if args.check is not None:
        ref = pd.read_csv(args.check, parse_dates=["datetime"])
        raise SystemExit(0 if compare(df, ref) else 1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False, float_format="%.10g")
    print(f"wrote {args.out} ({len(df)} rows x {df.shape[1]} columns)")


if __name__ == "__main__":
    main()
