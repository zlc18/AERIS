"""rev08 - Feature dictionary and information-availability classification.

Answers Reviewer 2 comment 2 (exact feature-engineering procedure) and supplies
the table Reviewer 1 comment 2 asks for: for every input family, what it is,
how it is computed, and whether its value for interval t is an ex-ante
quantity (published before t) or an ex-post / near-real-time quantity
(published at or after the end of t).

Every derived definition written here was verified numerically against
outputs/processed/aligned_feature_table.csv; the check is re-run by this script
so that the table in the paper cannot drift from the data.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.revision.rev_common import REV_TABLE_DIR, load_feature_table  # noqa: E402

TOL = 1e-6

import re  # noqa: E402

CALENDAR = {"month", "day_of_year", "day_of_week", "is_weekend", "hour", "minute",
            "quarter_index", "quarter_sin", "quarter_cos", "dayofweek_sin",
            "dayofweek_cos", "month_sin", "month_cos"}
BALANCING = {"ace", "systemimbalance", "systemimbalance_abs", "alpha", "alpha_prime",
             "marginalincrementalprice", "marginaldecrementalprice", "imbalanceprice",
             "imbalanceprice_abs", "price_spread", "imbalance_stress_index"}
COMPOSITE = {"intermittency_index", "renewable_uncertainty_total",
             "renewable_recent_error_ratio", "renewable_ramp_ratio", "wind_solar_ratio",
             "renewable_recent_error_abs", "renewable_dayahead_error_abs",
             "renewable_dayahead_error_ratio", "renewable_ramp_abs"}

FAMILY_INFO = {
    "Renewable measurement": (
        "Wind: sum over the disjoint ODS031 region / grid-connection records of the "
        "quarter-hour; photovoltaic: the national ODS032 record; the load factor is "
        "the capacity-weighted mean of the record-level load factors",
        "Published for interval t shortly after it ends", "ex post"),
    "Forecasts and uncertainty bands": (
        "Published forecast levels and the width of the 10-90% confidence band, "
        "bw_h = c90_h - c10_h, for the most-recent, day-ahead-11h, day-ahead and "
        "week-ahead vintages", "Issued before interval t", "ex ante"),
    "Forecast deviations": (
        "e_h = measured - forecast_h; the ratios divide |e_h| by monitored capacity",
        "Requires the measurement of interval t", "ex post"),
    "Ramp indicators": (
        "ramp = |measured_t - measured_{t-1}|; the ratio divides it by monitored capacity",
        "Requires the measurement of interval t", "ex post"),
    "Composite renewable stress": (
        "I_t = ramp ratio (wind) + ramp ratio (solar) + recent error ratio (wind) + "
        "recent error ratio (solar); R_t and F_t are the system-level ramp and "
        "forecast-error ratios; wind_solar_ratio = wind / (solar + 1e-6)",
        "Requires the measurement of interval t", "ex post"),
    "Balancing and market state": (
        "ODS134 settlement quantities; price_spread = |MIP - MDP|; "
        "imbalance_stress_index = |SI| x F", "Settled after interval t", "ex post"),
    "Rolling statistics": (
        "Trailing rolling mean and standard deviation over 4, 16 and 96 intervals "
        "(1 h, 4 h, 1 d) of nine base series", "Follows the underlying series", "ex post"),
    "Calendar encodings": (
        "Deterministic functions of the timestamp, including sine/cosine encodings of "
        "the quarter-hour, weekday and month", "Known arbitrarily far ahead", "ex ante"),
    "Label-related inputs": (
        "The four standardised constituents of the composite score and their weighted "
        "sum at time t", "Requires the measurement of interval t", "ex post"),
}


def family_of(column: str) -> str:
    if column == "event_proxy_score" or column.endswith("_zscore"):
        return "Label-related inputs"
    if re.search(r"_(mean|std)_(1h|4h|1d)$", column):
        return "Rolling statistics"
    if column in CALENDAR:
        return "Calendar encodings"
    if column in BALANCING:
        return "Balancing and market state"
    if column in COMPOSITE:
        return "Composite renewable stress"
    if "ramp" in column:
        return "Ramp indicators"
    if "error" in column:
        return "Forecast deviations"
    if "forecast" in column or "confidence" in column or "bandwidth" in column:
        return "Forecasts and uncertainty bands"
    return "Renewable measurement"


def verify(df: pd.DataFrame) -> pd.DataFrame:
    checks = []

    def chk(name, a, b):
        a, b = np.asarray(a, float), np.asarray(b, float)
        m = np.isfinite(a) & np.isfinite(b)
        # relative to the magnitude, since wind_solar_ratio reaches ~1e9 at night
        dev = float(np.max(np.abs(a[m] - b[m]) / np.maximum(1.0, np.abs(b[m]))))
        checks.append({"Definition": name,
                       "Max relative deviation": dev,
                       "Verified": dev < TOL})

    chk("wind_mostrecent_error = wind_measured - wind_mostrecentforecast",
        df.wind_mostrecent_error, df.wind_measured - df.wind_mostrecentforecast)
    chk("wind_recent_bandwidth = c90 - c10",
        df.wind_recent_bandwidth, df.wind_mostrecentconfidence90 - df.wind_mostrecentconfidence10)
    chk("wind_ramp_abs = |diff(wind_measured)|", df.wind_ramp_abs, df.wind_measured.diff().abs())
    chk("wind_recent_error_ratio = |error| / capacity",
        df.wind_recent_error_ratio, df.wind_mostrecent_error.abs() / df.wind_monitoredcapacity)
    chk("renewable_measured_total = wind + solar",
        df.renewable_measured_total, df.wind_measured + df.solar_measured)
    chk("renewable_recent_error_abs = |e_wind| + |e_solar|",
        df.renewable_recent_error_abs, df.wind_mostrecent_error.abs() + df.solar_mostrecent_error.abs())
    chk("renewable_ramp_ratio = (ramp_wind + ramp_solar) / capacity_total",
        df.renewable_ramp_ratio,
        (df.wind_ramp_abs + df.solar_ramp_abs) / df.renewable_capacity_total)
    chk("renewable_uncertainty_total = bw_wind + bw_solar",
        df.renewable_uncertainty_total, df.wind_recent_bandwidth + df.solar_recent_bandwidth)
    chk("intermittency_index = ramp_ratio_w + ramp_ratio_s + err_ratio_w + err_ratio_s",
        df.intermittency_index,
        df.wind_ramp_ratio.fillna(0) + df.solar_ramp_ratio.fillna(0)
        + df.wind_recent_error_ratio + df.solar_recent_error_ratio)
    chk("imbalance_stress_index = |SI| x renewable_recent_error_ratio",
        df.imbalance_stress_index, df.systemimbalance_abs * df.renewable_recent_error_ratio)
    chk("price_spread = |MIP - MDP|",
        df.price_spread, (df.marginalincrementalprice - df.marginaldecrementalprice).abs())
    chk("wind_solar_ratio = wind / (solar + 1e-6)",
        df.wind_solar_ratio, df.wind_measured / (df.solar_measured + 1e-6))
    chk("systemimbalance_mean_1h = rolling mean over 4 intervals",
        df.systemimbalance_mean_1h, df.systemimbalance.rolling(4, min_periods=1).mean())
    chk("systemimbalance_std_1d = rolling std over 96 intervals",
        df.systemimbalance_std_1d, df.systemimbalance.rolling(96, min_periods=2).std())
    chk("extreme_imbalance_flag = 1[|SI| >= Q95(|SI|)] (excluded from inputs)",
        df.extreme_imbalance_flag.astype(float),
        (df.systemimbalance_abs >= df.systemimbalance_abs.quantile(0.95)).astype(float))
    return pd.DataFrame(checks)


def main() -> None:
    from scripts.revision.rev_common import build_matrices

    df = load_feature_table()
    *_, meta = build_matrices(normalisation="full", threshold_source="full")
    columns = meta["feature_names"]

    assignments = pd.DataFrame({"Column": columns,
                                "Family": [family_of(c) for c in columns]})
    assignments.to_csv(REV_TABLE_DIR / "rev08_feature_assignment.csv",
                       index=False, encoding="utf-8-sig")

    rows = []
    for family, (definition, timing, klass) in FAMILY_INFO.items():
        members = assignments.loc[assignments["Family"] == family, "Column"].tolist()
        rows.append({
            "Family": family,
            "Inputs": len(members),
            "Definition": definition,
            "Availability of the value for interval t": timing,
            "Class": klass,
            "Example columns": ", ".join(members[:3]),
        })
    families = pd.DataFrame(rows).sort_values("Inputs", ascending=False)
    families.to_csv(REV_TABLE_DIR / "rev08_feature_families.csv",
                    index=False, encoding="utf-8-sig")

    checks = verify(df)
    checks.to_csv(REV_TABLE_DIR / "rev08_definition_checks.csv",
                  index=False, encoding="utf-8-sig")
    print(families[["Family", "Inputs", "Class",
                    "Availability of the value for interval t"]].to_string(index=False))
    print()
    print(checks.to_string(index=False))
    print(f"\nall definitions verified: {bool(checks['Verified'].all())}")
    print(f"family counts sum to {int(families['Inputs'].sum())} of {len(columns)} inputs")


if __name__ == "__main__":
    main()
