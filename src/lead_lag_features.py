"""
Tests whether lead/lag features from the diffusion analysis add information
beyond the temporal features.

The diffusion analysis found that the US tends to lead on technology topics
and the UAE on regional and cultural topics. The Exp 3 regional features
compare regions within the same week, so they do not test whether a leading
region's movement predicts a follower. This script adds three lagged
features built from the other regions' Google Trends series:

    leader_vel_lag1   the leading region's velocity last week
                      (for rows from the leading region itself, the mean
                      of the other regions is used)
    others_vel_lag1   mean velocity of the other three regions last week
    others_vel_lag2   the same, two weeks back

All four regions are used, including Bahrain and the UAE, since only the
Trends series is needed. The leading region for each topic type comes from
the diffusion analysis on the 2022-2024 pull (trends_raw.csv).

Exp 1 (temporal) is compared with Exp 1 plus the lead/lag features on the
same rows and folds as run_experiments.py, using DeLong's test and the 0.02
minimum gain, pooled and per region.

Run:  python src/lead_lag_features.py
"""

import sys
import warnings

import numpy as np
import pandas as pd

from run_experiments import (TEMPORAL, SENTIMENT, MIN_GAIN, add_regional_features,
                             delong_test, expanding_folds, metrics, run_experiment)

warnings.filterwarnings("ignore")

MATRIX_CSV = "data/feature_matrix.csv"
TRENDS_CSV = "data/trends_raw_extended.csv"
OUT_CSV    = "results/lead_lag_results.csv"

TECH = {"ChatGPT", "artificial intelligence", "bitcoin", "electric cars"}
LEADER = {"tech": "US", "cultural": "AE"}     # from the diffusion analysis
LEADLAG = ["leader_vel_lag1", "others_vel_lag1", "others_vel_lag2"]


def trends_velocity(tr):
    """Velocity for every region-topic series, same formula as build_matrix.py."""
    tr = tr.sort_values(["geo", "topic", "week"]).copy()
    prev = tr.groupby(["geo", "topic"])["gt_score"].shift(1)
    tr["velocity"] = (tr["gt_score"] - prev) / (prev + 1.0)
    return tr.pivot_table(index=["topic", "week"], columns="geo", values="velocity")


def add_lead_lag(df, wide):
    """Adds the lagged features. Every value comes from weeks before the row's week."""
    df = df.copy()
    lag1 = wide.groupby(level="topic").shift(1)
    lag2 = wide.groupby(level="topic").shift(2)
    leader_v, others1, others2 = [], [], []
    for _, r in df.iterrows():
        key = (r["topic"], r["week"])
        if key not in lag1.index:
            leader_v.append(np.nan); others1.append(np.nan); others2.append(np.nan)
            continue
        l1, l2 = lag1.loc[key], lag2.loc[key]
        others = [g for g in l1.index if g != r["geo"]]
        o1, o2 = l1[others].mean(), l2[others].mean()
        lead = LEADER["tech" if r["topic"] in TECH else "cultural"]
        leader_v.append(o1 if lead == r["geo"] else l1.get(lead, np.nan))
        others1.append(o1)
        others2.append(o2)
    df["leader_vel_lag1"] = leader_v
    df["others_vel_lag1"] = others1
    df["others_vel_lag2"] = others2
    return df


def main():
    try:
        fm = pd.read_csv(MATRIX_CSV, parse_dates=["week"])
        tr = pd.read_csv(TRENDS_CSV, parse_dates=["week"])
    except FileNotFoundError as e:
        sys.exit(f"{e}\nRun build_matrix.py first.")

    # same rows, in the same order, as run_experiments.py
    df = add_regional_features(fm)
    df = df.sort_values("week").reset_index(drop=True)
    df = df.dropna(subset=[c for c in TEMPORAL + SENTIMENT if c in df.columns])
    df = df.reset_index(drop=True)
    df = add_lead_lag(df, trends_velocity(tr))
    missing = int(df[LEADLAG].isna().any(axis=1).sum())
    if missing:
        print(f"  {missing} rows have no lead/lag history, set to 0")
        df[LEADLAG] = df[LEADLAG].fillna(0.0)
    print(f"Rows: {len(df)}, folds: {len(expanding_folds(df['week']))}")

    base = run_experiment(df, TEMPORAL, "Exp1_temporal")
    lead = run_experiment(df, TEMPORAL + LEADLAG, "Exp1_plus_leadlag")
    models = [c[2:] for c in base.columns if c.startswith("p_")]

    rows = []
    for region in ["all", "SA", "US"]:
        A = base if region == "all" else base[base.geo == region]
        B = lead if region == "all" else lead[lead.geo == region]
        A, B = A.reset_index(drop=True), B.reset_index(drop=True)
        print(f"\n=== {region}: {len(A)} test predictions ===")
        for m in models:
            a_new, a_old, _, p = delong_test(A["y_true"].values,
                                             B[f"p_{m}"].values, A[f"p_{m}"].values)
            gain = a_new - a_old
            ok = p == p and p < 0.05 and gain >= MIN_GAIN
            print(f"  {m:<8} {a_old:.3f} -> {a_new:.3f}  gain {gain:+.3f}  "
                  f"p={p:.4f}  {'established' if ok else 'not established'}")
            rows.append({"region": region, "model": m, "n": len(A),
                         "auc_exp1": round(a_old, 4), "auc_leadlag": round(a_new, 4),
                         "gain": round(gain, 4), "delong_p": round(p, 4),
                         "established": ok})

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV}")


if __name__ == "__main__":
    main()
