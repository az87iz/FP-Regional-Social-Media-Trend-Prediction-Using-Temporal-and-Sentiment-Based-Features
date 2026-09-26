"""
Builds the feature matrix used in Experiments 1-3.

    trends_raw_extended.csv + sentiment_features.csv -> feature_matrix.csv

Week alignment: Google Trends weeks start on a Sunday, but the sentiment
weeks were made with to_period("W-SUN"), whose start_time is a Monday. The
two are one day apart, so merging on 'week' directly gives zero matches
without any error. The sentiment weeks are shifted back to line up, and the
script stops if the join comes back empty.

Run:  python build_matrix.py
"""

import sys

import numpy as np
import pandas as pd

TRENDS_CSV    = "data/trends_raw_extended.csv"
SENTIMENT_CSV = "data/sentiment_features.csv"
OUT_CSV       = "data/feature_matrix.csv"

TOP_PCT = 0.80            # top-20% label threshold, per region
THRESH_TRAIN_FRAC = 0.35  # threshold is set from this early share of weeks
ROLL    = 3               # rolling window in weeks


def temporal_features(tr):
    """Temporal features for each (geo, topic) series.

    Trends reports 0 when search volume is below its floor, which happens a
    lot for Bahrain and for smaller topics. A plain pct_change would divide
    by zero there and the gaps would carry into the lags and rolling
    windows. Velocity is calculated as (x_t - x_{t-1}) / (x_{t-1} + 1)
    so it stays defined when the previous week is 0.
    """
    tr = tr.sort_values(["geo", "topic", "week"]).copy()
    key = ["geo", "topic"]

    prev = tr.groupby(key)["gt_score"].shift(1)
    tr["velocity"] = (tr["gt_score"] - prev) / (prev + 1.0)
    tr["velocity"] = tr["velocity"].replace([np.inf, -np.inf], np.nan)

    tr["accel"] = tr.groupby(key)["velocity"].diff()
    tr["roll_mean"] = (tr.groupby(key)["velocity"]
                         .transform(lambda s: s.rolling(ROLL, min_periods=2).mean()))
    tr["roll_std"] = (tr.groupby(key)["velocity"]
                        .transform(lambda s: s.rolling(ROLL, min_periods=2).std()))
    tr["vel_lag1"] = tr.groupby(key)["velocity"].shift(1)
    tr["vel_lag2"] = tr.groupby(key)["velocity"].shift(2)

    # Label: velocity above the region's 80th percentile, shifted to next week.
    # The threshold only uses the first 35% of weeks, so the test period has
    # no influence on what counts as a top-20% week.
    weeks = np.sort(tr["week"].unique())
    cutoff = weeks[int(len(weeks) * THRESH_TRAIN_FRAC)]
    hist = tr[tr["week"] <= cutoff]
    thr = (hist.groupby("geo")["velocity"]
               .quantile(TOP_PCT).rename("thresh").reset_index())
    tr = tr.merge(thr, on="geo", how="left")
    tr["is_top20"] = (tr["velocity"] > tr["thresh"]).astype(int)
    tr["target"] = tr.groupby(key)["is_top20"].shift(-1)
    print(f"  label threshold from weeks up to {str(cutoff)[:10]}")

    n_zero = int((tr["gt_score"] == 0).sum())
    if n_zero:
        print(f"  {n_zero:,} weeks have gt_score 0")
    return tr


def main():
    try:
        tr = pd.read_csv(TRENDS_CSV, parse_dates=["week"])
        se = pd.read_csv(SENTIMENT_CSV, parse_dates=["week"])
    except FileNotFoundError as e:
        sys.exit(f"{e}\nBoth CSVs need to be in the same folder as this script.")

    print(f"Trends:    {len(tr):,} rows, {tr.week.nunique()} weeks, "
          f"{sorted(tr.geo.unique())}")
    print(f"Sentiment: {len(se):,} rows, {se.week.nunique()} weeks, "
          f"{sorted(se.region.unique())}")

    # Trends weeks start Sunday, sentiment weeks start Monday
    tr_dow = tr.week.dt.day_name().mode()[0]
    se_dow = se.week.dt.day_name().mode()[0]
    print(f"\nWeek starts: Trends={tr_dow}, sentiment={se_dow}")
    if tr_dow != se_dow:
        shift = (se.week.dt.dayofweek.mode()[0] - tr.week.dt.dayofweek.mode()[0]) % 7
        print(f"  shifting sentiment weeks back {shift} day(s)")
        se["week"] = se["week"] - pd.Timedelta(days=shift)
        assert se.week.dt.day_name().mode()[0] == tr_dow, "shift failed"

    # Temporal features are built on the full Trends grid before the join.
    # Sentiment weeks have gaps, so building them after the join would break
    # the lags and rolling windows for any week following a gap.
    tr = temporal_features(tr)

    se = se.rename(columns={"region": "geo"})
    sent_cols = ["n_comments", "sent_mean", "sent_std", "prop_pos", "prop_neg",
                 "prop_neu", "sent_mean_delta", "prop_neg_delta",
                 "prop_pos_delta", "sent_mean_lag1"]

    m = tr.merge(se[["geo", "topic", "week"] + sent_cols],
                 on=["geo", "topic", "week"], how="inner")

    if len(m) == 0:
        sys.exit("Join returned zero rows. Check the week starts and topic names.")
    print(f"\nJoined: {len(m):,} rows "
          f"({len(m)/len(se)*100:.1f}% of sentiment rows matched)")
    if len(m) < 0.5 * len(se):
        print("  warning: under half the sentiment rows matched, check the topic names")

    # Sentiment is left unnormalised here. Normalising over the whole timeline
    # would use test-period statistics, so it is done inside each fold in
    # run_experiments.py using the training rows only.

    m = m.dropna(subset=["target"]).copy()
    m["target"] = m["target"].astype(int)

    # drop the first weeks of each series, which have no lag history yet
    need = ["velocity", "accel", "roll_mean", "roll_std", "vel_lag1", "vel_lag2"]
    before = len(m)
    lost = m[m[need].isna().any(axis=1)]
    m = m.dropna(subset=need)
    print(f"Dropped {before - len(m)} rows with no temporal history")
    if before - len(m) > 0.15 * before:
        print("  warning: that is a lot of rows, affected cells:")
        print("   ", lost.groupby(["geo", "topic"]).size().to_dict())

    m = m.sort_values(["week", "geo", "topic"]).reset_index(drop=True)
    m.to_csv(OUT_CSV, index=False)

    print(f"\n{OUT_CSV}: {len(m):,} rows")
    print(f"Weeks: {m.week.min().date()} to {m.week.max().date()}")
    print(f"Positive rate: {m.target.mean():.1%}")
    print("\nRows per region-topic:")
    print(m.groupby(["geo", "topic"]).size().to_string())
    print("\nPositive rate per region:")
    print(m.groupby("geo")["target"].agg(["size", "mean"]).round(3).to_string())


if __name__ == "__main__":
    main()
