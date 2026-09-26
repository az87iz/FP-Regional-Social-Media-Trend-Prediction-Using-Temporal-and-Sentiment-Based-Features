"""
Pulls weekly Google Trends interest for every topic and region over the
same window as the YouTube comments, so the two can be joined on
(region, topic, week).

Output columns: week, gt_score, geo, topic

Trends rate-limits heavily (HTTP 429), so every finished topic-region pair
is cached to disk. If the script stops, running it again continues from
where it left off.

Usage:
    pip install pytrends pandas
    python src/pull_trends.py
    python src/pull_trends.py "--status"     # show progress without requesting
    python src/pull_trends.py "--rebuild"    # rebuild the CSV from the cache

In PowerShell the flags need quotes, as above.
"""

import os
import random
import sys
import time
from datetime import datetime

import pandas as pd

try:
    from pytrends.request import TrendReq
except ImportError:
    sys.exit("Run: pip install pytrends pandas")

# same window as the YouTube collection
WINDOW_START = "2024-06-01"
WINDOW_END   = datetime.now().strftime("%Y-%m-%d")

# Google geo codes (UAE is "AE")
REGIONS = ["BH", "SA", "AE", "US"]

# Key is the topic name used in the rest of the project, value is the
# search term sent to Trends. One term per topic, because Trends scales
# 0-100 within each query and adding terms would change the scale.
TOPICS = {
    "ChatGPT":                 "ChatGPT",
    "artificial intelligence": "artificial intelligence",
    "bitcoin":                 "bitcoin",
    "electric cars":           "electric car",
    "Ramadan":                 "Ramadan",
    "inflation":               "inflation",
    "weather":                 "weather",
    "World Cup":               "World Cup",
}

CACHE_DIR = "data/raw/trends_cache"
OUT_CSV   = "data/trends_raw_extended.csv"

# kept slow on purpose to avoid getting blocked
BASE_SLEEP    = 12      # seconds between requests
MAX_RETRIES   = 4
BACKOFF_START = 60      # seconds to wait after a 429, doubled each retry


def cache_path(topic, geo):
    safe = topic.replace(" ", "_").replace("/", "_")
    return os.path.join(CACHE_DIR, f"{safe}__{geo}.csv")


def already_done(topic, geo):
    return os.path.exists(cache_path(topic, geo))


def pull_one(pytrends, topic, query, geo):
    """Pull one topic-region series. Returns a DataFrame or None."""
    timeframe = f"{WINDOW_START} {WINDOW_END}"
    pytrends.build_payload([query], cat=0, timeframe=timeframe, geo=geo, gprop="")
    df = pytrends.interest_over_time()
    if df is None or df.empty:
        return None
    if "isPartial" in df.columns:
        df = df[~df["isPartial"]] if df["isPartial"].any() else df.drop(columns=["isPartial"])
    if "isPartial" in df.columns:
        df = df.drop(columns=["isPartial"])
    out = pd.DataFrame({
        "week": pd.to_datetime(df.index),
        "gt_score": df[query].astype(int).values,
        "geo": geo,
        "topic": topic,
    })
    return out


def rebuild_from_cache():
    if not os.path.isdir(CACHE_DIR):
        sys.exit("No cache folder yet, run a pull first.")
    frames = []
    for fn in sorted(os.listdir(CACHE_DIR)):
        if fn.endswith(".csv"):
            frames.append(pd.read_csv(os.path.join(CACHE_DIR, fn),
                                      parse_dates=["week"]))
    if not frames:
        sys.exit("Cache is empty.")
    all_df = pd.concat(frames, ignore_index=True)
    all_df = all_df.sort_values(["topic", "geo", "week"])
    all_df.to_csv(OUT_CSV, index=False)
    return all_df


def show_status():
    total = len(TOPICS) * len(REGIONS)
    done = sum(1 for t in TOPICS for g in REGIONS if already_done(t, g))
    print(f"Pairs complete: {done}/{total}")
    missing = [(t, g) for t in TOPICS for g in REGIONS if not already_done(t, g)]
    if missing:
        print("\nStill to pull:")
        for t, g in missing:
            print(f"  {g}  {t}")
    if os.path.exists(OUT_CSV):
        df = pd.read_csv(OUT_CSV, parse_dates=["week"])
        print(f"\n{OUT_CSV}: {len(df):,} rows, "
              f"{df['week'].min().date()} to {df['week'].max().date()}")


def main():
    if "--status" in sys.argv:
        show_status()
        return
    if "--rebuild" in sys.argv:
        df = rebuild_from_cache()
        print(f"Rebuilt {OUT_CSV}: {len(df):,} rows")
        return

    os.makedirs(CACHE_DIR, exist_ok=True)
    pytrends = TrendReq(hl="en-US", tz=180, retries=2, backoff_factor=0.5)

    pairs = [(t, q, g) for t, q in TOPICS.items() for g in REGIONS]
    todo = [(t, q, g) for t, q, g in pairs if not already_done(t, g)]

    print(f"Window: {WINDOW_START} to {WINDOW_END}")
    print(f"{len(pairs) - len(todo)} of {len(pairs)} pairs cached, "
          f"{len(todo)} to pull\n")

    if not todo:
        df = rebuild_from_cache()
        print(f"{OUT_CSV}: {len(df):,} rows")
        return

    failed = []
    for i, (topic, query, geo) in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {geo:3} {topic}")
        backoff = BACKOFF_START
        ok = False
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                df = pull_one(pytrends, topic, query, geo)
                if df is None or df.empty:
                    print("    no data returned")
                    # save an empty file so this pair is not retried every run
                    pd.DataFrame(columns=["week", "gt_score", "geo", "topic"]) \
                        .to_csv(cache_path(topic, geo), index=False)
                    ok = True
                    break
                df.to_csv(cache_path(topic, geo), index=False)
                print(f"    {len(df)} weeks cached")
                ok = True
                break
            except Exception as e:
                msg = str(e)
                if "429" in msg or "Too Many Requests" in msg or "rate" in msg.lower():
                    print(f"    rate limited (attempt {attempt}/{MAX_RETRIES}), "
                          f"waiting {backoff}s")
                    time.sleep(backoff)
                    backoff *= 2
                else:
                    print(f"    error: {e}")
                    time.sleep(10)
        if not ok:
            failed.append((topic, geo))
            print("    failed, will retry next run")

        # small random delay between requests
        time.sleep(BASE_SLEEP + random.uniform(0, 5))

    df = rebuild_from_cache()
    print(f"\n{OUT_CSV}: {len(df):,} rows")
    print(f"Weeks: {df['week'].min().date()} to {df['week'].max().date()}")
    print("\nWeeks per region:")
    print(df.groupby("geo")["week"].nunique().to_string())
    print("\nRows per topic-region:")
    print(df.groupby(["topic", "geo"]).size().to_string())

    if failed:
        print(f"\n{len(failed)} pairs failed, run the script again:")
        for t, g in failed:
            print(f"  {g}  {t}")


if __name__ == "__main__":
    main()
