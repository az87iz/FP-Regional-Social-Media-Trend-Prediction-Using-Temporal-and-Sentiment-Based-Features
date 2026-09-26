"""
Scores the YouTube comments with XLM-T and builds weekly sentiment features
for each region, topic and week.

    yt_comments.csv -> XLM-T -> sentiment_features.csv

This needs a GPU. On a Colab T4 the full comment set takes around an hour,
on a CPU it takes many hours.

In Colab:
    Runtime > Change runtime type > T4 GPU
    !pip install transformers torch pandas --quiet
    (upload yt_comments.csv or mount Drive)
    !python sentiment_features.py

Notes:
- The model is XLM-T (Barbieri et al. 2022), the same one checked in the
  prototype on UMSAB (English 0.732, Arabic 0.670).
- Each comment gets P(pos) - P(neg) as a continuous score, as well as its
  top label for the proportion features.
- Weeks with fewer than 20 comments are dropped, not filled in, so a week
  with too little text simply has no sentiment features.
- Region-topic cells with fewer than 15 usable weeks are dropped as well,
  since features from a few noisy weeks are more likely to hurt than help.
- Scores are saved every 20,000 comments so a Colab disconnect does not
  lose the whole run.
"""

import os
import sys

import numpy as np
import pandas as pd

IN_CSV          = "data/raw/yt_comments.csv"
SCORED_CSV      = "data/raw/yt_comments_scored.csv"      # per-comment scores, used to resume
OUT_CSV         = "data/sentiment_features.csv"      # weekly features
COVERAGE_CSV    = "data/sentiment_coverage.csv"      # usable weeks per cell

MODEL_NAME      = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
BATCH_SIZE      = 128
MAX_LEN         = 128            # comments are short
CHECKPOINT_EVERY = 20000

MIN_COMMENTS_PER_WEEK = 20       # same threshold as the coverage analysis
MIN_USABLE_WEEKS      = 15       # drop region-topic cells thinner than this
MIN_CHARS             = 3        # skip empty or emoji-only comments


def load_comments():
    if not os.path.exists(IN_CSV):
        sys.exit(f"{IN_CSV} not found. Upload it first.")
    df = pd.read_csv(IN_CSV, encoding="utf-8-sig", low_memory=False)
    need = {"comment_id", "region", "topic", "week", "text"}
    missing = need - set(df.columns)
    if missing:
        sys.exit(f"{IN_CSV} missing columns: {missing}")

    n0 = len(df)
    df = df.drop_duplicates(subset="comment_id")
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() >= MIN_CHARS]
    df["week"] = pd.to_datetime(df["week"])
    print(f"Loaded {n0:,} rows, {len(df):,} after removing duplicates and empty comments")
    return df.reset_index(drop=True)


def score_comments(df):
    """Runs XLM-T on the comments, skipping any already in SCORED_CSV."""
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    done = pd.DataFrame()
    if os.path.exists(SCORED_CSV):
        done = pd.read_csv(SCORED_CSV, encoding="utf-8-sig")
        print(f"Resuming: {len(done):,} comments already scored")
        df = df[~df["comment_id"].isin(set(done["comment_id"]))]
        if df.empty:
            print("Nothing left to score.")
            return done

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cpu":
        print("\nNo GPU found, this will be very slow.\n")

    print(f"Loading {MODEL_NAME} on {dev}")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(dev)
    model.eval()

    # cardiffnlp label order: 0 negative, 1 neutral, 2 positive
    id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}
    print("label map:", id2label)

    texts = df["text"].tolist()
    ids = df["comment_id"].tolist()
    out_rows, buf = [], []

    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start:start + BATCH_SIZE]
        enc = tok(batch, padding=True, truncation=True,
                  max_length=MAX_LEN, return_tensors="pt").to(dev)
        with torch.no_grad():
            probs = torch.softmax(model(**enc).logits, dim=-1).cpu().numpy()

        for j, p in enumerate(probs):
            row = {"comment_id": ids[start + j]}
            for k, lab in id2label.items():
                row[f"p_{lab}"] = float(p[k])
            buf.append(row)

        if len(buf) >= CHECKPOINT_EVERY:
            out_rows.extend(buf)
            _checkpoint(buf)
            buf = []
            pct = 100 * (start + len(batch)) / len(texts)
            print(f"  {start + len(batch):,}/{len(texts):,} ({pct:.1f}%)")

    if buf:
        out_rows.extend(buf)
        _checkpoint(buf)

    scored = pd.DataFrame(out_rows)
    if len(done):
        scored = pd.concat([done, scored], ignore_index=True)
    print(f"Scored {len(scored):,} comments total")
    return scored


def _checkpoint(rows):
    d = pd.DataFrame(rows)
    header = not os.path.exists(SCORED_CSV)
    d.to_csv(SCORED_CSV, mode="a", header=header, index=False,
             encoding="utf-8-sig")


def build_features(df, scored):
    """Averages the comment scores into weekly region-topic features."""
    m = df.merge(scored, on="comment_id", how="inner")
    print(f"Merged: {len(m):,} scored comments with metadata")

    # continuous score, plus the top label for the proportions
    m["sent_score"] = m["p_positive"] - m["p_negative"]
    m["label"] = m[["p_negative", "p_neutral", "p_positive"]].values.argmax(1)
    m["is_neg"] = (m["label"] == 0).astype(int)
    m["is_neu"] = (m["label"] == 1).astype(int)
    m["is_pos"] = (m["label"] == 2).astype(int)

    g = m.groupby(["region", "topic", "week"])
    f = g.agg(
        n_comments   = ("sent_score", "size"),
        sent_mean    = ("sent_score", "mean"),
        sent_std     = ("sent_score", "std"),
        prop_pos     = ("is_pos", "mean"),
        prop_neg     = ("is_neg", "mean"),
        prop_neu     = ("is_neu", "mean"),
    ).reset_index()
    f["sent_std"] = f["sent_std"].fillna(0.0)

    # coverage table, made before filtering so it shows every week
    cov = (f.assign(usable=f["n_comments"] >= MIN_COMMENTS_PER_WEEK)
             .groupby(["region", "topic"])
             .agg(weeks_any=("week", "size"),
                  weeks_usable=("usable", "sum"),
                  median_comments=("n_comments", "median"))
             .reset_index())
    cov.to_csv(COVERAGE_CSV, index=False)

    # drop weeks with too few comments
    before = len(f)
    f = f[f["n_comments"] >= MIN_COMMENTS_PER_WEEK].copy()
    print(f"Weeks: {before:,}, {len(f):,} with {MIN_COMMENTS_PER_WEEK}+ comments")

    # drop region-topic cells with too few usable weeks
    counts = f.groupby(["region", "topic"]).size()
    keep = counts[counts >= MIN_USABLE_WEEKS].index
    dropped = counts[counts < MIN_USABLE_WEEKS]
    f = f.set_index(["region", "topic"]).loc[keep].reset_index()

    if len(dropped):
        print(f"\nExcluded cells (under {MIN_USABLE_WEEKS} usable weeks):")
        for (r, t), n in dropped.items():
            print(f"  {r} / {t}: {n} weeks")

    # week-on-week changes within each series
    f = f.sort_values(["region", "topic", "week"])
    for col in ("sent_mean", "prop_neg", "prop_pos"):
        f[f"{col}_delta"] = f.groupby(["region", "topic"])[col].diff()
    # the first week of each series is left as NaN
    f["sent_mean_lag1"] = f.groupby(["region", "topic"])["sent_mean"].shift(1)

    return f, cov


def main():
    df = load_comments()
    scored = score_comments(df)
    feats, cov = build_features(df, scored)

    feats.to_csv(OUT_CSV, index=False)

    print(f"\nRows (region-topic-weeks): {len(feats):,}")
    print(f"Regions: {sorted(feats['region'].unique())}")
    print(f"Weeks:   {feats['week'].min().date()} to {feats['week'].max().date()}")

    print("\nUsable weeks per region-topic:")
    print(feats.groupby(["region", "topic"]).size().to_string())

    print("\nMean sentiment by region (-1 to +1):")
    print(feats.groupby("region")["sent_mean"].mean().round(3).to_string())

    print(f"\nWrote {OUT_CSV}, {COVERAGE_CSV} and {SCORED_CSV}")


if __name__ == "__main__":
    main()
