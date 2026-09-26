"""
Draws a sample of comments for blind hand labelling.

XLM-T was checked on UMSAB, which is made of tweets, but this project uses
it on YouTube comments. Following the draft feedback, a sample of the
project's own Arabic and English comments is hand-labelled so the model can
be checked on the actual data, with a confusion matrix for each language.

    yt_comments.csv + yt_comments_scored.csv
        -> label_sample.csv   (labelled by hand, no model output in it)
        -> label_answers.csv  (model predictions, opened after labelling)

Labels: positive, negative, neutral or unusable.
  - a factual statement with no opinion is neutral
  - sarcasm gets the meaning intended
  - mixed comments get the stronger sentiment
  - emoji only, unreadable or another language is unusable
All rows are labelled before the answers file is opened, and no label is
changed afterwards.

For Saudi-channel rows there is also a looks_saudi column (yes, no or
unclear), based on Gulf dialect, Saudi place names or local references. This
gives a rough lower bound on how many Saudi-channel comments come from the
region.

Usage:
    python make_label_sample.py
    (fill in label for every row and looks_saudi for SA rows)
    python score_labels.py
"""

import os
import sys

import numpy as np
import pandas as pd

COMMENTS_CSV = "yt_comments.csv"
SCORED_CSV   = "yt_comments_scored.csv"
SAMPLE_CSV   = "label_sample.csv"
ANSWERS_CSV  = "label_answers.csv"

N_PER_LANGUAGE = 50       # 50 Arabic (SA) and 50 English (US)
SEED           = 42       # fixed so the same sample can be drawn again
MIN_CHARS      = 10       # skip very short comments


def main():
    for f in (COMMENTS_CSV, SCORED_CSV):
        if not os.path.exists(f):
            sys.exit(f"{f} not found. It needs to be in the same folder as this script.")

    com = pd.read_csv(COMMENTS_CSV, encoding="utf-8-sig", low_memory=False)
    sc  = pd.read_csv(SCORED_CSV, encoding="utf-8-sig")
    df = com.merge(sc, on="comment_id", how="inner")
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() >= MIN_CHARS]
    df["week"] = pd.to_datetime(df["week"])
    print(f"Pool: {len(df):,} scored comments "
          f"(regions: {sorted(df['region'].unique())})")

    df["model_label"] = (df[["p_negative", "p_neutral", "p_positive"]]
                         .values.argmax(1))
    df["model_label"] = df["model_label"].map({0: "negative", 1: "neutral",
                                               2: "positive"})

    rng = np.random.default_rng(SEED)
    picks = []

    for region in ["SA", "US"]:
        sub = df[df["region"] == region]
        if sub.empty:
            print(f"  no comments for {region}")
            continue

        # Sample evenly across topics and predicted classes. Otherwise the
        # sample would be mostly whichever class the model predicts most,
        # which for the US is negative.
        strata = []
        topics = sorted(sub["topic"].unique())
        per_cell = max(1, N_PER_LANGUAGE // (len(topics) * 3))

        for t in topics:
            for lab in ["negative", "neutral", "positive"]:
                cell = sub[(sub["topic"] == t) & (sub["model_label"] == lab)]
                if cell.empty:
                    continue
                take = min(per_cell, len(cell))
                strata.append(cell.sample(take, random_state=rng.integers(1e9)))

        got = pd.concat(strata) if strata else sub.head(0)
        if len(got) < N_PER_LANGUAGE:
            rest = sub[~sub["comment_id"].isin(got["comment_id"])]
            need = min(N_PER_LANGUAGE - len(got), len(rest))
            if need > 0:
                got = pd.concat([got, rest.sample(need,
                                random_state=rng.integers(1e9))])
        got = got.head(N_PER_LANGUAGE)
        picks.append(got)
        print(f"  {region}: {len(got)} sampled across {got['topic'].nunique()} topics, "
              f"{got['channel'].nunique()} channels")

    sample = pd.concat(picks, ignore_index=True)
    sample = sample.sample(frac=1, random_state=SEED).reset_index(drop=True)
    sample.insert(0, "row", range(1, len(sample) + 1))

    # file to label, without the model's predictions
    blind = sample[["row", "comment_id", "region", "topic", "channel",
                    "week", "text"]].copy()
    blind["label"] = ""
    blind["looks_saudi"] = np.where(blind["region"] == "SA", "", "n/a")
    blind["note"] = ""
    blind.to_csv(SAMPLE_CSV, index=False, encoding="utf-8-sig")

    # model predictions, kept separate
    key = sample[["row", "comment_id", "region", "topic", "channel",
                  "model_label", "p_negative", "p_neutral", "p_positive"]]
    key.to_csv(ANSWERS_CSV, index=False, encoding="utf-8-sig")

    print(f"\nWrote {SAMPLE_CSV}  ({len(blind)} rows)")
    print(f"Wrote {ANSWERS_CSV} (open after labelling)")
    print("\nlabel: positive, negative, neutral or unusable")
    print("looks_saudi (SA rows): yes, no or unclear")


if __name__ == "__main__":
    main()
