"""
Writes comment_ids.csv for the repository.

The comment text itself is not shared. This file lists the ID of every
scored comment along with its video, channel, region, topic and week, so
the comments can be fetched again through the YouTube Data API.

Run from the folder with yt_comments.csv and yt_comments_scored.csv:
    python make_comment_ids.py
"""
import os

import pandas as pd

comments = pd.read_csv("data/raw/yt_comments.csv", encoding="utf-8-sig", low_memory=False,
                       usecols=["comment_id", "video_id", "channel",
                                "region", "topic", "week"])
scored = pd.read_csv("data/raw/yt_comments_scored.csv", encoding="utf-8-sig",
                     usecols=["comment_id"])

ids = (comments.drop_duplicates("comment_id")
               .merge(scored.drop_duplicates("comment_id"), on="comment_id"))
ids = ids.sort_values(["region", "topic", "week"]).reset_index(drop=True)

ids.to_csv("data/comment_ids.csv", index=False)
size_mb = os.path.getsize("data/comment_ids.csv") / 1e6

# GitHub's web uploader rejects files over 25 MB
if size_mb > 24:
    ids.to_csv("data/comment_ids.csv.gz", index=False, compression="gzip")
    os.remove("data/comment_ids.csv")
    out, size_mb = "data/comment_ids.csv.gz", os.path.getsize("comment_ids.csv.gz") / 1e6
else:
    out = "data/comment_ids.csv"

print(f"Wrote {out}: {len(ids):,} comments, {size_mb:.1f} MB")
print(ids.groupby("region").size().to_string())
