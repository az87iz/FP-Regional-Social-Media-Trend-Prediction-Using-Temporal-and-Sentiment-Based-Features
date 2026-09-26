"""
Compares the hand labels in label_sample.csv with the model's predictions.
Run it once every row has a label.

Outputs:
  1. a confusion matrix for each language, with per-class precision and recall
  2. how often each predicted class is actually negative, as a check on
     whether model error could explain the gap in sentiment between regions
  3. the share of Saudi-channel comments that look Saudi (looks_saudi)

    python score_labels.py
"""

import os
import sys

import numpy as np
import pandas as pd

SAMPLE_CSV  = "label_sample.csv"
ANSWERS_CSV = "label_answers.csv"
OUT_CSV     = "validation_results.csv"
CM_CSV      = "confusion_matrices.csv"

# accuracy on the UMSAB tweets from the prototype, for comparison
UMSAB = {"SA": 0.670, "US": 0.732}
VALID = {"positive", "negative", "neutral", "unusable"}
CLASSES = ["negative", "neutral", "positive"]

# mean sentiment by region from sentiment_features.py
MEASURED_SENTIMENT = {"SA": 0.259, "US": -0.371}


def confusion_matrix(y_true, y_pred, classes):
    """Rows are the hand label, columns are the model's label."""
    cm = pd.DataFrame(0, index=classes, columns=classes)
    for t, p in zip(y_true, y_pred):
        cm.loc[t, p] += 1
    return cm


def main():
    for f in (SAMPLE_CSV, ANSWERS_CSV):
        if not os.path.exists(f):
            sys.exit(f"{f} not found.")

    s = pd.read_csv(SAMPLE_CSV, encoding="utf-8-sig")
    k = pd.read_csv(ANSWERS_CSV, encoding="utf-8-sig")

    s["label"] = s["label"].astype(str).str.strip().str.lower()
    bad = set(s[~s["label"].isin(VALID)]["label"]) - {"nan", ""}
    if bad:
        sys.exit(f"Unrecognised labels: {bad}\nUse: {sorted(VALID)}")
    blank = (s["label"].isin(["", "nan"])).sum()
    if blank:
        sys.exit(f"{blank} rows still have no label.")

    df = s.merge(k[["row", "model_label"]], on="row", how="inner")
    n_unusable = (df["label"] == "unusable").sum()
    df = df[df["label"] != "unusable"].copy()
    print(f"Labelled: {len(df)} usable ({n_unusable} marked unusable)\n")
    df["correct"] = df["label"] == df["model_label"]

    # 1. confusion matrix for each language
    print("=== Confusion matrices (rows = hand label, columns = model) ===")
    cms = {}
    acc_rows = []
    for region in sorted(df["region"].unique()):
        d = df[df["region"] == region]
        lang = "Arabic" if region == "SA" else "English"
        cm = confusion_matrix(d["label"], d["model_label"], CLASSES)
        cms[region] = cm
        acc = d["correct"].mean()

        print(f"\n{region} ({lang}), n={len(d)}, overall accuracy {acc:.3f} "
              f"(UMSAB tweets: {UMSAB[region]:.3f})")
        print(cm.to_string())

        print("\n  per class:")
        for cls in CLASSES:
            true_n = cm.loc[cls].sum()
            pred_n = cm[cls].sum()
            tp = cm.loc[cls, cls]
            recall = tp / true_n if true_n else float("nan")
            precision = tp / pred_n if pred_n else float("nan")
            print(f"    {cls:<9} true n={true_n:<4} recall={recall:.3f}  "
                  f"pred n={pred_n:<4} precision={precision:.3f}")
            acc_rows.append({"region": region, "language": lang, "class": cls,
                             "true_n": int(true_n), "recall": round(recall, 4)
                             if recall == recall else None,
                             "pred_n": int(pred_n),
                             "precision": round(precision, 4)
                             if precision == precision else None})

    cm_out = pd.concat({r: cms[r] for r in cms}, names=["region", "true_label"])
    cm_out.to_csv(CM_CSV)
    print(f"\nWrote {CM_CSV}")

    # 2. sensitivity check. The regions differ a lot in mean sentiment, so
    # this shows, for each class the model predicts, how the hand labels are
    # spread across the true classes.
    print("\n=== Sensitivity check ===")
    print(f"Mean sentiment: SA {MEASURED_SENTIMENT['SA']:+.3f}, "
          f"US {MEASURED_SENTIMENT['US']:+.3f}\n")
    for region in sorted(df["region"].unique()):
        cm = cms[region]
        d = df[df["region"] == region]
        lang = "Arabic" if region == "SA" else "English"

        # P(true class | predicted class)
        pred_totals = cm.sum(axis=0)
        cond = cm.div(pred_totals, axis=1).fillna(0)   # columns sum to 1

        print(f"{region} ({lang}):")
        print("  P(true class | predicted class):")
        print(cond.round(3).to_string())

        print(f"\n  predicted positive but actually negative: "
              f"{cond.loc['negative','positive']:.1%}")
        print(f"  predicted neutral but actually negative:  "
              f"{cond.loc['negative','neutral']:.1%}")
        print()

    # With about 50 comments per language the sample is too small to correct
    # the corpus mean reliably, so only the direction and rough size of the
    # error are reported.

    # 3. how many Saudi-channel comments look Saudi
    sa = df[df["region"] == "SA"]
    if "looks_saudi" in sa.columns and sa["looks_saudi"].notna().any():
        print("\n=== looks_saudi (SA rows) ===")
        vc = sa["looks_saudi"].astype(str).str.strip().str.lower().value_counts()
        print(vc.to_string())
        yes = vc.get("yes", 0)
        total = vc.sum()
        if total:
            print(f"\n{yes}/{total} ({yes/total:.1%}) look Saudi. This is a lower "
                  f"bound, since not every Saudi comment shows it.")

    pd.DataFrame(acc_rows).to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV}")


if __name__ == "__main__":
    main()
