"""
Trains the final model for each region and saves it.

    feature_matrix.csv -> models/<region>_model.joblib

run_experiments.py only fits models inside the evaluation folds. This
script fits one model per region on all the data and saves it with what
predict.py needs to rebuild its inputs.

Choices:
- Temporal features only. The experiments found no established gain from
  sentiment in either region, so it is not included.
- One model per region, since the label threshold is set per region and
  the temporal baseline is different for Saudi Arabia and the US.
- Logistic regression for both. The three model types were very close in
  the pooled results (0.588, 0.586, 0.545), so the simplest one to
  interpret was used.

Each saved file holds the fitted model and scaler, the sentiment
normalisation values (empty here), the ordered feature list, and the
training rows, dates and library versions. The feature list is stored in
the file so predict.py always uses the same features as training.

Run:  python src/train_final_model.py
"""

import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

try:
    import joblib
except ImportError:
    sys.exit("Run: pip install joblib")

IN_CSV = "data/feature_matrix.csv"
MODEL_DIR = "models"

TEMPORAL = ["gt_score", "velocity", "accel", "roll_mean", "roll_std",
            "vel_lag1", "vel_lag2", "is_top20"]
SENT_LEVEL = ["sent_mean", "prop_pos", "prop_neg", "prop_neu", "sent_std"]
SENT_DELTA = ["sent_mean_delta", "prop_neg_delta", "prop_pos_delta"]

# features and model per region
REGION_CONFIG = {
    "SA": {"features": TEMPORAL,
           "model": "logreg",
           "reason": "no established sentiment gain (-0.036, +0.037, +0.022)"},
    "US": {"features": TEMPORAL,
           "model": "logreg",
           "reason": "no established sentiment gain (-0.007, -0.022, -0.009)"},
}


def fit_sentiment_normaliser(df, cols):
    """Mean and std for the sentiment level features.

    During evaluation these came from each fold's training rows. Here they
    are fitted on all the training data and saved so predict.py applies the
    same transform.
    """
    stats = {}
    for c in cols:
        s = df[c].astype(float)
        mu = float(s.mean())
        sd = float(s.std())
        stats[c] = {"mean": mu, "std": sd if sd and sd > 0 else 1.0}
    return stats


def apply_sentiment_normaliser(df, stats):
    df = df.copy()
    for c, s in stats.items():
        if c in df.columns:
            df[c] = (df[c].astype(float) - s["mean"]) / s["std"]
    return df


def main():
    if not os.path.exists(IN_CSV):
        sys.exit(f"{IN_CSV} not found. Run build_matrix.py first.")
    df = pd.read_csv(IN_CSV, parse_dates=["week"])
    os.makedirs(MODEL_DIR, exist_ok=True)

    print(f"Loaded {len(df):,} rows, {df.week.nunique()} weeks, "
          f"{df.week.min().date()} to {df.week.max().date()}\n")

    summary = []
    for region, cfg in REGION_CONFIG.items():
        sub = df[df["geo"] == region].copy()
        feats = [f for f in cfg["features"] if f in sub.columns]
        missing = [f for f in cfg["features"] if f not in sub.columns]
        if missing:
            print(f"[{region}] missing features, skipped: {missing}")

        sub = sub.dropna(subset=feats + ["target"])
        if len(sub) == 0 or sub["target"].nunique() < 2:
            print(f"[{region}] not enough usable data, skipped")
            continue

        # sentiment normalisation (not used while the features are temporal only)
        lvl = [f for f in feats if f in SENT_LEVEL]
        sent_stats = fit_sentiment_normaliser(sub, lvl) if lvl else {}
        sub_n = apply_sentiment_normaliser(sub, sent_stats)

        X = sub_n[feats].values.astype(float)
        y = sub_n["target"].values.astype(int)

        scaler = StandardScaler().fit(X)
        Xs = scaler.transform(X)

        if cfg["model"] == "rf":
            est = RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                         class_weight="balanced",
                                         random_state=42, n_jobs=-1)
        else:
            est = LogisticRegression(max_iter=2000, class_weight="balanced")
        est.fit(Xs, y)

        artefact = {
            "region": region,
            "model_name": cfg["model"],
            "estimator": est,
            "scaler": scaler,
            "sentiment_stats": sent_stats,
            "features": feats,                 # predict.py uses this order
            "feature_reason": cfg["reason"],
            "n_train_rows": int(len(sub)),
            "positive_rate": float(y.mean()),
            "train_weeks": [str(sub.week.min().date()), str(sub.week.max().date())],
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "sklearn_version": __import__("sklearn").__version__,
            "pandas_version": pd.__version__,
        }
        path = os.path.join(MODEL_DIR, f"{region}_model.joblib")
        joblib.dump(artefact, path)

        print(f"[{region}] {cfg['model']} on {len(feats)} features, "
              f"{len(sub)} rows, {y.mean():.1%} positive")
        print(f"         {cfg['reason']}")
        print(f"         saved to {path}\n")
        summary.append({"region": region, "model": cfg["model"],
                        "n_features": len(feats), "n_rows": int(len(sub)),
                        "positive_rate": round(float(y.mean()), 4),
                        "artefact": path})

    if not summary:
        sys.exit("No models were trained.")

    with open(os.path.join(MODEL_DIR, "manifest.json"), "w") as f:
        json.dump({"models": summary,
                   "built_from": IN_CSV,
                   "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds")},
                  f, indent=2)

    print(pd.DataFrame(summary).to_string(index=False))
    print("\nThese are trained on all the data. The evaluation figures come from")
    print("the held-out folds in run_experiments.py.")


if __name__ == "__main__":
    main()
