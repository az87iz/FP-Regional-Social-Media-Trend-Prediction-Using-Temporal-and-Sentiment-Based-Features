"""
Makes a prediction from a saved regional model.

    python predict.py --region SA --topic weather
    python predict.py --all
    python predict.py --all --csv predictions.csv

For a region and topic, it gives the probability that the topic's weekly
search growth will be in that region's top 20% in the week after the
latest week in the data. The top-20% threshold is the one fixed per region
in build_matrix.py.

The feature list is read from the saved model file, so it always matches
what the model was trained on.

This is not a live forecast. It uses the latest week in feature_matrix.csv,
and a forecast for the current week would need that week's Trends data
collected first. Held-out performance was around 0.59 ROC-AUC, so the
probabilities are a weak signal.
"""

import argparse
import os
import sys

import pandas as pd

try:
    import joblib
except ImportError:
    sys.exit("Run: pip install joblib")

IN_CSV = "feature_matrix.csv"
MODEL_DIR = "models"


def load_model(region):
    path = os.path.join(MODEL_DIR, f"{region}_model.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No model for region '{region}'. Expected {path}. "
            f"Run train_final_model.py first.")
    return joblib.load(path)


def latest_row(df, region, topic):
    sub = df[(df["geo"] == region) & (df["topic"] == topic)]
    if sub.empty:
        available = sorted(df[df["geo"] == region]["topic"].unique())
        raise ValueError(
            f"No data for topic '{topic}' in region '{region}'.\n"
            f"Available topics for {region}: {', '.join(available) or 'none'}")
    return sub.sort_values("week").iloc[-1]


def predict_one(df, art, region, topic):
    row = latest_row(df, region, topic)
    feats = art["features"]

    missing = [f for f in feats if f not in row.index or pd.isna(row[f])]
    if missing:
        raise ValueError(
            f"Cannot predict {region}/{topic}: the most recent week is missing "
            f"values for {missing}.")

    x = pd.DataFrame([[float(row[f]) for f in feats]], columns=feats)

    # same transforms as training, from the saved file
    for c, s in art.get("sentiment_stats", {}).items():
        if c in x.columns:
            x[c] = (x[c] - s["mean"]) / s["std"]
    xs = art["scaler"].transform(x.values)

    prob = float(art["estimator"].predict_proba(xs)[0, 1])
    return {
        "region": region,
        "topic": topic,
        "based_on_week": pd.Timestamp(row["week"]).date().isoformat(),
        "prediction_for_week": (pd.Timestamp(row["week"]) +
                                pd.Timedelta(days=7)).date().isoformat(),
        "probability_top20": round(prob, 4),
        "predicted_label": int(prob >= 0.5),
        "model": art["model_name"],
        "n_features": len(feats),
    }


def main():
    ap = argparse.ArgumentParser(
        description="Predict whether a topic enters the top 20% of weekly "
                    "search growth in a region next week.")
    ap.add_argument("--region", help="region code, e.g. SA or US")
    ap.add_argument("--topic", help="topic name, e.g. weather")
    ap.add_argument("--all", action="store_true",
                    help="predict every region-topic pair with a model")
    ap.add_argument("--csv", help="write results to this CSV")
    ap.add_argument("--data", default=IN_CSV, help=f"feature file ({IN_CSV})")
    args = ap.parse_args()

    if not args.all and not (args.region and args.topic):
        ap.error("give --region and --topic, or --all")
    if not os.path.exists(args.data):
        sys.exit(f"{args.data} not found. Run build_matrix.py first.")

    df = pd.read_csv(args.data, parse_dates=["week"])

    if args.all:
        results, skipped = [], []
        for region in sorted(df["geo"].unique()):
            try:
                art = load_model(region)
            except FileNotFoundError:
                skipped.append((region, "*", "no model for this region"))
                continue
            for topic in sorted(df[df["geo"] == region]["topic"].unique()):
                try:
                    results.append(predict_one(df, art, region, topic))
                except ValueError as e:
                    skipped.append((region, topic, str(e).split("\n")[0]))

        if not results:
            sys.exit("No predictions could be made.")
        out = pd.DataFrame(results).sort_values(
            ["region", "probability_top20"], ascending=[True, False])

        print("\nRegional trend predictions")
        print("=" * 72)
        for region, g in out.groupby("region"):
            wk = g["prediction_for_week"].max()
            print(f"\n{region}  (week beginning {wk})")
            print(f"  {'topic':<26}{'P(top 20%)':>12}   forecast")
            for _, r in g.iterrows():
                flag = "TOP 20%" if r["predicted_label"] else "-"
                print(f"  {r['topic']:<26}{r['probability_top20']:>12.3f}   {flag}")

        if skipped:
            print(f"\nNot predicted ({len(skipped)}):")
            for region, topic, why in skipped:
                print(f"  {region}/{topic}: {why}")

        if args.csv:
            out.to_csv(args.csv, index=False)
            print(f"\nWrote {args.csv}")
    else:
        try:
            art = load_model(args.region)
        except FileNotFoundError as e:
            have = sorted(f.split("_model")[0] for f in os.listdir(MODEL_DIR)
                          if f.endswith("_model.joblib")) if os.path.isdir(MODEL_DIR) else []
            sys.exit(f"{e}\nRegions with a trained model: "
                     f"{', '.join(have) if have else 'none'}")
        try:
            r = predict_one(df, art, args.region, args.topic)
        except ValueError as e:
            sys.exit(str(e))
        print(f"\nRegion            {r['region']}")
        print(f"Topic             {r['topic']}")
        print(f"Based on week     {r['based_on_week']}")
        print(f"Prediction for    week beginning {r['prediction_for_week']}")
        print(f"P(top 20% growth) {r['probability_top20']:.3f}")
        print(f"Forecast          {'IN the top 20%' if r['predicted_label'] else 'NOT in the top 20%'}")
        print(f"Model             {r['model']}, {r['n_features']} features")

    print("\nNote: evaluated held-out performance is around 0.59 ROC-AUC, so")
    print("these probabilities indicate a weak signal, not a reliable forecast.")


if __name__ == "__main__":
    main()
