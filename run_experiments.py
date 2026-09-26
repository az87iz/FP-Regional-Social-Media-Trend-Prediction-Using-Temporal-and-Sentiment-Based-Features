"""
Runs the three experiments on the feature matrix.

    feature_matrix.csv -> experiment_results.csv, per-region results,
                          test-set predictions and DeLong tests

    Exp 1  temporal features only (baseline)
    Exp 2  temporal + sentiment
    Exp 3  temporal + sentiment + regional

How it is set up:
- All three experiments use the same rows and folds, since DeLong's test
  compares two models on the same test set.
- Validation is an expanding window over weeks, so a model is never
  trained on weeks after the ones it is tested on.
- The scaler and the sentiment normalisation are fitted on the training
  rows of each fold only.
- Models: logistic regression, random forest and XGBoost if installed.
- Baselines: majority class, and persistence (next week is a top-20% week
  if this week was).
- ROC-AUC is the main metric because only about 22% of rows are positive.
  F1 is reported as well.
- A gain counts if DeLong's p < 0.05 and it is at least 0.02 AUC.

Run:  python run_experiments.py
"""

import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    HAVE_XGB = True
except ImportError:
    HAVE_XGB = False

IN_CSV      = "feature_matrix.csv"
OUT_CSV     = "experiment_results.csv"
REGION_CSV  = "experiment_results_by_region.csv"
PRED_CSV    = "experiment_predictions.csv"

N_FOLDS   = 5
MIN_TRAIN = 0.35     # first fold trains on at least this share of weeks
MIN_GAIN  = 0.02     # smallest AUC gain treated as meaningful

TEMPORAL = ["gt_score", "velocity", "accel", "roll_mean", "roll_std",
            "vel_lag1", "vel_lag2", "is_top20"]
# Level features are z-scored per region inside each fold (normalise_fold).
# Delta features are changes, so they are used as they are.
SENT_LEVEL = ["sent_mean", "prop_pos", "prop_neg", "prop_neu", "sent_std"]
SENT_DELTA = ["sent_mean_delta", "prop_neg_delta", "prop_pos_delta"]
SENTIMENT = SENT_LEVEL + SENT_DELTA
REGIONAL = ["vel_vs_other_regions", "sent_vs_other_regions",
            "region_rank_velocity"]


def add_regional_features(df):
    """Regional features for Experiment 3: how a region compares with the
    other regions for the same topic and week."""
    df = df.copy()
    wk = df.groupby(["topic", "week"])

    # mean velocity of the other regions for the same topic-week
    df["_vel_sum"] = wk["velocity"].transform("sum")
    df["_vel_n"] = wk["velocity"].transform("count")
    other_mean = (df["_vel_sum"] - df["velocity"]) / (df["_vel_n"] - 1).replace(0, np.nan)
    df["vel_vs_other_regions"] = df["velocity"] - other_mean

    if "sent_mean" in df.columns:
        df["_s_sum"] = wk["sent_mean"].transform("sum")
        df["_s_n"] = wk["sent_mean"].transform("count")
        s_other = (df["_s_sum"] - df["sent_mean"]) / (df["_s_n"] - 1).replace(0, np.nan)
        df["sent_vs_other_regions"] = df["sent_mean"] - s_other
    else:
        df["sent_vs_other_regions"] = np.nan

    # rank of this region's velocity within the topic-week
    df["region_rank_velocity"] = wk["velocity"].rank(pct=True)

    # rows with no other region to compare against get 0
    for c in REGIONAL:
        df[c] = df[c].fillna(0.0)
    return df.drop(columns=[c for c in df.columns if c.startswith("_")])


def normalise_fold(tr, te, cols):
    """Z-scores `cols` within each region using the training rows only.

    Mean sentiment differs by about 0.75 between regions, so without this a
    pooled model could tell the region apart from the sentiment level. The
    mean and standard deviation come from the training part of the fold so
    no test-period information is used.
    """
    tr, te = tr.copy(), te.copy()
    for c in cols:
        stats_ = tr.groupby("geo")[c].agg(["mean", "std"])
        for part in (tr, te):
            mu = part["geo"].map(stats_["mean"])
            sd = part["geo"].map(stats_["std"]).replace(0, np.nan)
            part[c] = (part[c] - mu) / sd
        # a region missing from the training part gets 0
        tr[c] = tr[c].fillna(0.0)
        te[c] = te[c].fillna(0.0)
    return tr, te


def expanding_folds(weeks, n_folds=N_FOLDS, min_train=MIN_TRAIN):
    """Expanding-window splits over ordered unique weeks."""
    uw = np.array(sorted(pd.unique(weeks)))
    n = len(uw)
    start = int(n * min_train)
    if start < 2 or n - start < n_folds:
        n_folds = max(1, min(n_folds, n - start))
    edges = np.linspace(start, n, n_folds + 1).astype(int)
    folds = []
    for i in range(len(edges) - 1):
        tr_end, te_end = edges[i], edges[i + 1]
        if te_end <= tr_end:
            continue
        folds.append((uw[:tr_end], uw[tr_end:te_end]))
    return folds


def make_models(pos_weight):
    m = {
        "logreg": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "rf": RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                     class_weight="balanced", random_state=42,
                                     n_jobs=-1),
    }
    if HAVE_XGB:
        m["xgboost"] = XGBClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
            scale_pos_weight=pos_weight, eval_metric="logloss",
            random_state=42, n_jobs=-1)
    return m


def run_experiment(df, feats, name):
    """Fits each model fold by fold and returns its test-set predictions."""
    folds = expanding_folds(df["week"])
    preds = {k: [] for k in make_models(1.0)}
    truth, meta = [], []

    for tr_weeks, te_weeks in folds:
        tr = df[df["week"].isin(tr_weeks)]
        te = df[df["week"].isin(te_weeks)]
        if len(te) == 0 or tr["target"].nunique() < 2:
            continue

        # per-region sentiment normalisation from the training rows
        lvl = [c for c in feats if c in SENT_LEVEL]
        if lvl:
            tr, te = normalise_fold(tr, te, lvl)

        Xtr_raw, Xte_raw = tr[feats].values, te[feats].values
        sc = StandardScaler().fit(Xtr_raw)
        Xtr, Xte = sc.transform(Xtr_raw), sc.transform(Xte_raw)
        ytr, yte = tr["target"].values, te["target"].values

        pw = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
        for mname, model in make_models(pw).items():
            model.fit(Xtr, ytr)
            preds[mname].append(model.predict_proba(Xte)[:, 1])

        truth.append(yte)
        meta.append(te[["geo", "topic", "week"]])

    if not truth:
        return None
    y = np.concatenate(truth)
    out = {k: np.concatenate(v) for k, v in preds.items() if v}
    meta = pd.concat(meta, ignore_index=True)
    meta["y_true"] = y
    for k, v in out.items():
        meta[f"p_{k}"] = v
    meta["experiment"] = name
    return meta


def delong_test(y, p1, p2):
    """DeLong's test for two correlated ROC-AUCs.

    Midrank version from Sun and Xu (2014). Returns (auc1, auc2, z, p).
    """
    y = np.asarray(y)
    pos = y == 1
    neg = ~pos
    m, n = pos.sum(), neg.sum()
    if m == 0 or n == 0:
        return np.nan, np.nan, np.nan, np.nan

    def structural(p):
        x, yv = np.asarray(p)[pos], np.asarray(p)[neg]
        # midrank-based V components
        tx = np.array([(np.sum(yv < xi) + 0.5 * np.sum(yv == xi)) / n for xi in x])
        ty = np.array([(np.sum(x > yj) + 0.5 * np.sum(x == yj)) / m for yj in yv])
        auc = tx.mean()
        return auc, tx, ty

    a1, tx1, ty1 = structural(p1)
    a2, tx2, ty2 = structural(p2)

    s10 = np.cov(np.vstack([tx1, tx2]))
    s01 = np.cov(np.vstack([ty1, ty2]))
    S = s10 / m + s01 / n
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    if var <= 0:
        return a1, a2, np.nan, np.nan
    z = (a1 - a2) / np.sqrt(var)
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return a1, a2, z, p


def metrics(y, p):
    if len(np.unique(y)) < 2:
        return np.nan, np.nan
    return roc_auc_score(y, p), f1_score(y, (p >= 0.5).astype(int),
                                         zero_division=0)


def main():
    try:
        df = pd.read_csv(IN_CSV, parse_dates=["week"])
    except FileNotFoundError:
        sys.exit(f"{IN_CSV} not found. Run build_matrix.py first.")

    df = add_regional_features(df)
    df = df.sort_values("week").reset_index(drop=True)

    # The delta features are NaN on the first week of each series. Rows with
    # any missing feature are dropped here, once, using every feature from
    # all three experiments, so each experiment gets exactly the same rows.
    all_feats = list(dict.fromkeys(TEMPORAL + SENTIMENT + REGIONAL))
    present = [f for f in all_feats if f in df.columns]
    before = len(df)
    na_by_col = df[present].isna().sum()
    df = df.dropna(subset=present).reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        print(f"Dropped {dropped} rows with missing features:")
        for c, n in na_by_col[na_by_col > 0].items():
            print(f"    {c}: {n} NaN")

    print(f"\nRows: {len(df):,}  weeks: {df.week.nunique()}  "
          f"positive: {df.target.mean():.1%}")
    print(f"XGBoost available: {HAVE_XGB}")
    folds = expanding_folds(df["week"])
    print(f"Expanding-window folds: {len(folds)}")
    for i, (tr, te) in enumerate(folds, 1):
        print(f"  fold {i}: train {len(tr)} weeks, test {len(te)} weeks "
              f"({str(te[0])[:10]} to {str(te[-1])[:10]})")

    sets = {
        "Exp1_temporal": TEMPORAL,
        "Exp2_temporal_sentiment": TEMPORAL + SENTIMENT,
        "Exp3_temporal_sentiment_regional": TEMPORAL + SENTIMENT + REGIONAL,
    }

    all_preds, rows = [], []
    for name, feats in sets.items():
        missing = [f for f in feats if f not in df.columns]
        if missing:
            print(f"\n{name}: missing {missing}, skipped")
            continue
        res = run_experiment(df, feats, name)
        if res is None:
            print(f"\n{name}: no usable folds")
            continue
        all_preds.append(res)
        print(f"\n=== {name}  ({len(feats)} features, "
              f"{len(res)} test predictions) ===")
        for m in [c[2:] for c in res.columns if c.startswith("p_")]:
            auc, f1 = metrics(res["y_true"], res[f"p_{m}"])
            print(f"  {m:<9} ROC-AUC {auc:.3f}   F1 {f1:.3f}")
            rows.append({"experiment": name, "model": m,
                         "n_test": len(res), "roc_auc": round(auc, 4),
                         "f1": round(f1, 4)})

    if not all_preds:
        sys.exit("No experiments ran.")

    P = pd.concat(all_preds, ignore_index=True)
    P.to_csv(PRED_CSV, index=False)

    # baselines
    base = P[P.experiment == "Exp1_temporal"]
    y = base["y_true"].values
    maj_auc = 0.5
    pers = base.merge(df[["geo", "topic", "week", "is_top20"]],
                      on=["geo", "topic", "week"], how="left")["is_top20"].fillna(0)
    pers_auc, pers_f1 = metrics(y, pers.values.astype(float))
    print("\n=== Baselines ===")
    print(f"  majority class   ROC-AUC {maj_auc:.3f}")
    print(f"  persistence      ROC-AUC {pers_auc:.3f}   F1 {pers_f1:.3f}")
    rows.append({"experiment": "baseline_majority", "model": "-",
                 "n_test": len(y), "roc_auc": maj_auc, "f1": 0.0})
    rows.append({"experiment": "baseline_persistence", "model": "-",
                 "n_test": len(y), "roc_auc": round(pers_auc, 4),
                 "f1": round(pers_f1, 4)})

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

    # results per region
    reg_rows = []
    for (exp, geo), g in P.groupby(["experiment", "geo"]):
        for m in [c[2:] for c in P.columns if c.startswith("p_")]:
            auc, f1 = metrics(g["y_true"], g[f"p_{m}"])
            reg_rows.append({"experiment": exp, "region": geo, "model": m,
                             "n": len(g), "roc_auc": round(auc, 4) if auc == auc else None,
                             "f1": round(f1, 4) if f1 == f1 else None})
    rdf = pd.DataFrame(reg_rows)
    rdf.to_csv(REGION_CSV, index=False)
    print("\n=== Per region ===")
    print(rdf.pivot_table(index=["experiment", "model"], columns="region",
                          values="roc_auc").round(3).to_string())

    # does each added feature group improve on the previous one?
    print(f"\n=== DeLong tests (p < 0.05, gain of at least {MIN_GAIN}) ===\n")

    models = [c[2:] for c in P.columns if c.startswith("p_")]
    comparisons = [("Exp1_temporal", "Exp2_temporal_sentiment"),
                   ("Exp2_temporal_sentiment", "Exp3_temporal_sentiment_regional")]
    for a, b in comparisons:
        if a not in set(P.experiment) or b not in set(P.experiment):
            continue
        A = P[P.experiment == a].reset_index(drop=True)
        B = P[P.experiment == b].reset_index(drop=True)
        if len(A) != len(B) or not (A["y_true"].values == B["y_true"].values).all():
            print(f"{a} vs {b}: test sets differ, skipped")
            continue
        print(f"--- {b}  vs  {a} ---")
        for m in models:
            auc_b, auc_a, z, p = delong_test(A["y_true"].values,
                                             B[f"p_{m}"].values, A[f"p_{m}"].values)
            gain = auc_b - auc_a
            sig = "significant" if (p == p and p < 0.05) else "not significant"
            meaningful = "and >= 0.02" if gain >= MIN_GAIN else "but < 0.02"
            print(f"  {m:<9} {auc_a:.3f} -> {auc_b:.3f}  "
                  f"gain {gain:+.3f}  p={p:.4f}  ({sig}, {meaningful})")
        print()

    print(f"Wrote {OUT_CSV}, {REGION_CSV}, {PRED_CSV}")


if __name__ == "__main__":
    main()
