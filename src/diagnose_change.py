"""
Checks why the Saudi result changed between the two collection rounds.

The draft (round 1) found a significant Saudi gain from sentiment (logreg
+0.150, rf +0.242). With about 2.7 times more Saudi data in round 2 the gain
went away (-0.036, +0.037, +0.022, none significant). Several things changed
together between the rounds:
  - Saudi rows went from about 100 to 261
  - AI and inflation were added to the Saudi topics, which had only been
    weather, Ramadan and World Cup
  - Saudi comments now come from four channels, where before there was one

To separate these, Exp 1 and Exp 2 are compared again on subsets of the
round 2 data:
  A. Saudi, original three topics only
  B. Saudi, the two new topics only
  C. Saudi, all five topics (the reported result)
  D. Saudi, original topics, cut down to the round 1 row count, repeated
     over ten random draws

The folds, normalisation and feature lists are the same as in
run_experiments.py. The subsets are smaller, so these results are only
used to explain the change and are not a separate test.

Run:  python diagnose_change.py
"""

import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

IN_CSV = "data/feature_matrix.csv"
OUT_CSV = "diagnosis_results.csv"

ORIGINAL_SA_TOPICS = ["weather", "Ramadan", "World Cup"]
NEW_SA_TOPICS = ["artificial intelligence", "inflation"]
ROUND1_SA_ROWS = 99          # Saudi rows in the round 1 matrix

TEMPORAL = ["gt_score", "velocity", "accel", "roll_mean", "roll_std",
            "vel_lag1", "vel_lag2", "is_top20"]
SENT_LEVEL = ["sent_mean", "prop_pos", "prop_neg", "prop_neu", "sent_std"]
SENT_DELTA = ["sent_mean_delta", "prop_neg_delta", "prop_pos_delta"]
SENTIMENT = SENT_LEVEL + SENT_DELTA

N_FOLDS, MIN_TRAIN = 5, 0.35


def expanding_folds(weeks, n_folds=N_FOLDS, min_train=MIN_TRAIN):
    uw = np.array(sorted(pd.unique(weeks)))
    n = len(uw)
    start = int(n * min_train)
    if start < 2 or n - start < 2:
        return []
    n_folds = max(1, min(n_folds, n - start))
    edges = np.linspace(start, n, n_folds + 1).astype(int)
    out = []
    for i in range(len(edges) - 1):
        if edges[i + 1] > edges[i]:
            out.append((uw[:edges[i]], uw[edges[i]:edges[i + 1]]))
    return out


def normalise_fold(tr, te, cols):
    tr, te = tr.copy(), te.copy()
    for c in cols:
        st = tr.groupby("geo")[c].agg(["mean", "std"])
        for part in (tr, te):
            mu = part["geo"].map(st["mean"])
            sd = part["geo"].map(st["std"]).replace(0, np.nan)
            part[c] = (part[c] - mu) / sd
        tr[c] = tr[c].fillna(0.0)
        te[c] = te[c].fillna(0.0)
    return tr, te


def run(df, feats, seed=42):
    """Test-set probabilities from the expanding-window folds."""
    folds = expanding_folds(df["week"])
    if not folds:
        return None, None
    preds = {"logreg": [], "rf": []}
    truth = []
    for trw, tew in folds:
        tr, te = df[df.week.isin(trw)], df[df.week.isin(tew)]
        if len(te) == 0 or tr["target"].nunique() < 2:
            continue
        lvl = [c for c in feats if c in SENT_LEVEL]
        if lvl:
            tr, te = normalise_fold(tr, te, lvl)
        sc = StandardScaler().fit(tr[feats].values)
        Xtr, Xte = sc.transform(tr[feats].values), sc.transform(te[feats].values)
        ytr = tr["target"].values
        models = {
            "logreg": LogisticRegression(max_iter=2000, class_weight="balanced"),
            "rf": RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                         class_weight="balanced",
                                         random_state=seed, n_jobs=-1)}
        for k, mdl in models.items():
            mdl.fit(Xtr, ytr)
            preds[k].append(mdl.predict_proba(Xte)[:, 1])
        truth.append(te["target"].values)
    if not truth:
        return None, None
    y = np.concatenate(truth)
    if len(np.unique(y)) < 2:
        return None, None
    return y, {k: np.concatenate(v) for k, v in preds.items() if v}


def delong(y, p1, p2):
    y = np.asarray(y)
    pos, neg = y == 1, y == 0
    m, n = pos.sum(), neg.sum()
    if m == 0 or n == 0:
        return np.nan, np.nan, np.nan

    def comp(p):
        p = np.asarray(p, float)
        x, yv = p[pos], p[neg]
        tx = np.array([(np.sum(yv < xi) + .5 * np.sum(yv == xi)) / n for xi in x])
        ty = np.array([(np.sum(x > yj) + .5 * np.sum(x == yj)) / m for yj in yv])
        return tx.mean(), tx, ty

    a1, tx1, ty1 = comp(p1)
    a2, tx2, ty2 = comp(p2)
    S = np.cov(np.vstack([tx1, tx2])) / m + np.cov(np.vstack([ty1, ty2])) / n
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    if var <= 0:
        return a1, a2, np.nan
    z = (a1 - a2) / np.sqrt(var)
    return a1, a2, 2 * (1 - stats.norm.cdf(abs(z)))


def compare(df, label, note=""):
    y1, p1 = run(df, TEMPORAL)
    y2, p2 = run(df, TEMPORAL + SENTIMENT)
    if y1 is None or y2 is None or len(y1) != len(y2):
        print(f"  {label:<38} not enough data for a walk-forward comparison")
        return []
    rows = []
    print(f"  {label:<38} rows={len(df):<5} test={len(y1):<5} pos={int(y1.sum())}")
    for mdl in ("logreg", "rf"):
        a2, a1, p = delong(y1, p2[mdl], p1[mdl])
        gain = a2 - a1
        flag = "*" if (p == p and p < 0.05 and gain >= 0.02) else " "
        print(f"      {mdl:<8} Exp1 {a1:.3f} -> Exp2 {a2:.3f}   "
              f"gain {gain:+.3f}  p={p:.3f} {flag}")
        rows.append({"subset": label, "note": note, "model": mdl,
                     "n_rows": len(df), "n_test": len(y1),
                     "n_positive": int(y1.sum()),
                     "auc_exp1": round(a1, 4), "auc_exp2": round(a2, 4),
                     "gain": round(gain, 4),
                     "p": round(p, 4) if p == p else None})
    return rows


def main():
    try:
        df = pd.read_csv(IN_CSV, parse_dates=["week"])
    except FileNotFoundError:
        sys.exit(f"{IN_CSV} not found. Run build_matrix.py first.")

    need = TEMPORAL + SENTIMENT + ["target"]
    df = df.dropna(subset=[c for c in need if c in df.columns])
    sa = df[df.geo == "SA"].sort_values("week")

    print(f"Saudi rows available: {len(sa)}  "
          f"topics: {sorted(sa.topic.unique())}\n")
    results = []

    print("A. Saudi, original three topics")
    a = sa[sa.topic.isin(ORIGINAL_SA_TOPICS)]
    results += compare(a, "SA original topics", "weather, Ramadan, World Cup")

    print("\nB. Saudi, two new topics")
    b = sa[sa.topic.isin(NEW_SA_TOPICS)]
    results += compare(b, "SA new topics", "AI, inflation")

    print("\nC. Saudi, all five topics")
    results += compare(sa, "SA all topics", "all topics")

    print(f"\nD. Saudi, original topics, downsampled to about {ROUND1_SA_ROWS} rows")
    print("   ten random draws of whole weeks, in time order")
    rng = np.random.default_rng(42)
    gains = {"logreg": [], "rf": []}
    for i in range(10):
        weeks = np.array(sorted(a.week.unique()))
        keep_n = max(8, int(len(weeks) * ROUND1_SA_ROWS / max(len(a), 1)))
        keep = np.sort(rng.choice(weeks, size=min(keep_n, len(weeks)),
                                  replace=False))
        sub = a[a.week.isin(keep)]
        y1, p1 = run(sub, TEMPORAL)
        y2, p2 = run(sub, TEMPORAL + SENTIMENT)
        if y1 is None or y2 is None or len(y1) != len(y2):
            continue
        for mdl in ("logreg", "rf"):
            try:
                gains[mdl].append(roc_auc_score(y1, p2[mdl]) -
                                  roc_auc_score(y1, p1[mdl]))
            except ValueError:
                pass
    for mdl in ("logreg", "rf"):
        g = gains[mdl]
        if g:
            print(f"      {mdl:<8} mean gain {np.mean(g):+.3f}   "
                  f"range [{min(g):+.3f}, {max(g):+.3f}]   n={len(g)} draws")
            results.append({"subset": "SA downsampled", "note": f"{len(g)} draws",
                            "model": mdl, "n_rows": ROUND1_SA_ROWS,
                            "n_test": None, "n_positive": None,
                            "auc_exp1": None, "auc_exp2": None,
                            "gain": round(float(np.mean(g)), 4), "p": None})

    if results:
        pd.DataFrame(results).to_csv(OUT_CSV, index=False)

    print(f"\nWrote {OUT_CSV}")


if __name__ == "__main__":
    main()
