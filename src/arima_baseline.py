"""
ARIMA baseline for the trend label.

For each region-topic series, an ARIMA model is fitted on weekly search
interest using the training weeks of each expanding-window fold only. It
then makes one-step-ahead forecasts through the test weeks, updating with
each observed week but without refitting. The forecast for next week is
turned into a forecast velocity, (forecast - this week) / (this week + 1),
which is the same velocity used to define the label, and that is used as
the score for "top 20% next week".

It is evaluated on the same 509 test rows and folds as run_experiments.py,
so the ROC-AUC can be compared directly with Exp 1 and persistence. RMSE of
the velocity forecast is reported too.

Run:  python src/arima_baseline.py
"""

import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")

MATRIX_CSV = "data/feature_matrix.csv"
TRENDS_CSV = "data/trends_raw_extended.csv"
OUT_CSV    = "results/arima_baseline.csv"
PRED_CSV   = "results/arima_predictions.csv"
EXP_CSV    = "results/experiment_predictions.csv"   # for the comparison with Exp 1

N_FOLDS, MIN_TRAIN = 5, 0.35
ORDERS = [(1, 0, 0), (1, 0, 1), (2, 0, 0), (1, 1, 1), (0, 1, 1)]   # chosen by AIC per fold

TEMPORAL = ["gt_score", "velocity", "accel", "roll_mean", "roll_std",
            "vel_lag1", "vel_lag2", "is_top20"]
SENTIMENT = ["sent_mean", "prop_pos", "prop_neg", "prop_neu", "sent_std",
             "sent_mean_delta", "prop_neg_delta", "prop_pos_delta"]


def expanding_folds(weeks, n_folds=N_FOLDS, min_train=MIN_TRAIN):
    """Same splits as run_experiments.py."""
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


def fit_best(y):
    """Fit each candidate order and keep the lowest AIC."""
    best, best_aic = None, np.inf
    for order in ORDERS:
        try:
            res = ARIMA(y, order=order).fit()
        except Exception:
            continue
        if res.aic < best_aic:
            best, best_aic = res, res.aic
    return best


def delong_test(y, p1, p2):
    """DeLong's test for two correlated ROC-AUCs. Returns (auc1, auc2, p)."""
    y = np.asarray(y)
    pos, neg = y == 1, y == 0
    m, n = pos.sum(), neg.sum()

    def comp(p):
        p = np.asarray(p, float)
        x, yv = p[pos], p[neg]
        tx = np.array([(np.sum(yv < xi) + 0.5 * np.sum(yv == xi)) / n for xi in x])
        ty = np.array([(np.sum(x > yj) + 0.5 * np.sum(x == yj)) / m for yj in yv])
        return tx.mean(), tx, ty

    a1, tx1, ty1 = comp(p1)
    a2, tx2, ty2 = comp(p2)
    S = np.cov(np.vstack([tx1, tx2])) / m + np.cov(np.vstack([ty1, ty2])) / n
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    if var <= 0:
        return a1, a2, np.nan
    z = (a1 - a2) / np.sqrt(var)
    return a1, a2, 2 * (1 - stats.norm.cdf(abs(z)))


def main():
    try:
        fm = pd.read_csv(MATRIX_CSV, parse_dates=["week"])
        tr = pd.read_csv(TRENDS_CSV, parse_dates=["week"])
    except FileNotFoundError as e:
        sys.exit(f"{e}\nRun build_matrix.py first.")

    # same rows as run_experiments.py
    df = fm.dropna(subset=[c for c in TEMPORAL + SENTIMENT if c in fm.columns])
    df = df.sort_values("week").reset_index(drop=True)
    folds = expanding_folds(df["week"])
    print(f"Rows: {len(df)}, folds: {len(folds)}")

    rows = []
    for fi, (tr_weeks, te_weeks) in enumerate(folds, 1):
        cutoff = tr_weeks[-1]
        test = df[df["week"].isin(te_weeks)]
        for (geo, topic), g in test.groupby(["geo", "topic"]):
            s = (tr[(tr.geo == geo) & (tr.topic == topic)]
                   .set_index("week")["gt_score"].astype(float).sort_index())
            s = s.asfreq("7D")
            y_train = s[s.index <= cutoff]
            if len(y_train) < 20 or y_train.std() == 0:
                continue
            res = fit_best(y_train.values)
            if res is None:
                continue
            # one-step-ahead forecasts through the test weeks, updating
            # with each observed week but keeping the fitted parameters
            later = s[s.index > cutoff]
            full = res.append(later.values, refit=False)
            # position p holds the forecast for week p made with data up to p-1
            pred = full.predict(start=len(y_train), end=len(s))
            idx = list(later.index) + [s.index[-1] + pd.Timedelta(days=7)]
            fc = pd.Series(np.asarray(pred), index=idx)
            for _, r in g.iterrows():
                nxt = r["week"] + pd.Timedelta(days=7)
                if nxt not in fc.index:
                    continue
                x_t = s.get(r["week"], np.nan)
                f = fc[nxt]
                vel_fc = (f - x_t) / (x_t + 1.0)
                x_next = s.get(nxt, np.nan)
                vel_true = (x_next - x_t) / (x_t + 1.0)
                rows.append({"fold": fi, "geo": geo, "topic": topic,
                             "week": r["week"], "y_true": int(r["target"]),
                             "p_arima": vel_fc, "vel_true": vel_true})

    P = pd.DataFrame(rows)
    P.to_csv(PRED_CSV, index=False)
    print(f"Test predictions: {len(P)}")

    auc = roc_auc_score(P["y_true"], P["p_arima"])
    ok = P.dropna(subset=["vel_true"])
    rmse = float(np.sqrt(((ok["p_arima"] - ok["vel_true"]) ** 2).mean()))
    # naive forecast of no change, velocity 0, for comparison
    rmse_naive = float(np.sqrt((ok["vel_true"] ** 2).mean()))

    print(f"\nARIMA   ROC-AUC {auc:.3f}")
    print(f"        velocity RMSE {rmse:.3f} (no-change forecast {rmse_naive:.3f})")

    out = [{"model": "arima", "region": "all", "n": len(P),
            "roc_auc": round(auc, 4), "rmse_velocity": round(rmse, 4),
            "rmse_no_change": round(rmse_naive, 4)}]
    print("\nPer region:")
    for geo, g in P.groupby("geo"):
        a = roc_auc_score(g["y_true"], g["p_arima"])
        print(f"  {geo}  n={len(g)}  ROC-AUC {a:.3f}")
        out.append({"model": "arima", "region": geo, "n": len(g),
                    "roc_auc": round(a, 4), "rmse_velocity": None,
                    "rmse_no_change": None})
    pd.DataFrame(out).to_csv(OUT_CSV, index=False)

    # compare with the Exp 1 classifiers on the same rows
    try:
        E = pd.read_csv(EXP_CSV, parse_dates=["week"])
        E = E[E["experiment"] == "Exp1_temporal"]
        m = E.merge(P[["geo", "topic", "week", "p_arima"]],
                    on=["geo", "topic", "week"])
        print(f"\nExp 1 against ARIMA on the same {len(m)} rows (DeLong):")
        for c in [c for c in m.columns if c.startswith("p_") and c != "p_arima"]:
            a1, a2, p = delong_test(m["y_true"], m[c], m["p_arima"])
            print(f"  {c[2:]:<8} {a1:.3f} vs ARIMA {a2:.3f}   p={p:.4f}")
    except FileNotFoundError:
        print(f"\n{EXP_CSV} not found, skipping the comparison with Exp 1")
    print(f"\nWrote {OUT_CSV} and {PRED_CSV}")


if __name__ == "__main__":
    main()
