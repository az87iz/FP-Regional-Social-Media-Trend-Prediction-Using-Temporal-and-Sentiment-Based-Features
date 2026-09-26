"""
Significance tests for each region separately.

    experiment_predictions.csv -> per_region_significance.csv

The pooled comparison can hide regions moving in opposite directions, with
a gain in one region cancelled out by a loss in the other. So each region
is tested on its own. The Saudi test set is small, so as well as DeLong's
test this reports:
- paired bootstrap 95% confidence intervals for each AUC and for the gain
- the number of positive cases, since AUC on very few positives is unstable

A gain is only counted as established if DeLong's p < 0.05, the bootstrap
interval on the gain does not include zero, and the gain is at least 0.02.

Run:  python per_region_tests.py
"""

import sys

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

PRED_CSV = "experiment_predictions.csv"
OUT_CSV  = "per_region_significance.csv"

N_BOOT   = 5000
SEED     = 42
MIN_GAIN = 0.02

PAIRS = [("Exp1_temporal", "Exp2_temporal_sentiment"),
         ("Exp2_temporal_sentiment", "Exp3_temporal_sentiment_regional")]


def delong_test(y, p1, p2):
    """DeLong's test for two correlated ROC-AUCs. Returns (auc1, auc2, z, p)."""
    y = np.asarray(y)
    pos, neg = y == 1, y == 0
    m, n = pos.sum(), neg.sum()
    if m == 0 or n == 0:
        return np.nan, np.nan, np.nan, np.nan

    def structural(p):
        p = np.asarray(p, dtype=float)
        x, yv = p[pos], p[neg]
        tx = np.array([(np.sum(yv < xi) + 0.5 * np.sum(yv == xi)) / n for xi in x])
        ty = np.array([(np.sum(x > yj) + 0.5 * np.sum(x == yj)) / m for yj in yv])
        return tx.mean(), tx, ty

    a1, tx1, ty1 = structural(p1)
    a2, tx2, ty2 = structural(p2)
    s10 = np.cov(np.vstack([tx1, tx2]))
    s01 = np.cov(np.vstack([ty1, ty2]))
    S = s10 / m + s01 / n
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    if var <= 0:
        return a1, a2, np.nan, np.nan
    z = (a1 - a2) / np.sqrt(var)
    return a1, a2, z, 2 * (1 - stats.norm.cdf(abs(z)))


def boot_auc_and_gain(y, p_new, p_old, n_boot=N_BOOT, seed=SEED):
    """Paired bootstrap of both AUCs and the gain between them.

    Both models are scored on the same resampled rows each time, since they
    were tested on the same rows.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    p_new, p_old = np.asarray(p_new), np.asarray(p_old)
    n = len(y)
    a_new, a_old, gains = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yy = y[idx]
        if len(np.unique(yy)) < 2:
            continue
        an = roc_auc_score(yy, p_new[idx])
        ao = roc_auc_score(yy, p_old[idx])
        a_new.append(an); a_old.append(ao); gains.append(an - ao)
    if len(gains) < 100:
        return (np.nan,) * 6
    q = lambda a, lo, hi: (float(np.percentile(a, lo)), float(np.percentile(a, hi)))
    return (*q(a_new, 2.5, 97.5), *q(a_old, 2.5, 97.5), *q(gains, 2.5, 97.5))


def main():
    try:
        P = pd.read_csv(PRED_CSV, parse_dates=["week"])
    except FileNotFoundError:
        sys.exit(f"{PRED_CSV} not found. Run run_experiments.py first.")

    models = [c[2:] for c in P.columns if c.startswith("p_")]
    rows = []

    for region in sorted(P["geo"].unique()):
        R = P[P["geo"] == region]
        n_pos = int(R[R.experiment == R.experiment.iloc[0]]["y_true"].sum())
        n_tot = len(R[R.experiment == R.experiment.iloc[0]])
        print(f"\n=== {region}: {n_tot} test predictions, {n_pos} positives ===")
        if n_pos < 15:
            print(f"  only {n_pos} positive cases, AUC will be unstable")

        for old, new in PAIRS:
            A = R[R.experiment == old].sort_values(["topic", "week"]).reset_index(drop=True)
            B = R[R.experiment == new].sort_values(["topic", "week"]).reset_index(drop=True)
            if len(A) == 0 or len(B) == 0 or len(A) != len(B):
                continue
            if not (A["y_true"].values == B["y_true"].values).all():
                print(f"  {old} vs {new}: rows do not line up, skipped")
                continue

            print(f"\n  {new}  vs  {old}")
            for m in models:
                y = A["y_true"].values
                p_old, p_new = A[f"p_{m}"].values, B[f"p_{m}"].values
                a_new, a_old, z, p = delong_test(y, p_new, p_old)
                gain = a_new - a_old
                (nl, nh, ol, oh, gl, gh) = boot_auc_and_gain(y, p_new, p_old)

                spans_zero = (gl <= 0 <= gh) if gl == gl else True
                verdict = ("ESTABLISHED" if (p == p and p < 0.05
                           and not spans_zero and gain >= MIN_GAIN)
                           else "not established")
                print(f"    {m:<8} {a_old:.3f} [{ol:.3f},{oh:.3f}] -> "
                      f"{a_new:.3f} [{nl:.3f},{nh:.3f}]")
                print(f"             gain {gain:+.3f} "
                      f"95% CI [{gl:+.3f},{gh:+.3f}]  p={p:.4f}  {verdict}")

                rows.append({
                    "region": region, "comparison": f"{new}_vs_{old}",
                    "model": m, "n_test": n_tot, "n_positive": n_pos,
                    "auc_old": round(a_old, 4), "auc_old_lo": round(ol, 4),
                    "auc_old_hi": round(oh, 4),
                    "auc_new": round(a_new, 4), "auc_new_lo": round(nl, 4),
                    "auc_new_hi": round(nh, 4),
                    "gain": round(gain, 4), "gain_lo": round(gl, 4),
                    "gain_hi": round(gh, 4),
                    "delong_p": round(p, 5) if p == p else None,
                    "gain_ci_spans_zero": bool(spans_zero),
                    "verdict": verdict})

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)

    print()
    est = out[out.verdict == "ESTABLISHED"]
    if len(est):
        print("Established gains:")
        for _, r in est.iterrows():
            print(f"  {r['region']} {r['model']} {r['comparison']}: "
                  f"{r['gain']:+.3f} [{r['gain_lo']:+.3f},{r['gain_hi']:+.3f}]")
    else:
        print("No gain met all three criteria.")
    print(f"\nWrote {OUT_CSV}")


if __name__ == "__main__":
    main()
