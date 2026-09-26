"""
Makes all the report figures that come from the project's data.

    figure_trends.png      weekly search interest for four topics by region
    figure_coverage.png    usable weeks of sentiment per region-topic,
                           round 1 and round 2
    figure_gains.png       change in ROC-AUC from Exp 1 to Exp 2 by region,
                           with bootstrap intervals
    figure_roc.png         pooled ROC curves for the three experiments
    figure_confusion.png   hand labels against XLM-T, by language
    diffusion.png          which region leads which, by topic type

Figures are saved in the figures folder. If an input file is missing, that
figure is skipped with a message. The architecture diagram and the Gantt
chart are not made here.

Inputs (all written by the pipeline scripts):
    trends_raw_extended.csv, trends_raw.csv,
    per_region_significance.csv, experiment_predictions.csv,
    feature_matrix.csv, confusion_matrices.csv

Run:  python make_figures.py
"""

import os
import warnings
from itertools import combinations, permutations

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from sklearn.metrics import roc_auc_score, roc_curve

sns.set_theme(style="white")
# harmless tight_layout notice from the gains figure
warnings.filterwarnings("ignore", message=".*tight_layout.*")
DEEP = "#1F3864"
BLUE = "#2E5496"
PALE = "#B9C6D4"
GREY = "#5A6B7B"
ORANGE = "#B4632A"
RED = "#B4322A"

OUT_DIR = "figures"
REGION_COLORS = {"US": BLUE, "SA": "#4E7A2A", "AE": ORANGE, "BH": "#7A4E73"}
REGION_NAMES = {"US": "United States", "SA": "Saudi Arabia", "AE": "UAE",
                "BH": "Bahrain"}
MODEL_NAMES = {"logreg": "logistic regression", "rf": "random forest",
               "xgboost": "XGBoost"}
MIN_GAIN = 0.02   # smallest gain treated as meaningful

# Usable weeks after round 1, as reported in the draft. Round 2 values are
# the rows per region-topic in feature_matrix.csv.
ROUND1_COVERAGE = {
    ("SA", "artificial intelligence"): 8, ("SA", "weather"): 59,
    ("SA", "World Cup"): 20, ("SA", "Ramadan"): 20, ("SA", "inflation"): 9,
    ("US", "weather"): 71, ("US", "inflation"): 67,
    ("US", "artificial intelligence"): 65, ("US", "electric cars"): 38,
    ("US", "bitcoin"): 32, ("US", "World Cup"): 30, ("US", "ChatGPT"): 29,
}


def have(*files):
    missing = [f for f in files if not os.path.exists(f)]
    if missing:
        print(f"  skipped, missing {', '.join(missing)}")
    return not missing


def clean_axes(ax, keep_left=False):
    for sp in ("top", "right") + (() if keep_left else ("left",)):
        ax.spines[sp].set_visible(False)


def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def figure_trends():
    print("figure_trends")
    if not have("data/trends_raw_extended.csv"):
        return
    topics = [("Ramadan", "shared seasonal pattern"),
              ("ChatGPT", "shared rising pattern"),
              ("weather", "regional seasonal pattern"),
              ("electric cars", "region-specific pattern")]
    d = pd.read_csv("data/trends_raw_extended.csv", parse_dates=["week"])
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 6.4), sharex=True)
    for ax, (topic, label) in zip(axes.flat, topics):
        p = d[d.topic == topic].pivot(index="week", columns="geo", values="gt_score")
        c = p.corr().values
        r = np.nanmean(c[np.triu_indices(len(p.columns), 1)])
        for geo in ["US", "SA", "AE", "BH"]:
            if geo in p:
                ax.plot(p.index, p[geo], color=REGION_COLORS[geo], lw=1.3,
                        label=REGION_NAMES[geo])
        ax.set_title(f"{topic}: {label} (mean correlation {r:.2f})",
                     fontsize=9.5, weight="bold", color=DEEP, loc="left")
        ax.set_ylim(0, 105)
        ax.grid(axis="y", color="0.93", lw=0.8)
        clean_axes(ax, keep_left=True)
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.tick_params(labelsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("search interest (0-100, per region)", fontsize=8.5)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, -0.03))
    # a week starting late in a month is mostly in the next one
    start, end = d.week.min() + pd.Timedelta(days=6), d.week.max()
    fig.suptitle(f"Weekly search interest by region, {start:%B %Y} to {end:%B %Y}",
                 fontsize=11.5, weight="bold", color=DEEP, x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    save(fig, "figure_trends.png")


def figure_coverage():
    print("figure_coverage")
    if not have("data/feature_matrix.csv"):
        return
    fm = pd.read_csv("data/feature_matrix.csv")
    counts = fm.groupby(["geo", "topic"]).size()
    topics_all = sorted(set(fm.topic) | {t for _, t in ROUND1_COVERAGE}
                        | {"ChatGPT", "bitcoin", "electric cars", "Ramadan"})
    rows = []
    for region in ["SA", "US"]:
        for topic in topics_all:
            r2 = int(counts.get((region, topic), 0))
            rows.append((region, topic, ROUND1_COVERAGE.get((region, topic), 0), r2))
    df = pd.DataFrame(rows, columns=["region", "topic", "round1", "round2"])
    topics = (df.groupby("topic")["round2"].max()
                .sort_values(ascending=False).index.tolist())
    xmax = max(105, df[["round1", "round2"]].values.max() + 12)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.6), sharey=True)
    for ax, region in zip(axes, ["SA", "US"]):
        d = df[df.region == region].set_index("topic").loc[topics]
        y = np.arange(len(topics))
        ax.barh(y - 0.19, d["round1"], height=0.36, color=PALE)
        ax.barh(y + 0.19, d["round2"], height=0.36, color=BLUE)
        for i, (r1, r2) in enumerate(zip(d["round1"], d["round2"])):
            if r2:
                ax.text(r2 + 1.5, i + 0.19, str(r2), va="center", fontsize=7.5,
                        color=DEEP)
            if r1 and r1 != r2:
                ax.text(r1 + 1.5, i - 0.19, str(r1), va="center", fontsize=7.5,
                        color="#6B7C8F")
        ax.axvline(15, color=ORANGE, ls="--", lw=1.1)
        ax.set_yticks(y)
        ax.set_yticklabels(topics, fontsize=9)
        ax.set_xlim(0, xmax)
        ax.set_xlabel("usable weeks", fontsize=9)
        ax.set_title(REGION_NAMES[region], fontsize=10.5, weight="bold", color=DEEP)
        clean_axes(ax)
        ax.grid(axis="x", color="0.93", lw=0.8)
    axes[0].invert_yaxis()
    axes[1].text(16, -0.75, "threshold (15 weeks)", fontsize=7.5, color=ORANGE)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (PALE, BLUE)]
    fig.legend(handles, ["round 1 (draft report)", "round 2 (final)"],
               loc="lower center", ncol=2, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, -0.07))
    fig.suptitle("Usable weeks of sentiment coverage, before and after the "
                 "second collection round",
                 fontsize=11, weight="bold", color=DEEP, x=0.055, ha="left", y=1.02)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    save(fig, "figure_coverage.png")


def figure_gains():
    print("figure_gains")
    if not have("results/per_region_significance.csv"):
        return
    s = pd.read_csv("results/per_region_significance.csv")
    s = s[s.comparison == "Exp2_temporal_sentiment_vs_Exp1_temporal"]
    established = (s.verdict == "ESTABLISHED").any()

    fig, axes = plt.subplots(2, 1, figsize=(9.0, 4.6), sharex=True,
                             gridspec_kw={"hspace": 0.35})
    for ax, region in zip(axes, ["SA", "US"]):
        d = s[s.region == region].reset_index(drop=True)
        ax.axvspan(-MIN_GAIN, MIN_GAIN, color=ORANGE, alpha=0.07, zorder=0)
        ax.axvline(0, color=RED, lw=1.3, zorder=1)
        for i, r in d.iterrows():
            ax.plot([r["gain_lo"], r["gain_hi"]], [i, i], color=GREY, lw=2.6,
                    solid_capstyle="round", zorder=2)
            ax.plot(r["gain"], i, "o", ms=8, color=BLUE, zorder=3)
            ax.text(0.175, i, f"{r['gain']:+.3f}   p = {r['delong_p']:.3f}",
                    va="center", fontsize=8, color=GREY)
        ax.set_yticks(np.arange(len(d)))
        ax.set_yticklabels([MODEL_NAMES.get(m, m) for m in d["model"]], fontsize=9)
        ax.set_ylim(len(d) - 0.4, -0.6)
        ax.set_xlim(-0.16, 0.30)
        ax.set_title(REGION_NAMES[region], fontsize=10, weight="bold",
                     color=DEEP, loc="left")
        clean_axes(ax)
        ax.grid(axis="x", color="0.93", lw=0.8)
        ax.set_xticks([-0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15])

    axes[1].set_xlabel("change in ROC-AUC when sentiment features are added",
                       fontsize=9)
    axes[0].text(0, -0.95, "no change", ha="center", fontsize=7.5, color=RED)
    axes[0].text(MIN_GAIN + 0.004, -0.95,
                 "shaded: within the 0.02 minimum meaningful gain",
                 fontsize=7.5, color=ORANGE)
    fig.suptitle("Effect of adding sentiment features, by region",
                 fontsize=11.5, weight="bold", color=DEEP, x=0.015, ha="left",
                 y=1.03)
    if not established:
        fig.text(0.015, 0.965, "Every 95% interval crosses zero: no gain is "
                 "established in either region", fontsize=8.8, color=GREY,
                 ha="left")
    fig.legend(handles=[Line2D([], [], marker="o", color=BLUE, ls="", ms=8,
                               label="measured change"),
                        Line2D([], [], color=GREY, lw=2.6,
                               label="95% bootstrap interval")],
               loc="lower center", ncol=2, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=[0, 0.02, 1, 0.94])
    save(fig, "figure_gains.png")


def figure_roc():
    print("figure_roc")
    if not have("results/experiment_predictions.csv", "data/feature_matrix.csv"):
        return
    P = pd.read_csv("results/experiment_predictions.csv", parse_dates=["week"])
    fm = pd.read_csv("data/feature_matrix.csv", parse_dates=["week"])
    exps = [("Exp1_temporal", "Exp 1: temporal", BLUE, "-"),
            ("Exp2_temporal_sentiment", "Exp 2: + sentiment", ORANGE, "-"),
            ("Exp3_temporal_sentiment_regional", "Exp 3: + regional", GREY, "--")]
    models = [m for m in ("logreg", "rf", "xgboost") if f"p_{m}" in P.columns]

    # persistence predicts a top-20% week whenever this week was one,
    # so it gives a single point on the curve
    base = P[P.experiment == "Exp1_temporal"].merge(
        fm[["geo", "topic", "week", "is_top20"]],
        on=["geo", "topic", "week"], how="left")
    pred = base["is_top20"].fillna(0).astype(int)
    y = base["y_true"]
    tpr_p = ((pred == 1) & (y == 1)).sum() / max((y == 1).sum(), 1)
    fpr_p = ((pred == 1) & (y == 0)).sum() / max((y == 0).sum(), 1)

    fig, axes = plt.subplots(1, len(models), figsize=(4.0 * len(models), 4.1),
                             sharey=True)
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, models):
        for exp, label, color, ls in exps:
            d = P[P.experiment == exp]
            if d.empty:
                continue
            fpr, tpr, _ = roc_curve(d["y_true"], d[f"p_{m}"])
            auc = roc_auc_score(d["y_true"], d[f"p_{m}"])
            ax.plot(fpr, tpr, color=color, ls=ls, lw=1.6,
                    label=f"{label} ({auc:.3f})")
        ax.plot(fpr_p, tpr_p, "D", color=RED, ms=6, label="persistence")
        ax.plot([0, 1], [0, 1], color="0.75", lw=1, ls=":")
        ax.set_title(MODEL_NAMES[m], fontsize=10, weight="bold", color=DEEP,
                     loc="left")
        ax.set_xlabel("false positive rate", fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.grid(color="0.93", lw=0.8)
        clean_axes(ax, keep_left=True)
        ax.legend(frameon=False, fontsize=7.5, loc="lower right")
        ax.tick_params(labelsize=8)
    axes[0].set_ylabel("true positive rate", fontsize=9)
    fig.suptitle("ROC curves on the held-out folds, all regions pooled "
                 "(ROC-AUC in brackets)",
                 fontsize=11, weight="bold", color=DEEP, x=0.01, ha="left")
    fig.tight_layout()
    save(fig, "figure_roc.png")


def figure_confusion():
    print("figure_confusion")
    if not have("results/confusion_matrices.csv"):
        return
    cm = pd.read_csv("results/confusion_matrices.csv", index_col=[0, 1])
    classes = ["negative", "neutral", "positive"]
    regions = [r for r in ("SA", "US") if r in cm.index.get_level_values(0)]
    langs = {"SA": "Arabic (Saudi channels)", "US": "English (US channels)"}

    fig, axes = plt.subplots(1, len(regions), figsize=(4.6 * len(regions), 4.0))
    axes = np.atleast_1d(axes)
    for ax, region in zip(axes, regions):
        m = cm.loc[region].reindex(index=classes, columns=classes).fillna(0).astype(int)
        acc = np.trace(m.values) / max(m.values.sum(), 1)
        # shade by share of each hand-labelled class so rows can be compared
        share = m.div(m.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
        sns.heatmap(share, ax=ax, cmap="Blues", vmin=0, vmax=1, cbar=False,
                    annot=m.values, fmt="d", annot_kws={"fontsize": 10},
                    linewidths=1, linecolor="white", square=True)
        ax.set_title(f"{langs[region]}\nn = {m.values.sum()}, "
                     f"agreement {acc:.3f}",
                     fontsize=9.5, weight="bold", color=DEEP, loc="left")
        ax.set_xlabel("XLM-T label", fontsize=9)
        ax.set_ylabel("hand label", fontsize=9)
        ax.tick_params(labelsize=8.5, rotation=0)
    fig.suptitle("Hand labels against XLM-T predictions, by language",
                 fontsize=11, weight="bold", color=DEEP, x=0.01, ha="left")
    fig.text(0.01, -0.02, "Cells show counts. Shading shows the share of each "
             "hand-labelled class.", fontsize=8, color=GREY)
    fig.tight_layout()
    save(fig, "figure_confusion.png")


def _z(a):
    a = np.asarray(a, float)
    sd = a.std()
    return (a - a.mean()) / sd if sd else a


def _best_lead(x, y, max_lag=4):
    """Highest correlation of x leading y by 1 to max_lag weeks."""
    x, y = _z(x), _z(y)
    best = -np.inf
    for k in range(1, max_lag + 1):
        if len(x) - k < 5:
            break
        c = np.corrcoef(x[:-k], y[k:])[0, 1]
        if np.isfinite(c) and c > best:
            best = c
    return best


def _diffusion(raw, topics, edge_floor=0.2):
    """Lead-follow edges between regions, averaged over the given topics."""
    sub = raw[raw.topic.isin(topics)]
    regions = sorted(sub.geo.unique())
    leads = {(a, b): [] for a, b in permutations(regions, 2)}
    sync = []
    for _, g in sub.groupby("topic"):
        wide = g.pivot_table(index="week", columns="geo", values="gt_score").dropna()
        if len(wide) < 10:
            continue
        c = wide.corr().to_numpy()
        sync.append(c[np.triu_indices_from(c, 1)].mean())
        for a, b in permutations(regions, 2):
            v = _best_lead(wide[a], wide[b])
            if np.isfinite(v):
                leads[(a, b)].append(v)
    edges = []
    for a, b in combinations(regions, 2):
        ab = np.mean(leads[(a, b)]) if leads[(a, b)] else 0
        ba = np.mean(leads[(b, a)]) if leads[(b, a)] else 0
        if max(ab, ba) >= edge_floor:
            edges.append((a, b, ab) if ab >= ba else (b, a, ba))
    score = {r: 0.0 for r in regions}
    for u, v, w in edges:
        score[u] += w
        score[v] -= w
    return edges, score, float(np.nanmean(sync))


def figure_diffusion():
    print("diffusion")
    # uses the 2022-2024 Trends pull, which covers more weeks
    if not have("data/trends_raw.csv"):
        return
    raw = pd.read_csv("data/trends_raw.csv", parse_dates=["week"])
    tech_names = {"ChatGPT", "artificial intelligence", "bitcoin",
                  "electric car", "electric cars"}
    topics = sorted(raw.topic.unique())
    groups = [("Global technology topics", [t for t in topics if t in tech_names]),
              ("Regional and cultural topics", [t for t in topics if t not in tech_names])]
    pos = {"US": (0, 1), "SA": (-1, 0), "AE": (1, 0), "BH": (0, -0.5)}
    label = {"US": "United\nStates", "SA": "Saudi\nArabia", "AE": "UAE",
             "BH": "Bahrain"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    for ax, (title, ts) in zip(axes, groups):
        edges, score, sync = _diffusion(raw, ts)
        leader = max(score, key=score.get)
        for u, v, w in edges:
            ax.annotate("", xy=pos[v], xytext=pos[u],
                        arrowprops=dict(arrowstyle="->", color=GREY, lw=1.8,
                                        shrinkA=24, shrinkB=24,
                                        connectionstyle="arc3,rad=0.12"))
        for r, (x, y) in pos.items():
            if r not in score:
                continue
            lead = r == leader
            ax.scatter(x, y, s=2600, color=BLUE if lead else PALE, zorder=3)
            ax.text(x, y, label[r], ha="center", va="center", fontsize=9.5,
                    weight="bold", color="white" if lead else DEEP, zorder=4)
        ax.set_title(f"{title}\nmean correlation {sync:.2f}", fontsize=11,
                     weight="bold", color=DEEP)
        ax.set_xlim(-1.4, 1.4)
        ax.set_ylim(-0.85, 1.35)
        ax.axis("off")
    fig.suptitle("Cross-regional diffusion by topic type (arrow points from "
                 "leader to follower)", fontsize=12, weight="bold", color=DEEP)
    tidy = {"electric car": "electric cars", "ramadan": "Ramadan"}
    names = [", ".join(tidy.get(t, t) for t in g[1]) for g in groups]
    fig.text(0.5, 0.01, f"Technology topics: {names[0]}. "
             f"Cultural topics: {names[1]}.",
             ha="center", fontsize=8.5, color=GREY)
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    save(fig, "diffusion.png")


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    figure_trends()
    figure_coverage()
    figure_gains()
    figure_roc()
    figure_confusion()
    figure_diffusion()
