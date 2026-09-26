"""
sentiment_validation.py file

Sentiment backbone validation.

Checks how well XLM-T reads Arabic vs English tweet sentiment, using the UMSAB
benchmark (the same data XLM-T was evaluated on in Barbieri et al. 2022). MARBERT
is run on the Arabic split as an Arabic-specific point of comparison.


"""

import urllib.request

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from transformers import pipeline
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

sns.set_theme(style="whitegrid")

# UMSAB ships as plain text files in the XLM-T repo (one tweet / one label per
# line), which dodges the datasets-library script that breaks on v4+.
DATA_URL = "https://raw.githubusercontent.com/cardiffnlp/xlm-t/main/data/sentiment"
LABEL_TO_ID = {"negative": 0, "neutral": 1, "positive": 2}
CLASS_NAMES = ["neg", "neu", "pos"]

XLMT_MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
MARBERT_MODEL = "Ammar-alhaj-ali/arabic-MARBERT-sentiment"


def load_umsab(language, split="test"):
    def pull(name):
        url = f"{DATA_URL}/{language}/{name}"
        return urllib.request.urlopen(url).read().decode("utf-8").splitlines()
    return pd.DataFrame({
        "text": pull(f"{split}_text.txt"),
        "gold": [int(x) for x in pull(f"{split}_labels.txt")],
    })


def load_model(name):
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("sentiment-analysis", model=name, device=device,
                    truncation=True, max_length=128)


def predict(model, texts):
    # both models expose lowercase labels through the pipeline, so we map on the
    # string rather than the raw class id (their id orderings differ)
    return [LABEL_TO_ID[o["label"].lower()] for o in model(list(texts), batch_size=64)]


def metrics(df):
    return {
        "n": len(df),
        "accuracy": accuracy_score(df["gold"], df["pred"]),
        "macro_f1": f1_score(df["gold"], df["pred"], average="macro"),
    }


def confusion_panel(frames, fname="figures/output/sentiment_confusion.png"):
    fig, axes = plt.subplots(1, len(frames), figsize=(5 * len(frames), 4.2))
    for ax, (title, df) in zip(np.atleast_1d(axes), frames):
        cm = confusion_matrix(df["gold"], df["pred"], labels=[0, 1, 2])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
        ax.set(title=title, xlabel="predicted", ylabel="gold")
    fig.tight_layout()
    fig.savefig(fname, dpi=130)
    plt.close(fig)


def comparison_bar(summary, fname="figures/output/sentiment_accuracy.png"):
    long = summary.melt(id_vars="model", value_vars=["accuracy", "macro_f1"],
                        var_name="metric", value_name="score")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(data=long, x="model", y="score", hue="metric", ax=ax)
    ax.axhline(1 / 3, ls="--", color="0.5", lw=1)   # 3-class random baseline
    ax.set(xlabel="", ylabel="", ylim=(0, 1))
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(fname, dpi=130)
    plt.close(fig)


def main():
    from pathlib import Path
    Path("prototype/output").mkdir(exist_ok=True)

    arabic = load_umsab("arabic")
    english = load_umsab("english")
    print(f"loaded {len(arabic)} Arabic and {len(english)} English tweets")

    xlmt = load_model(XLMT_MODEL)
    arabic["pred"] = predict(xlmt, arabic["text"])
    english["pred"] = predict(xlmt, english["text"])

    marbert = load_model(MARBERT_MODEL)
    arabic_marbert = arabic[["text", "gold"]].copy()
    arabic_marbert["pred"] = predict(marbert, arabic_marbert["text"])

    runs = [("XLM-T (ar)", arabic), ("XLM-T (en)", english), ("MARBERT (ar)", arabic_marbert)]
    rows = []
    for name, df in runs:
        m = metrics(df)
        rows.append({"model": name, **m})
        print(f"  {name:14s} acc={m['accuracy']:.3f}  macro-F1={m['macro_f1']:.3f}")

    summary = pd.DataFrame(rows)
    gap = summary.loc[summary.model == "XLM-T (en)", "accuracy"].iloc[0] - \
          summary.loc[summary.model == "XLM-T (ar)", "accuracy"].iloc[0]
    print(f"\n  XLM-T English-minus-Arabic accuracy gap: {gap:+.3f}")

    summary.to_csv("prototype/output/sentiment_summary.csv", index=False)
    confusion_panel(runs)
    comparison_bar(summary)
    print("\nfigures written to figures/")


if __name__ == "__main__":
    main()
