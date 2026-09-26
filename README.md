# Beyond the Global Average: Regional Social Media Trend Prediction

CM3070 Final Project, BSc Computer Science (University of London).
Abdulaziz AlHasan.

The project predicts whether a topic's weekly Google Trends search growth
will be in the top 20% for a region in the following week. It compares
temporal features on their own with temporal features plus sentiment from
YouTube comments (scored with XLM-T), for Saudi Arabia and the United
States, with Bahrain and the UAE included for the cross-regional analysis.

## Repository layout

```
src/          pipeline scripts
prototype/    early XLM-T check on the UMSAB tweet benchmark
data/         Google Trends data, weekly sentiment features, feature matrix,
              comment IDs
data/labels/  hand-labelling sample and the model's answers for it
data/raw/     comment text and collector state (not in the repository)
results/      experiment, significance, diagnosis and validation results
models/       trained Saudi and US models
figures/      every figure used in the report
```

Run the scripts from the repository root. File paths are set relative to the
repository root, and output folders are created when needed.

## Pipeline

| Step | Script | Output |
|------|--------|--------|
| 1 | `src/pull_trends.py` | `data/trends_raw_extended.csv` |
| 2 | `src/youtube_collector2.py` | `data/raw/yt_comments.csv` |
| 3 | `src/sentiment_features.py` (GPU) | `data/sentiment_features.csv` |
| 4 | `src/build_matrix.py` | `data/feature_matrix.csv` |
| 5 | `src/run_experiments.py` | `results/experiment_*.csv` |
| 6 | `src/per_region_tests.py` | `results/per_region_significance.csv` |
| 7 | `src/diagnose_change.py` | `results/diagnosis_results.csv` |
| 8 | `src/train_final_model.py` | `models/SA_model.joblib`, `models/US_model.joblib` |
| 9 | `src/predict.py` | predictions |
| 10 | `src/make_figures.py` | `figures/*.png` |

Sentiment validation on the project's own comments:

| Script | Output |
|--------|--------|
| `src/make_label_sample.py` | `data/labels/label_sample.csv`, `data/labels/label_answers.csv` |
| `src/score_labels.py` | `results/confusion_matrices.csv`, `results/validation_results.csv` |

`src/make_comment_ids.py` writes `data/comment_ids.csv`.

Steps 1 to 3 need the API key and the comment text. Everything from step 4
onwards runs from the files in this repository:

```
pip install -r requirements.txt
python src/build_matrix.py
python src/run_experiments.py
python src/per_region_tests.py
python src/diagnose_change.py
python src/train_final_model.py
python src/score_labels.py
python src/make_figures.py
```

## Making a prediction

```
python src/predict.py --region SA --topic weather
python src/predict.py --all --csv predictions.csv
```

The model predicts from the latest week in `data/feature_matrix.csv`, so
this is not a live forecast. Held-out performance was around 0.59 ROC-AUC.

## Data

The YouTube comment text is not included, because it belongs to the people
who wrote it. `data/comment_ids.csv` lists the ID, video, channel, region,
topic and week of all 262,390 scored comments, so they can be fetched again
through the YouTube Data API. `data/labels/label_sample.csv` holds the hand
labels without the comment text.

Collecting comments needs a YouTube Data API key set as an environment
variable:

```
$env:YOUTUBE_API_KEY="your key"
```

The key is not stored anywhere in this repository.

## Figures

| File | Shows | Made by |
|------|-------|---------|
| `architecture.png` | pipeline architecture | diagram, drawn separately |
| `gantt_phased.png` | project work plan | diagram, drawn separately |
| `figure_trends.png` | weekly search interest for four topics | `make_figures.py` |
| `figure_coverage.png` | usable weeks of sentiment, round 1 and round 2 | `make_figures.py` |
| `figure_confusion.png` | hand labels against XLM-T, by language | `make_figures.py` |
| `figure_roc.png` | ROC curves for the three experiments | `make_figures.py` |
| `figure_gains.png` | change in ROC-AUC from adding sentiment, by region | `make_figures.py` |
| `diffusion.png` | which region leads which, by topic type | `make_figures.py` |

## Main results

- Temporal features only (Exp 1): ROC-AUC 0.588 (logistic regression),
  0.586 (random forest), 0.545 (XGBoost), against 0.517 for persistence.
- Adding sentiment gave no established gain in either region. Every 95%
  bootstrap interval on the gain includes zero.
- XLM-T agreed with the hand labels on 0.674 of comments in both languages.
