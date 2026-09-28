# Beyond the Global Average: Regional Social Media Trend Prediction Using Temporal and Sentiment-Based Features

CM3070 Final Project, BSc Computer Science (University of London). Abdulaziz AlHasan.

The project predicts whether a topic's weekly Google Trends search growth will be in the top 20% for a region in the following week. It compares temporal features on their own with temporal features plus sentiment from YouTube comments, scored with XLM-T, for Saudi Arabia and the United States. Bahrain and the UAE are included in the cross-regional diffusion analysis.

## Constraints

* All commands must be run from the root folder of the repository.
* The YouTube comment text is not included, since it belongs to the people who wrote it. The data collection and sentiment scoring steps (steps 1 to 3 of the pipeline) therefore cannot be rerun from this repository alone. Their outputs are included in the `data` folder, so every step from building the feature matrix onwards runs without them.
* Collecting new comments needs a YouTube Data API key, and scoring them needs a GPU. Neither is needed to reproduce the results.
* Development was done in Google Colab (Linux, Python 3.13) for the GPU steps and in Visual Studio Code on Windows with Python 3.12 for the rest.

## Project Structure

* `src` folder stores the Python scripts for the whole pipeline, from data collection to prediction and figures.
* `prototype` folder stores the early check of XLM-T on the UMSAB tweet benchmark.
* `data` folder stores the Google Trends data, the weekly sentiment features, the feature matrix and the list of comment IDs.
* `data/labels` folder stores the hand-labelled validation sample, without comment text, and the model's predictions for it.
* `data/raw` folder is not in the repository. It holds the comment text and collector files when the collection scripts are run.
* `results` folder stores the experiment results, per-region significance tests, the Saudi diagnosis and the sentiment validation results.
* `models` folder stores the trained Saudi Arabia and United States models.
* `figures` folder stores every figure used in the report.

## Instructions for Running the Project

### 1. Get the Code

* Open a terminal and run:
  ```
  git clone https://github.com/az87iz/cm3070-regional-trend-prediction.git fp
  cd fp
  ```
* Alternatively, download the ZIP from GitHub and extract it.

### 2. Create Environment with Dependencies

* In the root folder, create a virtual environment:
  ```
  python -m venv venv
  ```
* Activate it:
  ```
  venv\Scripts\activate          (Windows)
  source venv/bin/activate       (macOS or Linux)
  ```
* Install the dependencies:
  ```
  pip install -r requirements.txt
  ```
* Keep the terminal open for the next steps.

### 3. Run the Pipeline

* In the same terminal within the root folder, run each script in order:
  ```
  python src/build_matrix.py
  python src/run_experiments.py
  python src/per_region_tests.py
  python src/diagnose_change.py
  python src/train_final_model.py
  python src/score_labels.py
  python src/make_figures.py
  ```
* `per_region_tests.py` and `diagnose_change.py` take a few minutes because of the bootstrap resampling.

### 4. Make a Prediction

* For one region and topic:
  ```
  python src/predict.py --region SA --topic weather
  ```
* For every region and topic:
  ```
  python src/predict.py --all
  ```
* To save the predictions to a CSV file:
  ```
  python src/predict.py --all --csv predictions.csv
  ```
* The model predicts from the latest week in the feature matrix, so this is not a live forecast.

### 5. Expected Results

* `build_matrix.py`: 613 rows.
* `run_experiments.py`: 601 rows and 509 test predictions. Experiment 1 ROC-AUC is 0.588 for logistic regression, 0.586 for random forest and 0.545 for XGBoost. The persistence baseline is 0.517.
* `per_region_tests.py`: no gain from sentiment features meets all three criteria in either region.
* `score_labels.py`: accuracy of 0.674 in both Arabic and English, and 28 of 46 Saudi-channel comments showing signs of Gulf origin.
* `make_figures.py`: six figures written to the `figures` folder.

Logistic regression and random forest results reproduce exactly. XGBoost results can differ by around 0.01 between machines, because its row and column sampling and multi-threaded tree building are not fully deterministic across operating systems and CPUs. The reported figures were produced on Google Colab (Linux). None of the conclusions change.

### 6. Collecting the Data Again (Optional)

These steps are not needed to reproduce the results.

* Create the raw data folder in the root folder:
  ```
  mkdir data/raw
  ```
* Pull the Google Trends data:
  ```
  python src/pull_trends.py
  ```
* Set a YouTube Data API key and collect comments:
  ```
  $env:YOUTUBE_API_KEY="your key"          (Windows PowerShell)
  export YOUTUBE_API_KEY="your key"         (macOS or Linux)
  python src/youtube_collector2.py
  ```
* Score the comments on a GPU, for example in Google Colab:
  ```
  python src/sentiment_features.py
  ```
* `data/comment_ids.csv` lists the ID, video, channel, region, topic and week of all 262,390 scored comments, so the same comments can be fetched again through the YouTube Data API. The API key is not stored anywhere in this repository.

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

## Main Results

* Temporal features alone reach ROC-AUC 0.588 (logistic regression), 0.586 (random forest) and 0.545 (XGBoost), against 0.517 for persistence.
* Adding sentiment features gives no established gain in either region. Every 95% bootstrap interval on the gain includes zero.
* XLM-T agreed with the hand labels on 0.674 of comments in both languages.
