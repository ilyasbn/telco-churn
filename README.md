# Telco Customer Churn

Random Forest pipeline for the Kaggle Telco Customer Churn dataset:
https://www.kaggle.com/datasets/blastchar/telco-customer-churn/data

The goal is to predict whether a customer will churn (`Churn = Yes`) from
account, service, and billing information.

## Data

The dataset is stored at `data/churn.csv`.

- Rows: 7,043
- Columns: 21
- Target: `Churn`
- Churn rate: 26.5%

Important cleaning decisions:

- `customerID` is dropped before training.
- `Churn` is encoded as `Yes = 1` and `No = 0`.
- `TotalCharges` is converted from text to numeric.
- The 11 blank `TotalCharges` values are customers with `tenure = 0`, so they are set to `0`.
- Missing numeric values are imputed with the training median.
- Missing categorical values are imputed with the most frequent category using training data for each categorical column.
- Categorical columns are one-hot encoded with unknown categories ignored at prediction time.

## Project Files

```text
data/churn.csv
src/config.py          paths and model settings
src/data.py            loading, cleaning, target encoding, train/test split
src/preprocessing.py   numeric and categorical preprocessing
src/model.py           Random Forest pipeline
src/evaluate.py        metrics
src/train.py           training entry point
src/predict.py         prediction helpers
all_in_1_file.py       short script that trains and predicts once
artifacts/metrics.json latest saved metrics
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Train

```bash
python -m src.train
```

This trains an 80/20 stratified split and saves:

- `artifacts/telco_churn_random_forest.joblib`
- `artifacts/metrics.json`

## Latest Result

Random Forest settings:

- `n_estimators = 300`
- `min_samples_leaf = 2`
- `class_weight = balanced`
- `random_state = 42`

Held-out test results:

```text
Accuracy:  0.779
Precision: 0.575
Recall:    0.636
F1 score:  0.604
ROC AUC:   0.834
```

Confusion matrix:

```text
[[859, 176],
 [136, 238]]
```

Rows are actual classes and columns are predictions:

- 859 customers correctly predicted as not churned
- 238 customers correctly predicted as churned
- 176 false churn alerts
- 136 missed churners

For this problem, recall and ROC AUC matter more than accuracy alone because
the churn class is smaller and usually more important for retention work.

## Predict

Train first, then run:

```bash
python -m src.predict
```

Example output:

```text
{'prediction': 'Yes', 'churn_probability': 0.7963}
```

You can also call `predict_one_customer()` from Python with a customer
dictionary that uses the same feature names as `data/churn.csv`.
