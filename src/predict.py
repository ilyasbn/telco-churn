"""Use the saved churn model for customer predictions."""

from pathlib import Path
from typing import Iterable, List, Mapping, Optional

import joblib
import pandas as pd

from .config import ProjectConfig
from .data import clean_telco_data


def load_trained_model(model_path: Optional[Path] = None):
    """Load the model saved by train.py."""

    path = model_path or ProjectConfig().model_path
    return joblib.load(path)


def _churn_label(value: int) -> str:
    return "Yes" if int(value) == 1 else "No"


def predict(
    records: Iterable[Mapping[str, object]],
    model_path: Optional[Path] = None,
) -> List[str]:
    """Predict Yes/No churn labels for customer records."""

    model = load_trained_model(model_path)
    features = clean_telco_data(pd.DataFrame.from_records(records))
    predictions = model.predict(features)
    return [_churn_label(prediction) for prediction in predictions]


def predict_with_probability(
    records: Iterable[Mapping[str, object]],
    model_path: Optional[Path] = None,
) -> List[dict]:
    """Predict churn labels and churn probabilities."""

    model = load_trained_model(model_path)
    features = clean_telco_data(pd.DataFrame.from_records(records))
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1]

    return [
        {
            "prediction": _churn_label(prediction),
            "churn_probability": round(float(probability), 4),
        }
        for prediction, probability in zip(predictions, probabilities)
    ]


def predict_one_customer(
    customer: Mapping[str, object],
    model_path: Optional[Path] = None,
) -> dict:
    """Predict churn for a single customer dictionary."""

    return predict_with_probability([customer], model_path=model_path)[0]


if __name__ == "__main__":
    sample_customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 2,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.70,
        "TotalCharges": 151.65,
    }
    print(predict_one_customer(sample_customer))
