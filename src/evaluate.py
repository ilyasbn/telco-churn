"""Evaluation helpers for the churn classifier."""

from typing import Any, Dict

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


def evaluate_model(model: Any, features: pd.DataFrame, target: pd.Series) -> Dict[str, Any]:
    """Return the main classification metrics for a fitted model."""

    predictions = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1]
    precision, recall, f1, _ = precision_recall_fscore_support(
        target,
        predictions,
        average="binary",
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy_score(target, predictions)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc_score(target, probabilities)),
        "confusion_matrix": confusion_matrix(
            target, predictions, labels=[0, 1]
        ).tolist(),
        "report": classification_report(
            target,
            predictions,
            target_names=["No churn", "Churn"],
            zero_division=0,
        ),
        "report_dict": classification_report(
            target,
            predictions,
            target_names=["No churn", "Churn"],
            output_dict=True,
            zero_division=0,
        ),
    }
