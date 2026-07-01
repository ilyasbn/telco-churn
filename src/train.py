"""Train, evaluate, and save the Telco churn model."""

import json
from typing import Any, Dict, Optional

import joblib

from .config import ProjectConfig
from .data import load_data, split_data
from .evaluate import evaluate_model
from .model import build_model


def train_model(config: Optional[ProjectConfig] = None) -> Dict[str, Any]:
    """Run the full training pipeline and save the model plus metrics."""

    config = config or ProjectConfig()
    features, target = load_data(config)
    x_train, x_test, y_train, y_test = split_data(features, target, config)

    model = build_model(config)
    model.fit(x_train, y_train)

    metrics = evaluate_model(model, x_test, y_test)
    metrics["data"] = {
        "rows": int(len(features)),
        "features": features.columns.tolist(),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "churn_rate": float(target.mean()),
    }
    metrics["model"] = {
        "type": "RandomForestClassifier",
        "n_estimators": config.n_estimators,
        "max_depth": config.max_depth,
        "min_samples_leaf": config.min_samples_leaf,
        "class_weight": "balanced",
        "random_state": config.random_state,
    }

    config.model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, config.model_path)

    config.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with config.metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    return {
        "model_path": config.model_path,
        "metrics_path": config.metrics_path,
        **metrics,
    }


if __name__ == "__main__":
    result = train_model()
    print(f"Model saved to: {result['model_path']}")
    print(f"Metrics saved to: {result['metrics_path']}")
    print(f"Accuracy: {result['accuracy']:.3f}")
    print(f"Precision: {result['precision']:.3f}")
    print(f"Recall: {result['recall']:.3f}")
    print(f"F1 score: {result['f1']:.3f}")
    print(f"ROC AUC: {result['roc_auc']:.3f}")
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(result["confusion_matrix"])
    print(result["report"])
