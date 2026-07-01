"""Project settings for the Telco churn pipeline."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class ProjectConfig:
    """Values shared by training, evaluation, and prediction."""

    data_path: Path = Path("data/churn.csv")
    target_column: str = "Churn"
    customer_id_column: str = "customerID"
    model_path: Path = Path("artifacts/telco_churn_random_forest.joblib")
    metrics_path: Path = Path("artifacts/metrics.json")
    test_size: float = 0.20
    random_state: int = 42
    n_estimators: int = 300
    max_depth: Optional[int] = None
    min_samples_leaf: int = 2
