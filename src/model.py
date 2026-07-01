"""Build the Random Forest churn model."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from .config import ProjectConfig
from .preprocessing import build_preprocessor


def build_model(config: ProjectConfig | None = None) -> Pipeline:
    """Create the full preprocessing + model pipeline."""

    config = config or ProjectConfig()
    classifier = RandomForestClassifier(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        min_samples_leaf=config.min_samples_leaf,
        class_weight="balanced",
        random_state=config.random_state,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", classifier),
        ]
    )
