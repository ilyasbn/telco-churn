"""Load, clean, and split the Telco Customer Churn data."""

from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import ProjectConfig


CHURN_TARGET_MAP = {"No": 0, "Yes": 1}


def clean_telco_data(data: pd.DataFrame) -> pd.DataFrame:
    """Apply deterministic cleaning that is safe before train/test split."""

    cleaned = data.copy()

    text_columns = cleaned.select_dtypes(include=["object", "string"]).columns
    for column in text_columns:
        cleaned[column] = cleaned[column].astype("string").str.strip()
        cleaned[column] = cleaned[column].replace("", pd.NA)

    if "TotalCharges" in cleaned.columns:
        cleaned["TotalCharges"] = pd.to_numeric(
            cleaned["TotalCharges"], errors="coerce"
        )
        if "tenure" in cleaned.columns:
            new_customer_rows = cleaned["TotalCharges"].isna() & (
                cleaned["tenure"] == 0
            )
            cleaned.loc[new_customer_rows, "TotalCharges"] = 0.0

    if "SeniorCitizen" in cleaned.columns:
        cleaned["SeniorCitizen"] = pd.to_numeric(
            cleaned["SeniorCitizen"], errors="coerce"
        )

    return cleaned


def encode_churn_target(target: pd.Series) -> pd.Series:
    """Convert the original Yes/No churn labels into 1/0."""

    if pd.api.types.is_numeric_dtype(target):
        return target.astype(int)

    encoded = target.astype("string").str.strip().map(CHURN_TARGET_MAP)
    if encoded.isna().any():
        bad_values = sorted(target[encoded.isna()].dropna().unique().tolist())
        raise ValueError(f"Unexpected values in Churn column: {bad_values}")

    return encoded.astype(int)


def load_data(config: ProjectConfig) -> Tuple[pd.DataFrame, pd.Series]:
    """Read the CSV and separate model inputs from the churn target."""

    data = clean_telco_data(pd.read_csv(config.data_path))
    if config.target_column not in data.columns:
        raise ValueError(
            f"Target column '{config.target_column}' was not found in "
            f"{config.data_path}."
        )

    if (
        config.customer_id_column in data.columns
        and data[config.customer_id_column].duplicated().any()
    ):
        raise ValueError("customerID contains duplicate values.")

    features = data.drop(
        columns=[config.target_column, config.customer_id_column],
        errors="ignore",
    )
    target = encode_churn_target(data[config.target_column])
    return features, target


def split_data(
    features: pd.DataFrame, target: pd.Series, config: ProjectConfig
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create a stratified split so churn ratio stays stable in both parts."""

    return train_test_split(
        features,
        target,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=target,
    )
