"""Regression model construction, training, and held-out evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from studysync.data import FEATURE_COLUMNS, TARGET_COLUMN

MODEL_NAMES = ("Ridge Regression", "Random Forest", "Gradient Boosting")


def _build_pipeline(model_name: str) -> Pipeline:
    if model_name not in MODEL_NAMES:
        raise ValueError(f"Unknown model: {model_name}")

    preprocessing = ColumnTransformer(
        transformers=[
            ("subject", OneHotEncoder(handle_unknown="ignore"), ["subject"]),
            (
                "numeric",
                StandardScaler(),
                [column for column in FEATURE_COLUMNS if column != "subject"],
            ),
        ]
    )
    regressors: dict[str, Any] = {
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, min_samples_leaf=2, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": HistGradientBoostingRegressor(
            max_iter=120, learning_rate=0.08, l2_regularization=1.0, random_state=42
        ),
    }
    return Pipeline(
        [("preprocessing", preprocessing), ("regressor", regressors[model_name])]
    )


def _validate_dataset(data: pd.DataFrame) -> None:
    required = set(FEATURE_COLUMNS) | {TARGET_COLUMN}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(sorted(missing))}")
    if len(data) < 20:
        raise ValueError("At least 20 rows are required for model evaluation")
    if data[list(required)].isna().any().any():
        raise ValueError("Model input columns cannot contain missing values")


def compare_models(data: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    """Evaluate regression pipelines on one shared, held-out test split."""
    _validate_dataset(data)
    features = data.loc[:, FEATURE_COLUMNS]
    target = data[TARGET_COLUMN]
    train_x, test_x, train_y, test_y = train_test_split(
        features, target, test_size=0.2, random_state=random_state
    )

    results = []
    for model_name in MODEL_NAMES:
        pipeline = _build_pipeline(model_name)
        pipeline.fit(train_x, train_y)
        predictions = pipeline.predict(test_x)
        results.append(
            {
                "model": model_name,
                "mae": float(mean_absolute_error(test_y, predictions)),
                "rmse": float(np.sqrt(mean_squared_error(test_y, predictions))),
                "r2": float(r2_score(test_y, predictions)),
            }
        )
    return pd.DataFrame(results).sort_values("mae", ignore_index=True)


def train_model(data: pd.DataFrame, model_name: str) -> Pipeline:
    """Fit one validated pipeline on all supplied rows for prediction/demo use."""
    _validate_dataset(data)
    pipeline = _build_pipeline(model_name)
    pipeline.fit(data.loc[:, FEATURE_COLUMNS], data[TARGET_COLUMN])
    return pipeline