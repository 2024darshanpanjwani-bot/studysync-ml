"""Loading and transforming the separate research dashboard datasets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIRECTORY = Path(__file__).resolve().parent.parent / "data"

KAGGLE_2024_DATASET = "2024 Kaggle Dataset — Student Performance Factors"
ASSUMED_2026_DATASET = "Assumed Sample Data — Not the Original Research Dataset."
RESEARCH_DATASETS = (KAGGLE_2024_DATASET, ASSUMED_2026_DATASET)
RESEARCH_COLUMNS = (
    "hours_studied",
    "sleep_hours",
    "attendance_percent",
    "previous_scores",
    "exam_score",
)

_DATASET_FILES = {
    KAGGLE_2024_DATASET: "student_performance_factors_2024.csv",
    ASSUMED_2026_DATASET: "assumed_2026_sample.csv",
}
_KAGGLE_COLUMN_MAP = {
    "Hours_Studied": "hours_studied",
    "Sleep_Hours": "sleep_hours",
    "Attendance": "attendance_percent",
    "Previous_Scores": "previous_scores",
    "Exam_Score": "exam_score",
}


def extract_research_dataset(dataset_name: str) -> pd.DataFrame:
    """Read one selected dataset without combining it with another source."""
    try:
        filename = _DATASET_FILES[dataset_name]
    except KeyError as error:
        raise ValueError(f"Unknown research dataset: {dataset_name}") from error

    path = DATA_DIRECTORY / filename
    if not path.is_file():
        raise FileNotFoundError(f"Research dataset file not found: {path}")
    return pd.read_csv(path)


def transform_research_dataset(
    raw_data: pd.DataFrame, dataset_name: str
) -> pd.DataFrame:
    """Select and normalize the five numeric research fields used in the dashboard."""
    if dataset_name == KAGGLE_2024_DATASET:
        missing = set(_KAGGLE_COLUMN_MAP).difference(raw_data.columns)
        if missing:
            raise ValueError(
                f"Kaggle dataset is missing required columns: {', '.join(sorted(missing))}"
            )
        data = raw_data.loc[:, list(_KAGGLE_COLUMN_MAP)].rename(
            columns=_KAGGLE_COLUMN_MAP
        )
    elif dataset_name == ASSUMED_2026_DATASET:
        missing = set(RESEARCH_COLUMNS).difference(raw_data.columns)
        if missing:
            raise ValueError(
                f"Assumed sample dataset is missing required columns: {', '.join(sorted(missing))}"
            )
        data = raw_data.loc[:, RESEARCH_COLUMNS].copy()
    else:
        raise ValueError(f"Unknown research dataset: {dataset_name}")

    for column in RESEARCH_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    values = data.loc[:, RESEARCH_COLUMNS].to_numpy(dtype=float)
    if data.empty or not np.isfinite(values).all():
        raise ValueError("Research dataset must contain finite numeric values in every required field")
    if (data.loc[:, RESEARCH_COLUMNS] < 0).any().any():
        raise ValueError("Research dataset values cannot be negative")
    return data.reset_index(drop=True)
