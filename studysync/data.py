"""Synthetic data generation for the StudySync demonstration."""

from __future__ import annotations

import numpy as np
import pandas as pd

SUBJECTS = ("Mathematics", "Biology", "Chemistry", "Physics", "History")
FEATURE_COLUMNS = (
    "subject",
    "study_hours",
    "attendance_rate",
    "previous_score",
    "sleep_hours",
    "extracurricular_hours",
)
TARGET_COLUMN = "exam_score"


def generate_synthetic_dataset(
    n_samples: int = 1200, seed: int = 42
) -> pd.DataFrame:
    """Generate illustrative student records; this is not empirical data."""
    if n_samples < 20:
        raise ValueError("n_samples must be at least 20 for model evaluation")

    rng = np.random.default_rng(seed)
    subjects = rng.choice(SUBJECTS, size=n_samples)
    study_hours = rng.uniform(0.5, 8.0, size=n_samples)
    attendance_rate = rng.uniform(55.0, 100.0, size=n_samples)
    previous_score = rng.uniform(30.0, 96.0, size=n_samples)
    sleep_hours = rng.uniform(4.0, 9.5, size=n_samples)
    extracurricular_hours = rng.uniform(0.0, 6.0, size=n_samples)
    subject_effect = pd.Series(subjects).map(
        {"Mathematics": 1.0, "Biology": 0.5, "Chemistry": 0.0, "Physics": -1.0, "History": 1.5}
    ).to_numpy()
    noise = rng.normal(0.0, 7.0, size=n_samples)
    exam_score = (
        0.37 * previous_score
        + 4.8 * study_hours
        + 0.17 * attendance_rate
        + 1.6 * sleep_hours
        - 0.75 * extracurricular_hours
        + subject_effect
        + noise
    )

    return pd.DataFrame(
        {
            "student_id": np.arange(1, n_samples + 1),
            "subject": subjects,
            "study_hours": np.round(study_hours, 2),
            "attendance_rate": np.round(attendance_rate, 1),
            "previous_score": np.round(previous_score, 1),
            "sleep_hours": np.round(sleep_hours, 1),
            "extracurricular_hours": np.round(extracurricular_hours, 1),
            "exam_score": np.round(np.clip(exam_score, 0.0, 100.0), 1),
        }
    )