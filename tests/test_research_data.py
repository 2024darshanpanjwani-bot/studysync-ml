import pandas as pd
import pytest

from studysync.research_data import (
    ASSUMED_2026_DATASET,
    KAGGLE_2024_DATASET,
    RESEARCH_COLUMNS,
    extract_research_dataset,
    transform_research_dataset,
)


def test_kaggle_2024_source_is_loaded_and_transformed_without_generation():
    source = extract_research_dataset(KAGGLE_2024_DATASET)
    transformed = transform_research_dataset(source, KAGGLE_2024_DATASET)

    assert len(source) == 6607
    assert len(transformed) == len(source)
    assert tuple(transformed.columns) == RESEARCH_COLUMNS
    assert transformed.iloc[0].to_dict() == {
        "hours_studied": 23,
        "sleep_hours": 7,
        "attendance_percent": 84,
        "previous_scores": 73,
        "exam_score": 67,
    }


def test_assumed_2026_sample_is_separate_and_has_required_disclaimer():
    source = extract_research_dataset(ASSUMED_2026_DATASET)
    transformed = transform_research_dataset(source, ASSUMED_2026_DATASET)

    assert len(source) == 20
    assert tuple(transformed.columns) == RESEARCH_COLUMNS
    assert len(transformed) != len(extract_research_dataset(KAGGLE_2024_DATASET))
    assert all(pd.api.types.is_numeric_dtype(transformed[column]) for column in RESEARCH_COLUMNS)


def test_transform_rejects_missing_or_non_numeric_research_fields():
    incomplete = pd.DataFrame(
        {"Hours_Studied": [5], "Sleep_Hours": [7], "Attendance": [80], "Exam_Score": [75]}
    )
    with pytest.raises(ValueError, match="Previous_Scores"):
        transform_research_dataset(incomplete, KAGGLE_2024_DATASET)

    invalid = pd.DataFrame(
        {
            "hours_studied": ["five"],
            "sleep_hours": [7],
            "attendance_percent": [80],
            "previous_scores": [70],
            "exam_score": [75],
        }
    )
    with pytest.raises(ValueError, match="finite numeric"):
        transform_research_dataset(invalid, ASSUMED_2026_DATASET)
