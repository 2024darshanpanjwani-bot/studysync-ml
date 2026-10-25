import pandas as pd
import pytest

from studysync.analytics import subject_weaknesses, summarize_progress
from studysync.planner import generate_timetable


def test_timetable_allocates_exact_time_with_more_for_weaker_subject():
    timetable = generate_timetable(
        {"Physics": 45.0, "History": 85.0}, weekly_hours=4, days=4
    )
    totals = timetable.groupby("subject")["duration_minutes"].sum()

    assert timetable["duration_minutes"].sum() == 240
    assert totals["Physics"] > totals["History"]
    assert timetable["duration_minutes"].max() <= 50
    assert timetable["day"].nunique() <= 4


@pytest.mark.parametrize(
    "kwargs",
    [
        {"subject_scores": {}, "weekly_hours": 2},
        {"subject_scores": {"Math": 101}, "weekly_hours": 2},
        {"subject_scores": {"Math": 70}, "weekly_hours": 0},
        {"subject_scores": {"Math": 70}, "weekly_hours": 2, "days": 8},
    ],
)
def test_timetable_rejects_invalid_inputs(kwargs):
    with pytest.raises(ValueError):
        generate_timetable(**kwargs)


def test_subject_weaknesses_sorts_and_recommends():
    scores = pd.DataFrame(
        {"subject": ["Physics", "Physics", "History"], "exam_score": [50, 60, 80]}
    )
    summary = subject_weaknesses(scores, target_score=70)

    assert summary.iloc[0]["subject"] == "Physics"
    assert summary.iloc[0]["gap_to_target"] == 15
    assert "Prioritize Physics" in summary.iloc[0]["recommendation"]


def test_empty_progress_summary_has_zero_values():
    summary = summarize_progress(
        pd.DataFrame(columns=["duration_minutes", "focus_rating", "session_date"])
    )

    assert summary == {
        "session_count": 0,
        "total_minutes": 0,
        "average_focus": 0.0,
        "active_days": 0,
    }