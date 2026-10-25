"""Progress and subject-level analytics."""

from __future__ import annotations

import pandas as pd


def subject_weaknesses(data: pd.DataFrame, target_score: float = 70.0) -> pd.DataFrame:
    """Summarize subjects below target from the supplied cohort or score table."""
    if "subject" not in data.columns or "exam_score" not in data.columns:
        raise ValueError("Data must include subject and exam_score columns")
    summary = (
        data.groupby("subject", as_index=False)
        .agg(mean_score=("exam_score", "mean"), observations=("exam_score", "count"))
        .sort_values("mean_score", ignore_index=True)
    )
    summary["gap_to_target"] = (target_score - summary["mean_score"]).clip(lower=0)
    summary["recommendation"] = summary.apply(
        lambda row: (
            f"Prioritize {row['subject']} with two focused review blocks this week."
            if row["mean_score"] < target_score
            else f"Maintain steady practice in {row['subject']} and revisit missed questions."
        ),
        axis=1,
    )
    return summary


def summarize_progress(sessions: pd.DataFrame) -> dict[str, float | int]:
    """Compute summary KPIs for validated study-session records."""
    if sessions.empty:
        return {
            "session_count": 0,
            "total_minutes": 0,
            "average_focus": 0.0,
            "active_days": 0,
        }
    return {
        "session_count": int(len(sessions)),
        "total_minutes": int(sessions["duration_minutes"].sum()),
        "average_focus": float(sessions["focus_rating"].mean()),
        "active_days": int(sessions["session_date"].nunique()),
    }