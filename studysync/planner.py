"""Personalized weekly study-time allocation."""

from __future__ import annotations

import math

import pandas as pd


def generate_timetable(
    subject_scores: dict[str, float],
    weekly_hours: float,
    days: int = 5,
    target_score: float = 75.0,
    session_minutes: int = 50,
) -> pd.DataFrame:
    """Allocate available minutes toward weaker subjects and split into sessions."""
    if not subject_scores or any(not subject.strip() for subject in subject_scores):
        raise ValueError("Add at least one subject with a non-empty name")
    if any(not math.isfinite(score) or not 0 <= score <= 100 for score in subject_scores.values()):
        raise ValueError("Subject scores must be between 0 and 100")
    if not math.isfinite(weekly_hours) or weekly_hours <= 0:
        raise ValueError("weekly_hours must be greater than zero")
    if not 1 <= days <= 7:
        raise ValueError("days must be between 1 and 7")
    if session_minutes <= 0:
        raise ValueError("session_minutes must be greater than zero")

    total_minutes = round(weekly_hours * 60)
    weights = {
        subject: 1.0 + max(0.0, target_score - score) / 10.0
        for subject, score in subject_scores.items()
    }
    weight_total = sum(weights.values())
    allocations = {
        subject: int(total_minutes * weight / weight_total)
        for subject, weight in weights.items()
    }
    remainder = total_minutes - sum(allocations.values())
    for subject in sorted(subject_scores, key=subject_scores.get):
        if remainder <= 0:
            break
        allocations[subject] += 1
        remainder -= 1

    sessions = []
    day_names = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
    day_index = 0
    for subject, minutes in allocations.items():
        while minutes > 0:
            duration = min(session_minutes, minutes)
            sessions.append(
                {
                    "day": day_names[day_index % days],
                    "subject": subject,
                    "duration_minutes": duration,
                }
            )
            minutes -= duration
            day_index += 1

    return pd.DataFrame(sessions, columns=["day", "subject", "duration_minutes"])