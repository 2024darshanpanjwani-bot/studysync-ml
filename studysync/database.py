"""SQLite persistence for user-entered study sessions."""

from __future__ import annotations

import os
import sqlite3
from datetime import date
from pathlib import Path

import pandas as pd

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "studysync.sqlite3"


def _db_path(db_path: str | Path | None = None) -> Path:
    return Path(db_path or os.environ.get("STUDYSYNC_DB_PATH", DEFAULT_DB_PATH))


def init_db(db_path: str | Path | None = None) -> None:
    path = _db_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as connection:
        connection.execute(
            """CREATE TABLE IF NOT EXISTS study_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_date TEXT NOT NULL,
                subject TEXT NOT NULL,
                duration_minutes INTEGER NOT NULL CHECK (duration_minutes > 0),
                focus_rating INTEGER NOT NULL CHECK (focus_rating BETWEEN 1 AND 5),
                notes TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )"""
        )


def add_study_session(
    session_date: date,
    subject: str,
    duration_minutes: int,
    focus_rating: int,
    notes: str = "",
    db_path: str | Path | None = None,
) -> int:
    cleaned_subject = subject.strip()
    if not cleaned_subject:
        raise ValueError("Subject is required")
    if duration_minutes <= 0:
        raise ValueError("Duration must be greater than zero")
    if not 1 <= focus_rating <= 5:
        raise ValueError("Focus rating must be between 1 and 5")

    init_db(db_path)
    with sqlite3.connect(_db_path(db_path)) as connection:
        cursor = connection.execute(
            """INSERT INTO study_sessions
               (session_date, subject, duration_minutes, focus_rating, notes)
               VALUES (?, ?, ?, ?, ?)""",
            (session_date.isoformat(), cleaned_subject, duration_minutes, focus_rating, notes.strip()),
        )
        return int(cursor.lastrowid)


def fetch_study_sessions(db_path: str | Path | None = None) -> pd.DataFrame:
    init_db(db_path)
    with sqlite3.connect(_db_path(db_path)) as connection:
        return pd.read_sql_query(
            """SELECT id, session_date, subject, duration_minutes, focus_rating, notes
               FROM study_sessions ORDER BY session_date DESC, id DESC""",
            connection,
        )


def delete_study_session(session_id: int, db_path: str | Path | None = None) -> bool:
    init_db(db_path)
    with sqlite3.connect(_db_path(db_path)) as connection:
        cursor = connection.execute("DELETE FROM study_sessions WHERE id = ?", (session_id,))
        return cursor.rowcount > 0