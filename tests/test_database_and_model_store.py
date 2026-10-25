from datetime import date

import pytest

from studysync.data import generate_synthetic_dataset
from studysync.database import (
    add_study_session,
    delete_study_session,
    fetch_study_sessions,
)
from studysync.model_store import load_model, save_model
from studysync.models import MODEL_NAMES, train_model


def test_study_sessions_persist_validate_and_delete(tmp_path):
    db_path = tmp_path / "progress.sqlite3"
    session_id = add_study_session(date(2026, 1, 2), "  Biology ", 45, 4, db_path=db_path)
    sessions = fetch_study_sessions(db_path)

    assert len(sessions) == 1
    assert sessions.iloc[0]["subject"] == "Biology"
    assert sessions.iloc[0]["duration_minutes"] == 45
    assert delete_study_session(session_id, db_path)
    assert fetch_study_sessions(db_path).empty
    assert not delete_study_session(session_id, db_path)


@pytest.mark.parametrize(
    "subject,duration,focus",
    [(" ", 30, 3), ("Math", 0, 3), ("Math", 30, 6)],
)
def test_study_session_rejects_invalid_values(tmp_path, subject, duration, focus):
    with pytest.raises(ValueError):
        add_study_session(date.today(), subject, duration, focus, db_path=tmp_path / "test.sqlite3")


def test_joblib_model_round_trip(tmp_path):
    data = generate_synthetic_dataset(40)
    model = train_model(data, MODEL_NAMES[0])
    artifact = save_model(model, tmp_path / "models" / "model.joblib")

    loaded = load_model(artifact)
    assert loaded.predict(data.iloc[:2])[0] == pytest.approx(model.predict(data.iloc[:2])[0])


def test_joblib_model_load_reports_missing_artifact(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "missing.joblib")