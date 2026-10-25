"""Local Joblib persistence for trained demo model pipelines."""

from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline


def save_model(model: Pipeline, path: str | Path) -> Path:
    artifact_path = Path(path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, artifact_path)
    return artifact_path


def load_model(path: str | Path) -> Pipeline:
    """Load a trusted local artifact; Joblib files can execute code on load."""
    artifact_path = Path(path)
    if not artifact_path.is_file():
        raise FileNotFoundError(f"Model artifact not found: {artifact_path}")
    model = joblib.load(artifact_path)
    if not isinstance(model, Pipeline):
        raise ValueError("Artifact is not a scikit-learn Pipeline")
    return model