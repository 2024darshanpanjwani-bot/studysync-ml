# StudySync

StudySync is a Streamlit study-planning demo with exam-score estimates, model
comparison, subject-level study recommendations, and SQLite-backed progress
tracking.

## Architecture

- `app.py` contains the Streamlit navigation and user-facing views.
- `studysync/data.py` creates reproducible synthetic cohort data.
- `studysync/models.py` builds regression pipelines and evaluates them on a
	held-out split.
- `studysync/planner.py` allocates study time according to subject scores.
- `studysync/analytics.py` summarizes progress and identifies weaker subjects.
- `studysync/database.py` validates and persists individual study sessions.
- `studysync/model_store.py` saves locally trained model artifacts with Joblib.

## Run locally

Use Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The app generates a deterministic synthetic cohort on first use. To write a
CSV copy for experimentation:

```bash
python -m scripts.generate_synthetic_data --rows 1200 --seed 42
```

By default, progress is stored in `data/studysync.sqlite3`. Set
`STUDYSYNC_DB_PATH` to use another SQLite file. The app does not require an
account or send study logs to a remote service.

## Synthetic data and model limitations

All cohort observations are generated from an illustrative formula and random
noise. Predictions and evaluation scores are for demonstrating application
behavior only. They do not establish real-world accuracy, fairness, or
educational effectiveness, and should not be used for high-stakes decisions.
Real deployments require consented, representative data, independent
validation, privacy review, and ongoing monitoring. Avoid entering sensitive
personal information in study-session notes.

The model comparison uses a reproducible train/test split and reports MAE,
RMSE, and R-squared on the synthetic holdout. The displayed model artifact is
trained on the synthetic demo data; Joblib artifacts should only be loaded from
trusted local sources.

## Tests

```bash
pytest
```
