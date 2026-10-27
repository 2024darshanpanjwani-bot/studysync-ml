# StudySync

StudySync is a Streamlit study-planning demo with exam-score estimates, model
comparison, subject-level study recommendations, and SQLite-backed progress
tracking.

## Architecture

- `app.py` contains the Streamlit navigation and user-facing views.
- `studysync/research_data.py` extracts and transforms the separate dashboard
  research datasets.
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

The Dashboard uses the Kaggle 2024 dataset. A separate illustrative 2026
sample CSV is kept in `data/` for reference and is not presented as verified
research data. Neither file is generated or downloaded at runtime. The other
demo-model views continue to use a deterministic generated cohort. To write a
copy of that demo cohort for experimentation:

```bash
python -m scripts.generate_synthetic_data --rows 1200 --seed 42
```

By default, progress is stored in `data/studysync.sqlite3`. Set
`STUDYSYNC_DB_PATH` to use another SQLite file. The app does not require an
account or send study logs to a remote service.

## Research datasets and model limitations

### Primary dashboard dataset: Kaggle 2024

`data/student_performance_factors_2024.csv` is the original CSV from Kaggle's
[Student Performance Factors dataset](https://www.kaggle.com/datasets/lainguyn123/student-performance-factors)
(CC0: Public Domain). The dashboard extracts `Hours_Studied`, `Sleep_Hours`,
`Attendance`, `Previous_Scores`, and `Exam_Score`, and normalizes their names
for display. The source CSV is kept intact. Kaggle source data is not generated
by this application; this project does not independently verify how the
dataset's records were collected.

### Separate 2026 paper illustration

`data/assumed_2026_sample.csv` contains a small, project-created illustration
using the five requested columns. No original 2026 paper CSV was present in
the project. Its values are reasonable illustrative assumptions, not verified
research observations or values taken from or attributed to the paper. This
file is not used by the Dashboard. The Dashboard always uses the 2024 Kaggle
dataset.

The other demo-model, prediction, subject-insight, and model-comparison views
continue to use a reproducible cohort generated from an illustrative formula
and random noise. Those predictions and evaluation scores demonstrate
application behavior only; they do not establish real-world accuracy, fairness,
or educational effectiveness, and should not be used for high-stakes decisions.
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
