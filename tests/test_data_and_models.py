import pandas as pd
import pytest

from studysync.data import FEATURE_COLUMNS, generate_synthetic_dataset
from studysync.models import MODEL_NAMES, compare_models, train_model


def test_synthetic_data_is_reproducible_and_bounded():
    first = generate_synthetic_dataset(100, seed=11)
    second = generate_synthetic_dataset(100, seed=11)

    pd.testing.assert_frame_equal(first, second)
    assert first["exam_score"].between(0, 100).all()
    assert set(FEATURE_COLUMNS).issubset(first.columns)


def test_dataset_rejects_too_few_rows():
    with pytest.raises(ValueError, match="at least 20"):
        generate_synthetic_dataset(19)


def test_compare_models_returns_held_out_metrics():
    results = compare_models(generate_synthetic_dataset(120, seed=5))

    assert results["model"].tolist() == sorted(
        results["model"], key=results.set_index("model")["mae"].to_dict().get
    )
    assert set(results["model"]) == set(MODEL_NAMES)
    assert results[["mae", "rmse", "r2"]].notna().all().all()
    assert (results["mae"] >= 0).all()
    assert (results["rmse"] >= 0).all()


def test_prediction_pipeline_accepts_unseen_subject_category():
    data = generate_synthetic_dataset(100)
    model = train_model(data, MODEL_NAMES[0])
    prediction = model.predict(
        pd.DataFrame(
            [{
                "subject": "Art",
                "study_hours": 4.0,
                "attendance_rate": 90.0,
                "previous_score": 70.0,
                "sleep_hours": 7.0,
                "extracurricular_hours": 2.0,
            }]
        )
    )

    assert len(prediction) == 1
    assert pd.notna(prediction[0])


def test_model_validation_reports_missing_column():
    data = generate_synthetic_dataset(30).drop(columns="sleep_hours")
    with pytest.raises(ValueError, match="sleep_hours"):
        compare_models(data)