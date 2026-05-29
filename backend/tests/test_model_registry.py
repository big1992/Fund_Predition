from pathlib import Path

from data.storage import DatabaseManager


def test_model_registry_entries_track_version_and_flags():
    db_file = Path("backend/tests/tmp_model_registry.db")
    if db_file.exists():
        db_file.unlink()

    db = DatabaseManager(str(db_file))

    run1 = db.save_training_run(
        symbol="TEST.BK",
        model_type="all",
        metrics={"lstm": {"mape": 4.0, "directional_accuracy": 60.0}},
        params={"p": 1},
    )
    db.save_model_registry_entries(
        symbol="TEST.BK",
        metrics_by_model={"lstm": {"mape": 4.0, "directional_accuracy": 60.0}},
        params={"p": 1},
        training_run_id=run1,
        train_start_date="2024-01-01",
        train_end_date="2024-12-31",
        artifact_paths={"lstm": "models/lstm_v1.keras"},
    )

    run2 = db.save_training_run(
        symbol="TEST.BK",
        model_type="all",
        metrics={"lstm": {"mape": 2.0, "directional_accuracy": 70.0}},
        params={"p": 2},
    )
    db.save_model_registry_entries(
        symbol="TEST.BK",
        metrics_by_model={"lstm": {"mape": 2.0, "directional_accuracy": 70.0}},
        params={"p": 2},
        training_run_id=run2,
        train_start_date="2025-01-01",
        train_end_date="2025-12-31",
        artifact_paths={"lstm": "models/lstm_v2.keras"},
    )

    rows = db.get_model_registry("TEST.BK")
    assert len(rows) >= 2

    latest = rows[0]
    older = rows[1]

    assert latest["version"] == 2
    assert older["version"] == 1
    assert latest["is_deployed"] is True
    assert older["is_deployed"] is False
    assert latest["is_best_run"] is True
    assert older["is_best_run"] is False
    assert latest["artifact_path"] == "models/lstm_v2.keras"

    if db_file.exists():
        db_file.unlink()
