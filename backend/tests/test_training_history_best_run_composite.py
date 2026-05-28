from pathlib import Path

from data.storage import DatabaseManager


def test_get_best_training_run_uses_composite_score():
    db_file = Path("backend/tests/tmp_training_history.db")
    if db_file.exists():
        db_file.unlink()

    db = DatabaseManager(str(db_file))

    # Run A: lower MAPE but weaker directional accuracy.
    db.save_training_run(
        symbol="TEST.BK",
        model_type="all",
        metrics={
            "lstm": {"mape": 2.0, "directional_accuracy": 55.0},
            "xgboost": {"mape": 2.0, "directional_accuracy": 55.0},
        },
        params={},
    )

    # Run B: slightly higher MAPE but much better directional accuracy.
    # Composite should prefer B:
    # A score = 0.6*(100-55)+0.4*2.0 = 27.8
    # B score = 0.6*(100-70)+0.4*3.0 = 19.2
    run_b_id = db.save_training_run(
        symbol="TEST.BK",
        model_type="all",
        metrics={
            "lstm": {"mape": 3.0, "directional_accuracy": 70.0},
            "xgboost": {"mape": 3.0, "directional_accuracy": 70.0},
        },
        params={},
    )

    best = db.get_best_training_run("TEST.BK")
    assert best is not None
    assert best["id"] == run_b_id
    assert "composite_score" in best

    if db_file.exists():
        db_file.unlink()
