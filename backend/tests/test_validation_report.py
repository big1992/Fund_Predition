from services.validation_report import build_validation_report


def test_validation_report_includes_status_and_baseline_delta():
    metrics = {
        "naive": {"mape": 4.0, "directional_accuracy": 50.0, "description": "naive baseline"},
        "lstm": {"mape": 3.0, "directional_accuracy": 56.0, "rmse": 1.1, "mae": 0.8, "r_squared": 0.4},
        "xgboost": {"mape": 6.5, "directional_accuracy": 49.0, "rmse": 1.5, "mae": 1.0, "r_squared": 0.2},
    }

    report = build_validation_report("PTT.BK", metrics)
    cards = {c["model_name"]: c for c in report["cards"]}

    assert "lstm" in cards
    assert "xgboost" in cards
    assert "naive" in cards

    assert cards["lstm"]["status"] == "pass"
    assert cards["lstm"]["baseline_name"] == "naive"
    assert cards["lstm"]["mape_improvement_pct"] == 25.0

    assert cards["xgboost"]["status"] in {"warn", "fail"}
    assert cards["naive"]["status"] == "reference"
