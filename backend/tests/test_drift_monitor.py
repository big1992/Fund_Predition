import numpy as np

from services.drift_monitor import evaluate_simple_drift


def test_drift_monitor_flags_retrain_on_high_volatility():
    # High-vol regime
    returns = np.array([0.05, -0.06, 0.04, -0.05, 0.07, -0.04], dtype=float)
    out = evaluate_simple_drift(train_mape=12.0, latest_returns=returns)
    assert out["status"] in ("warn", "drifted")
    assert out["should_retrain"] is True
    assert out["drift_score"] >= 0.5


def test_drift_monitor_ok_on_stable_returns():
    returns = np.array([0.002, -0.001, 0.0015, -0.0008, 0.0012], dtype=float)
    out = evaluate_simple_drift(train_mape=3.0, latest_returns=returns)
    assert out["status"] == "ok"
    assert out["should_retrain"] is False
