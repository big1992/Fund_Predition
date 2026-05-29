from __future__ import annotations

import numpy as np


def evaluate_simple_drift(*, train_mape: float | None, latest_returns: np.ndarray) -> dict:
    reasons = []
    if latest_returns.size == 0:
        return {
            "status": "warn",
            "drift_score": 0.5,
            "should_retrain": False,
            "reasons": ["No recent returns available for drift check."],
            "metrics": {},
        }

    recent_vol = float(np.std(latest_returns))
    recent_abs_mean = float(np.mean(np.abs(latest_returns)))

    # Basic score anchored by current volatility/return regime.
    drift_score = min(1.0, recent_vol * 30.0 + recent_abs_mean * 15.0)

    status = "ok"
    if drift_score >= 0.8:
        status = "drifted"
    elif drift_score >= 0.5:
        status = "warn"

    if train_mape is not None and train_mape > 8:
        reasons.append(f"Last training MAPE is high ({train_mape:.2f}%).")
        drift_score = min(1.0, drift_score + 0.1)

    if recent_vol > 0.03:
        reasons.append("Recent return volatility is elevated.")
    if recent_abs_mean > 0.02:
        reasons.append("Average absolute recent return is elevated.")
    if not reasons:
        reasons.append("Recent market regime looks stable versus baseline thresholds.")

    should_retrain = drift_score >= 0.8 or (train_mape is not None and train_mape > 10)
    return {
        "status": status,
        "drift_score": round(float(drift_score), 3),
        "should_retrain": bool(should_retrain),
        "reasons": reasons,
        "metrics": {
            "recent_volatility": round(recent_vol, 5),
            "recent_abs_return_mean": round(recent_abs_mean, 5),
            "train_mape": train_mape,
        },
    }
