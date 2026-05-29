from __future__ import annotations


def build_watchlist_alerts(
    *,
    symbol: str,
    current_signal: str,
    previous_signal: str | None,
    avg_interval_pct: float,
    drift_entries: list[dict],
    validation_cards: list[dict],
) -> list[dict]:
    alerts = []

    if previous_signal and previous_signal != current_signal:
        alerts.append({
            "alert_type": "signal_change",
            "severity": "high",
            "message": f"{symbol}: signal changed from {previous_signal} to {current_signal}.",
            "payload": {"previous_signal": previous_signal, "current_signal": current_signal},
        })

    if avg_interval_pct >= 0.06:
        sev = "high" if avg_interval_pct >= 0.1 else "medium"
        alerts.append({
            "alert_type": "risk_range_expansion",
            "severity": sev,
            "message": f"{symbol}: forecast range is wide ({avg_interval_pct*100:.2f}% average band).",
            "payload": {"avg_interval_pct": avg_interval_pct},
        })

    for d in drift_entries:
        status = str(d.get("status", "ok"))
        if status in ("warn", "drifted"):
            alerts.append({
                "alert_type": "drift_warning",
                "severity": "high" if status == "drifted" else "medium",
                "message": f"{symbol}: {d.get('model_name')} drift status is {status}.",
                "payload": {"model_name": d.get("model_name"), "status": status, "drift_score": d.get("drift_score")},
            })

    for c in validation_cards:
        status = str(c.get("status", "reference"))
        if status in ("warn", "fail"):
            alerts.append({
                "alert_type": "validation_warning",
                "severity": "high" if status == "fail" else "medium",
                "message": f"{symbol}: {c.get('model_name')} validation status is {status}.",
                "payload": {"model_name": c.get("model_name"), "status": status},
            })
        if isinstance(c.get("mape_improvement_pct"), (int, float)) and c.get("mape_improvement_pct") < 0:
            alerts.append({
                "alert_type": "benchmark_underperformance",
                "severity": "medium",
                "message": f"{symbol}: {c.get('model_name')} underperforms baseline by MAPE delta {c.get('mape_improvement_pct'):.2f}%.",
                "payload": {"model_name": c.get("model_name"), "mape_improvement_pct": c.get("mape_improvement_pct")},
            })

    return alerts
