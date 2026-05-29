from services.alert_rules import build_watchlist_alerts


def test_alert_rules_emit_expected_alert_types():
    alerts = build_watchlist_alerts(
        symbol="TEST.BK",
        current_signal="SELL",
        previous_signal="BUY",
        avg_interval_pct=0.12,
        drift_entries=[{"model_name": "ensemble", "status": "drifted", "drift_score": 0.91}],
        validation_cards=[{"model_name": "ensemble", "status": "fail", "mape_improvement_pct": -2.5}],
    )
    types = {a["alert_type"] for a in alerts}
    assert "signal_change" in types
    assert "risk_range_expansion" in types
    assert "drift_warning" in types
    assert "validation_warning" in types
    assert "benchmark_underperformance" in types
