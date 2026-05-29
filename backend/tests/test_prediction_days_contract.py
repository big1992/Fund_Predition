import pandas as pd
from fastapi.testclient import TestClient

from api.main import app, app_state


class _FakeDB:
    def get_stock_prices(self, symbol: str):
        return pd.DataFrame({"close": [100.0, 101.0, 102.0]})


class _FakeTrainer:
    def __init__(self):
        self.received_days = None

    def predict(self, df, symbol, model_type="ensemble", days=None):
        self.received_days = days
        return {
            "symbol": symbol,
            "model": model_type,
            "current_price": 102.0,
            "predictions": [
                {
                    "date": f"2026-05-{i+1:02d}",
                    "predicted_price": 100.0 + i,
                    "predicted_low": 99.0 + i,
                    "predicted_high": 101.0 + i,
                    "confidence": 70.0,
                }
                for i in range(days or 0)
            ],
            "signal": "HOLD",
            "signal_reason": "ok",
            "confidence": 70.0,
            "model_disagreement_pct": 1.23,
            "confidence_band": "medium",
        }


def test_prediction_days_query_is_forwarded_to_trainer_and_response_length_matches():
    fake_trainer = _FakeTrainer()

    with TestClient(app) as client:
        old_db = app_state["db"]
        old_trainer = app_state["trainer"]
        app_state["db"] = _FakeDB()
        app_state["trainer"] = fake_trainer

        try:
            resp = client.get("/api/predictions/PTT.BK", params={"days": 7, "model": "ensemble"})

            assert resp.status_code == 200
            body = resp.json()
            assert fake_trainer.received_days == 7
            assert len(body["predictions"]) == 7
            assert body["confidence_band"] == "medium"
            assert body["model_disagreement_pct"] == 1.23
            assert all(
                p["predicted_low"] <= p["predicted_price"] <= p["predicted_high"]
                for p in body["predictions"]
            )
        finally:
            app_state["db"] = old_db
            app_state["trainer"] = old_trainer
