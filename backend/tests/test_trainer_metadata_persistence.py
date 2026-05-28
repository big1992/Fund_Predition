import json
from pathlib import Path

import pandas as pd

from models.trainer import ModelTrainer


def test_save_model_metadata_writes_json(tmp_path):
    trainer = ModelTrainer()
    trainer.model_dir = tmp_path

    df = pd.DataFrame(
        {"close": [100.0, 101.0, 102.0]},
        index=pd.date_range("2024-01-01", periods=3, freq="D"),
    )
    metrics = {"rmse": 1.23, "mape": 2.34, "training": {"loss": [0.1]}}

    trainer._save_model_metadata(
        symbol="PTT.BK",
        model_name="lstm",
        df=df,
        metrics=metrics,
        extra={"params": {"epochs": 10}},
    )

    out = tmp_path / "metadata_lstm_PTT_BK.json"
    assert out.exists()

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["symbol"] == "PTT.BK"
    assert payload["model_name"] == "lstm"
    assert payload["metrics"]["rmse"] == 1.23
    assert "training" not in payload["metrics"]
