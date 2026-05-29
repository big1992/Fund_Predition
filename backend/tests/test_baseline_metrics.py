import pytest
import pandas as pd

from models.baselines import evaluate_price_baselines


def test_evaluate_price_baselines_are_deterministic_on_synthetic_prices():
    df = pd.DataFrame(
        {
            "close": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0],
        },
        index=pd.date_range("2024-01-01", periods=10, freq="D"),
    )

    baselines = evaluate_price_baselines(df)

    assert set(baselines) == {"naive", "mean_return", "buy_hold"}
    assert baselines["naive"]["baseline_type"] == "last_close"
    assert baselines["naive"]["rmse"] == 1.0
    assert baselines["naive"]["mae"] == 1.0
    assert baselines["naive"]["directional_accuracy"] == 100.0

    assert baselines["mean_return"]["baseline_type"] == "train_mean_return"
    assert baselines["mean_return"]["rmse"] < baselines["naive"]["rmse"]

    assert baselines["buy_hold"]["baseline_type"] == "buy_and_hold"
    assert baselines["buy_hold"]["total_return"] == pytest.approx((109.0 - 107.0) / 107.0, abs=0.0001)
    assert baselines["buy_hold"]["max_drawdown"] == 0.0
