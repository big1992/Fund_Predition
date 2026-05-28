import pandas as pd
import numpy as np

from data.preprocessor import DataPreprocessor


def _build_ohlcv(n_rows: int = 340) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    idx = pd.date_range("2022-01-01", periods=n_rows, freq="D")

    close = np.linspace(50.0, 120.0, n_rows) + rng.normal(0, 0.7, n_rows)
    open_ = close + rng.normal(0, 0.3, n_rows)
    high = np.maximum(open_, close) + 0.5
    low = np.minimum(open_, close) - 0.5
    volume = rng.integers(800_000, 1_800_000, size=n_rows)

    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "adj_close": close,
            "volume": volume,
        },
        index=idx,
    )


def test_prepare_lstm_data_returns_multifeature_sequences():
    pre = DataPreprocessor()
    df = _build_ohlcv()

    out = pre.prepare_lstm_data(df, target_col="close")

    # Multi-feature contract: 3D tensor with >1 feature.
    assert out["X_train"].ndim == 3
    assert out["X_train"].shape[2] > 1
    assert len(out["feature_cols"]) > 1

    # Core columns from plan_2 should exist when data is sufficient.
    expected_core = {"close", "open", "high", "low", "volume", "rsi_14", "macd", "bb_middle", "atr_14", "daily_return"}
    assert expected_core.issubset(set(out["feature_cols"]))
