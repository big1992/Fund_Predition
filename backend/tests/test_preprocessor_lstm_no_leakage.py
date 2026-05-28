import numpy as np
import pandas as pd

from config.settings import DATA_SETTINGS
from data.preprocessor import DataPreprocessor
from features.technical_indicators import TechnicalIndicators


def _build_ohlcv(n_rows: int = 320) -> pd.DataFrame:
    """Create deterministic OHLCV data with an extreme tail segment."""
    rng = np.random.default_rng(42)
    idx = pd.date_range("2023-01-01", periods=n_rows, freq="D")

    base = np.linspace(100.0, 180.0, n_rows)
    close = base + rng.normal(0.0, 0.5, n_rows)

    # Make the final segment much larger to detect leakage easily.
    tail_start = int(n_rows * (DATA_SETTINGS["train_ratio"] + DATA_SETTINGS["val_ratio"]))
    close[tail_start:] = close[tail_start:] + 500.0

    open_ = close + rng.normal(0.0, 0.3, n_rows)
    high = np.maximum(open_, close) + 0.8
    low = np.minimum(open_, close) - 0.8
    volume = rng.integers(1_000_000, 2_000_000, size=n_rows)

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


def test_prepare_lstm_data_fits_scalers_on_train_only():
    pre = DataPreprocessor()
    df = _build_ohlcv()

    output = pre.prepare_lstm_data(df, target_col="close")
    assert output["X_train"].shape[0] > 0

    # Reproduce the same feature frame used in prepare_lstm_data for comparison.
    feat_df = TechnicalIndicators.calculate_all(pre.clean(df)).dropna()
    feature_cols = output["feature_cols"]

    n = len(feat_df)
    train_end = int(n * DATA_SETTINGS["train_ratio"])
    train_df = feat_df.iloc[:train_end]

    # If scalers are train-only, their max bounds must match train split bounds.
    np.testing.assert_allclose(
        pre.price_scaler.data_max_[0],
        train_df["close"].max(),
        rtol=0,
        atol=1e-9,
    )

    train_feature_max = train_df[feature_cols].max().values
    np.testing.assert_allclose(
        pre.lstm_feature_scaler.data_max_,
        train_feature_max,
        rtol=0,
        atol=1e-9,
    )
