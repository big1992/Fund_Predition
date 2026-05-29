import numpy as np
import pandas as pd

from config.settings import DATA_SETTINGS
from data.preprocessor import DataPreprocessor


def _build_ohlcv(n_rows: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    idx = pd.date_range("2023-01-01", periods=n_rows, freq="D")
    close = np.linspace(100.0, 160.0, n_rows) + rng.normal(0.0, 0.4, n_rows)
    open_ = close + rng.normal(0.0, 0.2, n_rows)
    high = np.maximum(open_, close) + 0.6
    low = np.minimum(open_, close) - 0.6
    volume = rng.integers(800_000, 1_500_000, size=n_rows)
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


def test_split_time_series_respects_gap_days():
    pre = DataPreprocessor()
    df = _build_ohlcv(200)
    train, val, test = pre.split_time_series(df, train_ratio=0.7, val_ratio=0.15, gap_days=5)

    assert len(train) == 140
    assert len(val) == 30
    assert len(test) == 20

    # Index separation confirms gap rows are excluded from val/test boundaries.
    assert (val.index[0] - train.index[-1]).days == 6
    assert (test.index[0] - val.index[-1]).days == 6


def test_prepare_lstm_data_split_meta_contains_gap_boundaries():
    pre = DataPreprocessor()
    df = _build_ohlcv(320)

    original_gap = DATA_SETTINGS.get("validation_gap_days", 0)
    DATA_SETTINGS["validation_gap_days"] = 4
    try:
        out = pre.prepare_lstm_data(df, target_col="close")
    finally:
        DATA_SETTINGS["validation_gap_days"] = original_gap

    split_meta = out.get("split_meta", {})
    assert split_meta.get("gap_days") == 4
    assert split_meta.get("train_end", 0) < split_meta.get("val_start", 0)
    assert split_meta.get("val_end", 0) <= split_meta.get("test_start", 0)
