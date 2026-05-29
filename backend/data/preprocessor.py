"""
Data preprocessing: cleaning, normalization, and train/val/test splitting.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Optional, Tuple
import logging

from config.settings import DATA_SETTINGS, LSTM_PARAMS

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Clean, normalize, and split financial time-series data."""

    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.price_scaler = MinMaxScaler(feature_range=(0, 1))
        self._is_fitted = False

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean raw OHLCV data:
        - Forward-fill missing values (market holidays)
        - Remove rows with zero/negative prices
        - Cap outliers at 3 standard deviations
        """
        df = df.copy()

        # Forward fill then backward fill
        df = df.ffill().bfill()

        # Remove invalid prices
        price_cols = ["open", "high", "low", "close", "adj_close"]
        existing_price_cols = [c for c in price_cols if c in df.columns]
        for col in existing_price_cols:
            df = df[df[col] > 0]

        # Outlier capping (Z-score > 3)
        for col in existing_price_cols:
            if col in df.columns and len(df) > 0:
                mean = df[col].mean()
                std = df[col].std()
                if std > 0:
                    lower = mean - 3 * std
                    upper = mean + 3 * std
                    df[col] = df[col].clip(lower=lower, upper=upper)

        return df

    def normalize(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Normalize all numeric columns to [0, 1]."""
        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if not numeric_cols:
            return df

        if fit:
            df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
            self._is_fitted = True
        else:
            if self._is_fitted:
                df[numeric_cols] = self.scaler.transform(df[numeric_cols])
        return df

    def inverse_normalize(self, values: np.ndarray, col_index: int = 0) -> np.ndarray:
        """Inverse transform normalized values back to original scale."""
        if not self._is_fitted:
            return values

        # Create a dummy array with correct shape
        dummy = np.zeros((len(values), self.scaler.n_features_in_))
        dummy[:, col_index] = values.flatten()
        result = self.scaler.inverse_transform(dummy)
        return result[:, col_index]

    def normalize_prices(self, prices: np.ndarray, fit: bool = True) -> np.ndarray:
        """Normalize price array for LSTM input."""
        prices_2d = prices.reshape(-1, 1)
        if fit:
            normalized = self.price_scaler.fit_transform(prices_2d)
        else:
            normalized = self.price_scaler.transform(prices_2d)
        return normalized.flatten()

    def inverse_normalize_prices(self, values: np.ndarray) -> np.ndarray:
        """Inverse transform prices."""
        return self.price_scaler.inverse_transform(values.reshape(-1, 1)).flatten()

    @staticmethod
    def augment_training_data(
        X: np.ndarray, y: np.ndarray, noise_factor: float = 0.005, n_augments: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Augment training data with Gaussian noise injection.
        noise_factor: std of noise relative to feature std (default 0.5%)
        Returns augmented (X, y) concatenated with originals.
        """
        X_aug = [X]
        y_aug = [y]
        for _ in range(n_augments):
            noise = np.random.normal(0, noise_factor, X.shape)
            X_noisy = X + noise * np.std(X, axis=0, keepdims=True)
            X_aug.append(X_noisy)
            y_aug.append(y)  # targets stay the same
        return np.concatenate(X_aug, axis=0), np.concatenate(y_aug, axis=0)

    def create_sequences(
        self, data: np.ndarray, seq_length: int = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create LSTM input sequences.
        Input:  [p1, p2, ..., pN]
        Output: X = [[p1..p60], [p2..p61], ...], y = [p61, p62, ...]
        """
        if seq_length is None:
            seq_length = LSTM_PARAMS["sequence_length"]

        X, y = [], []
        for i in range(seq_length, len(data)):
            X.append(data[i - seq_length : i])
            y.append(data[i])
        return np.array(X), np.array(y)

    def create_feature_sequences(
        self, features: np.ndarray, target: np.ndarray, seq_length: int = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create LSTM sequences with multiple features.
        features: (n_samples, n_features)
        target: (n_samples,)
        """
        if seq_length is None:
            seq_length = LSTM_PARAMS["sequence_length"]

        X, y = [], []
        for i in range(seq_length, len(features)):
            X.append(features[i - seq_length : i])
            y.append(target[i])
        return np.array(X), np.array(y)

    def split_time_series(
        self, df: pd.DataFrame,
        train_ratio: float = None,
        val_ratio: float = None,
        gap_days: int = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split time-series data chronologically (no shuffling).
        Returns: (train, validation, test) DataFrames
        """
        if train_ratio is None:
            train_ratio = DATA_SETTINGS["train_ratio"]
        if val_ratio is None:
            val_ratio = DATA_SETTINGS["val_ratio"]
        if gap_days is None:
            gap_days = int(DATA_SETTINGS.get("validation_gap_days", 0))
        gap_days = max(0, int(gap_days))

        n = len(df)
        train_end = int(n * train_ratio)
        val_len = int(n * val_ratio)
        val_start = min(train_end + gap_days, n)
        val_end = min(val_start + val_len, n)
        test_start = min(val_end + gap_days, n)

        train = df.iloc[:train_end]
        val = df.iloc[val_start:val_end]
        test = df.iloc[test_start:]

        logger.info(
            "Split: train=%d, gap=%d, val=%d, gap=%d, test=%d (total=%d)",
            len(train), gap_days, len(val), gap_days, len(test), n,
        )
        return train, val, test

    # LSTM feature columns (from technical indicators + sentiment)
    LSTM_FEATURE_COLS = [
        "close", "open", "high", "low", "volume",
        "rsi_14", "macd", "bb_middle", "atr_14", "daily_return",
        "momentum_5d", "momentum_20d", "volatility_regime",
        "bb_position", "price_sma20_ratio", "sma20_sma50_cross",
        "day_of_week", "sentiment_score",
    ]

    def prepare_lstm_data(
        self, df: pd.DataFrame, target_col: str = "close", sentiment_df=None
    ) -> dict:
        """
        Full LSTM data preparation pipeline with multi-feature input:
        1. Clean → 2. Calculate indicators → 3. Split → 4. Fit scaler on TRAIN only
        5. Normalize → 6. Create sequences
        """
        from features.technical_indicators import TechnicalIndicators

        df = self.clean(df)

        if target_col not in df.columns:
            raise ValueError(f"Column '{target_col}' not found in DataFrame")

        # Calculate technical indicators for multi-feature LSTM (with sentiment)
        df_feat = TechnicalIndicators.calculate_all(df, sentiment_df=sentiment_df)
        df_feat = df_feat.dropna()

        # Determine available feature columns
        feature_cols = [c for c in self.LSTM_FEATURE_COLS if c in df_feat.columns]
        if not feature_cols:
            feature_cols = [target_col]

        # Split boundaries FIRST to prevent data leakage
        n = len(df_feat)
        train_end = int(n * DATA_SETTINGS["train_ratio"])
        val_len = int(n * DATA_SETTINGS["val_ratio"])
        gap_days = max(0, int(DATA_SETTINGS.get("validation_gap_days", 0)))
        val_start = min(train_end + gap_days, n)
        val_end = min(val_start + val_len, n)
        test_start = min(val_end + gap_days, n)

        train_df = df_feat.iloc[:train_end]

        # Fit feature scaler ONLY on training data (data leakage fix)
        from sklearn.preprocessing import MinMaxScaler
        self.lstm_feature_scaler = MinMaxScaler(feature_range=(0, 1))
        self.lstm_feature_scaler.fit(train_df[feature_cols].values)
        self.lstm_feature_cols = feature_cols

        # Fit price scaler ONLY on training prices (data leakage fix)
        train_prices = train_df[target_col].values.reshape(-1, 1)
        self.price_scaler.fit(train_prices)

        # Transform full frame using train-fitted scalers.
        # We then split by target index so sequence windows can be built safely.
        all_features = self.lstm_feature_scaler.transform(df_feat[feature_cols].values)
        all_target = self.price_scaler.transform(
            df_feat[target_col].values.reshape(-1, 1)
        ).flatten()

        seq_len = LSTM_PARAMS["sequence_length"]

        # Build all sequences, then assign each sample by its target row index.
        X_all, y_all = self.create_feature_sequences(all_features, all_target, seq_len)
        target_indices = np.arange(seq_len, n)

        train_mask = target_indices < train_end
        val_mask = (target_indices >= val_start) & (target_indices < val_end)
        test_mask = target_indices >= test_start

        X_train, y_train = X_all[train_mask], y_all[train_mask]
        X_val, y_val = X_all[val_mask], y_all[val_mask]
        X_test, y_test = X_all[test_mask], y_all[test_mask]

        logger.info(
            "LSTM data: X_train=%s, gap=%d, X_val=%s, gap=%d, X_test=%s, features=%d",
            X_train.shape, gap_days, X_val.shape, gap_days, X_test.shape, len(feature_cols),
        )

        return {
            "X_train": X_train, "y_train": y_train,
            "X_val": X_val, "y_val": y_val,
            "X_test": X_test, "y_test": y_test,
            "dates": df_feat.index.tolist(),
            "original_prices": df_feat[target_col].values,
            "feature_cols": feature_cols,
            "split_meta": {
                "n_rows": int(n),
                "train_end": int(train_end),
                "val_start": int(val_start),
                "val_end": int(val_end),
                "test_start": int(test_start),
                "gap_days": int(gap_days),
                "seq_len": int(seq_len),
            },
        }

    def prepare_xgboost_data(
        self, df: pd.DataFrame, target_col: str = "close"
    ) -> dict:
        """
        Prepare data for XGBoost (tabular features + target).
        Target: next-day percentage return (not absolute price).
        Assumes df already has technical indicator columns.
        """
        df = self.clean(df)
        df = df.dropna()

        if target_col not in df.columns:
            raise ValueError(f"Column '{target_col}' not found")

        # Target: next-day percentage RETURN (price-independent)
        next_close = df[target_col].shift(-1)
        df["target"] = (next_close - df[target_col]) / df[target_col]
        df = df.dropna()

        # Exclude raw price columns to avoid price-level dependency
        # Tree models trained on absolute prices can't extrapolate
        exclude_cols = ["target", "close", "open", "high", "low", "adj_close"]
        feature_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c not in exclude_cols
        ]

        # Split
        train, val, test = self.split_time_series(df)

        X_train = train[feature_cols].values
        y_train = train["target"].values
        X_val = val[feature_cols].values
        y_val = val["target"].values
        X_test = test[feature_cols].values
        y_test = test["target"].values

        return {
            "X_train": X_train, "y_train": y_train,
            "X_val": X_val, "y_val": y_val,
            "X_test": X_test, "y_test": y_test,
            "feature_names": feature_cols,
            "dates": df.index.tolist(),
            # Base prices for converting returns back to prices (for metrics)
            "close_test": test["close"].values if "close" in test.columns else None,
            "split_meta": {
                "n_rows": int(len(df)),
                "train_rows": int(len(train)),
                "val_rows": int(len(val)),
                "test_rows": int(len(test)),
                "gap_days": int(DATA_SETTINGS.get("validation_gap_days", 0)),
            },
        }
