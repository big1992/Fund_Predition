"""
Technical indicators for feature engineering.
Calculates 33 features across 6 categories:
Trend, Momentum, Volatility, Volume, Price-derived, Lag features.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """Calculate technical analysis indicators for stock data."""

    @staticmethod
    def calculate_all(df: pd.DataFrame, sentiment_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Calculate all technical indicators on OHLCV DataFrame.
        Expects columns: open, high, low, close, adj_close, volume
        Optional: sentiment_df with 'date' and 'avg_sentiment' columns.
        """
        df = df.copy()

        # Ensure lowercase columns
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]

        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"].astype(float)

        # ========== TREND (9 features) ==========

        # Simple Moving Averages
        df["sma_20"] = close.rolling(window=20).mean()
        df["sma_50"] = close.rolling(window=50).mean()
        df["sma_200"] = close.rolling(window=200).mean()

        # Exponential Moving Averages
        df["ema_12"] = close.ewm(span=12, adjust=False).mean()
        df["ema_26"] = close.ewm(span=26, adjust=False).mean()

        # MACD
        df["macd"] = df["ema_12"] - df["ema_26"]
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_histogram"] = df["macd"] - df["macd_signal"]

        # ADX (Average Directional Index)
        df["adx"] = TechnicalIndicators._calculate_adx(high, low, close, period=14)

        # ========== MOMENTUM (5 features) ==========

        # RSI (Relative Strength Index)
        df["rsi_14"] = TechnicalIndicators._calculate_rsi(close, period=14)

        # Stochastic Oscillator
        low_14 = low.rolling(window=14).min()
        high_14 = high.rolling(window=14).max()
        df["stoch_k"] = ((close - low_14) / (high_14 - low_14)) * 100
        df["stoch_d"] = df["stoch_k"].rolling(window=3).mean()

        # Williams %R
        df["williams_r"] = ((high_14 - close) / (high_14 - low_14)) * -100

        # Rate of Change
        df["roc_10"] = ((close - close.shift(10)) / close.shift(10)) * 100

        # ========== VOLATILITY (5 features) ==========

        # Bollinger Bands
        df["bb_middle"] = close.rolling(window=20).mean()
        bb_std = close.rolling(window=20).std()
        df["bb_upper"] = df["bb_middle"] + 2 * bb_std
        df["bb_lower"] = df["bb_middle"] - 2 * bb_std

        # ATR (Average True Range)
        df["atr_14"] = TechnicalIndicators._calculate_atr(high, low, close, period=14)

        # Historical Volatility (20-day)
        df["hist_volatility_20"] = close.pct_change().rolling(window=20).std() * np.sqrt(252)

        # ========== VOLUME (4 features) ==========

        # OBV (On-Balance Volume)
        df["obv"] = TechnicalIndicators._calculate_obv(close, volume)

        # Volume SMA
        df["volume_sma_20"] = volume.rolling(window=20).mean()

        # Volume Ratio
        df["volume_ratio"] = volume / df["volume_sma_20"]

        # VWAP (Volume Weighted Average Price)
        typical_price = (high + low + close) / 3
        cumulative_tp_vol = (typical_price * volume).cumsum()
        cumulative_vol = volume.cumsum()
        df["vwap"] = cumulative_tp_vol / cumulative_vol

        # ========== PRICE-DERIVED (3 features) ==========

        # Daily Return
        df["daily_return"] = close.pct_change()

        # Log Return
        df["log_return"] = np.log(close / close.shift(1))

        # Price / SMA50 Ratio
        df["price_sma50_ratio"] = close / df["sma_50"]

        # ========== LAG FEATURES (7 features) ==========

        for lag in [1, 3, 5, 10]:
            df[f"close_lag_{lag}"] = close.shift(lag)

        for lag in [1, 3, 5]:
            df[f"return_lag_{lag}"] = df["daily_return"].shift(lag)

        # ========== MOMENTUM REGIME (3 features) ==========

        # Multi-timeframe momentum
        df["momentum_5d"] = close.pct_change(5)    # short-term
        df["momentum_20d"] = close.pct_change(20)   # medium-term

        # RSI divergence: when price makes new high but RSI doesn't
        price_change_20 = close.diff(20)
        rsi_change_20 = df["rsi_14"].diff(20)
        df["rsi_divergence"] = np.where(
            (price_change_20 > 0) & (rsi_change_20 < 0), -1.0,  # bearish divergence
            np.where(
                (price_change_20 < 0) & (rsi_change_20 > 0), 1.0,  # bullish divergence
                0.0
            )
        )

        # ========== VOLATILITY REGIME (3 features) ==========

        # ATR / price ratio (normalized vol level, price-independent)
        df["volatility_regime"] = df["atr_14"] / close

        # Bollinger Band width (squeeze/expansion detector)
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]

        # Price position within bands (0 = lower, 1 = upper)
        bb_range = df["bb_upper"] - df["bb_lower"]
        df["bb_position"] = np.where(
            bb_range > 0,
            (close - df["bb_lower"]) / bb_range,
            0.5
        )

        # ========== PRICE STRUCTURE (3 features) ==========

        # Price / SMA20 ratio
        df["price_sma20_ratio"] = close / df["sma_20"]

        # Price / SMA200 ratio (long-term trend position)
        df["price_sma200_ratio"] = close / df["sma_200"]

        # SMA20/SMA50 cross signal (golden/death cross)
        df["sma20_sma50_cross"] = np.where(
            df["sma_20"] > df["sma_50"], 1.0,
            np.where(df["sma_20"] < df["sma_50"], -1.0, 0.0)
        )

        # ========== CALENDAR (2 features) ==========

        if isinstance(df.index, pd.DatetimeIndex):
            df["day_of_week"] = df.index.dayofweek / 4.0       # normalized 0-1
            df["month_of_year"] = (df.index.month - 1) / 11.0  # normalized 0-1
        else:
            df["day_of_week"] = 0.5
            df["month_of_year"] = 0.5

        # ========== SENTIMENT (1 feature) ==========
        if sentiment_df is not None and len(sentiment_df) > 0:
            # Ensure sentiment_df has date index for merging
            sent = sentiment_df.copy()
            if 'date' in sent.columns:
                sent['date'] = pd.to_datetime(sent['date'])
                sent = sent.set_index('date')
            # Merge on index (date)
            if 'avg_sentiment' in sent.columns:
                df = df.join(sent[['avg_sentiment']], how='left')
                df.rename(columns={'avg_sentiment': 'sentiment_score'}, inplace=True)
            elif 'sentiment_score' in sent.columns:
                df = df.join(sent[['sentiment_score']], how='left')
            # Forward-fill and fill remaining NaN with 0 (neutral)
            if 'sentiment_score' in df.columns:
                df['sentiment_score'] = df['sentiment_score'].ffill().fillna(0.0)
        else:
            df['sentiment_score'] = 0.0

        n_features = len([c for c in df.columns if c not in ['open', 'high', 'low', 'close', 'adj_close', 'volume']])
        logger.info("Calculated %d technical indicators", n_features)
        return df

    @staticmethod
    def _calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI."""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()

        # Use exponential smoothing after initial SMA
        for i in range(period, len(avg_gain)):
            avg_gain.iloc[i] = (avg_gain.iloc[i - 1] * (period - 1) + gain.iloc[i]) / period
            avg_loss.iloc[i] = (avg_loss.iloc[i - 1] * (period - 1) + loss.iloc[i]) / period

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def _calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate ADX (Average Directional Index)."""
        plus_dm = high.diff()
        minus_dm = -low.diff()

        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0

        # When +DM > -DM, -DM = 0 and vice versa
        mask = plus_dm > minus_dm
        minus_dm[mask] = 0
        plus_dm[~mask] = 0

        tr = TechnicalIndicators._true_range(high, low, close)
        atr = tr.rolling(window=period).mean()

        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        return adx

    @staticmethod
    def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """Calculate True Range."""
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    @staticmethod
    def _calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate ATR."""
        tr = TechnicalIndicators._true_range(high, low, close)
        return tr.rolling(window=period).mean()

    @staticmethod
    def _calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate On-Balance Volume."""
        obv = pd.Series(index=close.index, dtype=float)
        obv.iloc[0] = volume.iloc[0]
        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i - 1]:
                obv.iloc[i] = obv.iloc[i - 1] + volume.iloc[i]
            elif close.iloc[i] < close.iloc[i - 1]:
                obv.iloc[i] = obv.iloc[i - 1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i - 1]
        return obv

    @staticmethod
    def get_feature_columns() -> list[str]:
        """Return list of all feature column names."""
        return [
            # Trend
            "sma_20", "sma_50", "sma_200", "ema_12", "ema_26",
            "macd", "macd_signal", "macd_histogram", "adx",
            # Momentum
            "rsi_14", "stoch_k", "stoch_d", "williams_r", "roc_10",
            # Volatility
            "bb_upper", "bb_middle", "bb_lower", "atr_14", "hist_volatility_20",
            # Volume
            "obv", "volume_sma_20", "volume_ratio", "vwap",
            # Price-derived
            "daily_return", "log_return", "price_sma50_ratio",
            # Lag
            "close_lag_1", "close_lag_3", "close_lag_5", "close_lag_10",
            "return_lag_1", "return_lag_3", "return_lag_5",
            # Momentum Regime
            "momentum_5d", "momentum_20d", "rsi_divergence",
            # Volatility Regime
            "volatility_regime", "bb_width", "bb_position",
            # Price Structure
            "price_sma20_ratio", "price_sma200_ratio", "sma20_sma50_cross",
            # Calendar
            "day_of_week", "month_of_year",
            # Sentiment
            "sentiment_score",
        ]
