"""
Data collection from yfinance and simulated fund NAV calculation.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional
import logging

from config.settings import STOCK_UNIVERSE, FUND_PROFILES, DATA_SETTINGS
from data.storage import DatabaseManager

logger = logging.getLogger(__name__)


class StockDataCollector:
    """Collect stock OHLCV data from yfinance for SET stocks."""

    def __init__(self, db: DatabaseManager):
        self.db = db

    def collect_stock(
        self, symbol: str, period: str = "5y",
        start: Optional[str] = None, end: Optional[str] = None,
    ) -> pd.DataFrame:
        """Download OHLCV data for a single stock."""
        try:
            ticker = yf.Ticker(symbol)
            if start and end:
                df = ticker.history(start=start, end=end, auto_adjust=False)
            else:
                df = ticker.history(period=period, auto_adjust=False)

            if df.empty:
                logger.warning("No data returned for %s", symbol)
                return pd.DataFrame()

            # Flatten multi-level columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # Keep standard columns
            expected_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
            for col in expected_cols:
                if col not in df.columns:
                    if col == "Adj Close":
                        df["Adj Close"] = df["Close"]
                    else:
                        df[col] = 0

            df = df[expected_cols]
            df = df.dropna(subset=["Close"])

            logger.info("Collected %d rows for %s", len(df), symbol)
            return df

        except Exception as e:
            logger.error("Error collecting %s: %s", symbol, e)
            return pd.DataFrame()

    def collect_multiple(
        self, symbols: list[str], period: str = "5y",
    ) -> dict[str, pd.DataFrame]:
        """Collect data for multiple stocks."""
        results = {}
        for symbol in symbols:
            df = self.collect_stock(symbol, period=period)
            if not df.empty:
                count = self.db.store_stock_prices(symbol, df)
                results[symbol] = df
                logger.info("Stored %d rows for %s", count, symbol)
        return results

    def collect_all(self, period: str = "5y") -> dict[str, pd.DataFrame]:
        """Collect all stocks in the universe."""
        symbols = list(STOCK_UNIVERSE.keys())
        return self.collect_multiple(symbols, period)


class FundNAVCalculator:
    """Calculate simulated fund NAV from weighted stock prices."""

    def __init__(self, db: DatabaseManager):
        self.db = db

    def calculate_nav(self, fund_name: str) -> pd.DataFrame:
        """
        Calculate NAV for a fund based on its holdings.
        NAV = weighted average of constituent stock adj_close prices,
        normalized to start at 10.0 (standard starting NAV).
        """
        if fund_name not in FUND_PROFILES:
            logger.error("Fund %s not found in profiles", fund_name)
            return pd.DataFrame()

        profile = FUND_PROFILES[fund_name]
        holdings = profile["holdings"]

        # Collect adjusted close for all holdings
        all_prices = {}
        for symbol, weight in holdings.items():
            df = self.db.get_stock_prices(symbol)
            if not df.empty:
                all_prices[symbol] = df["adj_close"]

        if not all_prices:
            logger.warning("No price data available for fund %s", fund_name)
            return pd.DataFrame()

        # Combine into a single DataFrame, align dates
        prices_df = pd.DataFrame(all_prices)
        prices_df = prices_df.dropna()

        if prices_df.empty:
            return pd.DataFrame()

        # Calculate weighted returns
        returns = prices_df.pct_change().dropna()
        weighted_return = sum(
            returns[sym] * w for sym, w in holdings.items() if sym in returns.columns
        )

        # Build NAV from cumulative returns, starting at 10.0
        nav_series = (1 + weighted_return).cumprod() * 10.0

        # Create result DataFrame
        nav_df = pd.DataFrame({
            "date": nav_series.index,
            "nav": nav_series.values,
            "daily_return": weighted_return.values,
        })

        # Store in database
        count = self.db.store_fund_nav(fund_name, nav_df)
        logger.info("Calculated and stored %d NAV records for %s", count, fund_name)

        return nav_df

    def calculate_all_funds(self) -> dict[str, pd.DataFrame]:
        """Calculate NAV for all defined funds."""
        results = {}
        for fund_name in FUND_PROFILES:
            df = self.calculate_nav(fund_name)
            if not df.empty:
                results[fund_name] = df
        return results
