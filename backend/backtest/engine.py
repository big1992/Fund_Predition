"""
Backtesting engine for evaluating prediction-based trading strategies.
"""

import numpy as np
import pandas as pd
import logging
from typing import Optional
from datetime import datetime

from config.settings import BACKTEST_SETTINGS

logger = logging.getLogger(__name__)


class BacktestEngine:
    """Simulate trading strategy based on ML predictions."""

    def __init__(self, params: dict = None):
        self.params = params or BACKTEST_SETTINGS
        self.initial_capital = self.params["initial_capital"]

    def run(
        self,
        prices: pd.Series,
        predictions: np.ndarray,
        dates: list,
        initial_capital: float = None,
    ) -> dict:
        """
        Run backtest simulation.

        Args:
            prices: Actual close prices (aligned with predictions)
            predictions: Predicted prices for next day
            dates: Corresponding dates
            initial_capital: Starting capital in THB
        """
        if initial_capital is None:
            initial_capital = self.initial_capital

        capital = initial_capital
        shares = 0
        position = "NONE"  # NONE, LONG
        trades = []
        equity_curve = []
        buy_threshold = self.params["buy_threshold"]
        sell_threshold = self.params["sell_threshold"]
        commission = self.params["commission"]

        prices_arr = prices.values if isinstance(prices, pd.Series) else np.array(prices)
        min_len = min(len(prices_arr), len(predictions), len(dates))

        for i in range(min_len - 1):  # -1 because we look at next day's actual price
            current_price = float(prices_arr[i])
            predicted_next = float(predictions[i])  # model's prediction for next period

            # Signal: compare prediction to current price (forward-looking)
            predicted_change = (predicted_next - current_price) / current_price if current_price > 0 else 0

            # Portfolio value
            portfolio_value = capital + (shares * current_price)
            equity_curve.append({
                "date": dates[i].strftime("%Y-%m-%d") if hasattr(dates[i], 'strftime') else str(dates[i]),
                "portfolio_value": round(portfolio_value, 2),
                "price": round(current_price, 2),
            })

            # Next day's actual price (for realistic P&L)
            next_price = float(prices_arr[i + 1])

            # Trading logic — execute at next day's price (realistic fill)
            if predicted_change > buy_threshold and position == "NONE":
                # BUY signal — fill at next day open (approximated by next close)
                fill_price = next_price
                buy_amount = capital * 0.95
                trade_cost = buy_amount * commission
                shares = int((buy_amount - trade_cost) / fill_price)
                if shares > 0:
                    cost = shares * fill_price + trade_cost
                    capital -= cost
                    position = "LONG"

                    trades.append({
                        "date": dates[i + 1].strftime("%Y-%m-%d") if hasattr(dates[i + 1], 'strftime') else str(dates[i + 1]),
                        "action": "BUY",
                        "price": round(fill_price, 2),
                        "shares": shares,
                        "value": round(cost, 2),
                    })

            elif predicted_change < sell_threshold and position == "LONG" and shares > 0:
                # SELL signal — fill at next day price
                fill_price = next_price
                sell_amount = shares * fill_price
                trade_cost = sell_amount * commission
                capital += sell_amount - trade_cost
                position = "NONE"

                trades.append({
                    "date": dates[i + 1].strftime("%Y-%m-%d") if hasattr(dates[i + 1], 'strftime') else str(dates[i + 1]),
                    "action": "SELL",
                    "price": round(fill_price, 2),
                    "shares": shares,
                    "value": round(sell_amount - trade_cost, 2),
                })
                shares = 0

        # Close any open position at end
        if shares > 0:
            final_price = float(prices_arr[min_len - 1])
            sell_amount = shares * final_price
            capital += sell_amount * (1 - commission)
            trades.append({
                "date": dates[min_len - 1].strftime("%Y-%m-%d") if hasattr(dates[min_len - 1], 'strftime') else str(dates[min_len - 1]),
                "action": "SELL",
                "price": round(final_price, 2),
                "shares": shares,
                "value": round(sell_amount, 2),
            })
            shares = 0

        # Calculate performance metrics
        final_capital = capital
        total_return = (final_capital - initial_capital) / initial_capital

        # Annualized return
        n_days = min_len
        n_years = n_days / 252 if n_days > 0 else 1
        annualized_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0

        # Sharpe ratio from equity curve
        if len(equity_curve) > 1:
            values = [e["portfolio_value"] for e in equity_curve]
            daily_returns = pd.Series(values).pct_change().dropna()
            sharpe = (daily_returns.mean() * 252) / (daily_returns.std() * np.sqrt(252)) if daily_returns.std() > 0 else 0
        else:
            sharpe = 0

        # Max drawdown
        max_drawdown = self._calculate_max_drawdown(equity_curve)

        # Win rate and profit factor
        win_rate, profit_factor = self._calculate_trade_metrics(trades)

        result = {
            "symbol": "",
            "model": "",
            "initial_capital": round(initial_capital, 2),
            "final_capital": round(final_capital, 2),
            "total_return": round(total_return, 4),
            "annualized_return": round(annualized_return, 4),
            "sharpe_ratio": round(float(sharpe), 4),
            "max_drawdown": round(max_drawdown, 4),
            "win_rate": round(win_rate, 2),
            "profit_factor": round(profit_factor, 2),
            "total_trades": len(trades),
            "trades": trades,
            "equity_curve": equity_curve,
        }

        logger.info(
            "Backtest complete: return=%.2f%%, sharpe=%.2f, drawdown=%.2f%%",
            total_return * 100, sharpe, max_drawdown * 100,
        )
        return result

    @staticmethod
    def _calculate_max_drawdown(equity_curve: list[dict]) -> float:
        """Calculate maximum drawdown from equity curve."""
        if not equity_curve:
            return 0.0

        values = [e["portfolio_value"] for e in equity_curve]
        peak = values[0]
        max_dd = 0.0

        for v in values:
            if v > peak:
                peak = v
            dd = (peak - v) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)

        return max_dd

    @staticmethod
    def _calculate_trade_metrics(trades: list[dict]) -> tuple[float, float]:
        """Calculate win rate and profit factor from trades."""
        if len(trades) < 2:
            return 0.0, 0.0

        # Pair buys and sells
        profits = []
        buy_price = 0
        for trade in trades:
            if trade["action"] == "BUY":
                buy_price = trade["price"]
            elif trade["action"] == "SELL" and buy_price > 0:
                profit = (trade["price"] - buy_price) / buy_price
                profits.append(profit)
                buy_price = 0

        if not profits:
            return 0.0, 0.0

        wins = [p for p in profits if p > 0]
        losses = [abs(p) for p in profits if p <= 0]

        win_rate = len(wins) / len(profits) * 100 if profits else 0
        total_wins = sum(wins) if wins else 0
        total_losses = max(sum(losses), 0.001)  # avoid division by zero
        profit_factor = total_wins / total_losses

        return win_rate, profit_factor
