"""
Portfolio optimization using Modern Portfolio Theory.
Efficient Frontier, Max Sharpe, Min Volatility, Risk Parity.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import logging
from typing import Optional

from config.settings import PORTFOLIO_SETTINGS

logger = logging.getLogger(__name__)


class PortfolioOptimizer:
    """Modern Portfolio Theory optimizer."""

    def __init__(self, params: dict = None):
        self.params = params or PORTFOLIO_SETTINGS
        self.risk_free_rate = self.params["risk_free_rate"]

    def optimize(
        self,
        returns_df: pd.DataFrame,
        method: str = "max_sharpe",
        risk_level: float = 0.5,
    ) -> dict:
        """
        Optimize portfolio allocation.

        Args:
            returns_df: DataFrame of daily returns (columns = stock symbols)
            method: Optimization method
            risk_level: 0.0 (conservative) to 1.0 (aggressive)

        Returns:
            dict with weights, metrics, and efficient frontier
        """
        symbols = returns_df.columns.tolist()
        n_assets = len(symbols)

        # Calculate expected returns and covariance
        mean_returns = returns_df.mean() * 252  # Annualize
        cov_matrix = returns_df.cov() * 252

        # Get optimal weights
        if method == "max_sharpe":
            weights = self._max_sharpe(mean_returns, cov_matrix, n_assets)
        elif method == "min_volatility":
            weights = self._min_volatility(mean_returns, cov_matrix, n_assets)
        elif method == "risk_parity":
            weights = self._risk_parity(mean_returns, cov_matrix, n_assets)
        else:
            weights = np.array([1 / n_assets] * n_assets)  # Equal weight

        # Blend with risk level (interpolate between min_vol and max_sharpe)
        if 0 < risk_level < 1 and method == "max_sharpe":
            min_vol_weights = self._min_volatility(mean_returns, cov_matrix, n_assets)
            weights = (1 - risk_level) * min_vol_weights + risk_level * weights

        # Calculate portfolio metrics
        port_return = float(np.sum(mean_returns * weights))
        port_vol = float(np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))))
        sharpe = (port_return - self.risk_free_rate) / port_vol if port_vol > 0 else 0

        # Generate efficient frontier
        frontier = self._efficient_frontier(mean_returns, cov_matrix, n_assets)

        # Build response
        weights_dict = {sym: round(float(w), 4) for sym, w in zip(symbols, weights)}

        result = {
            "method": method,
            "weights": weights_dict,
            "expected_return": round(port_return, 4),
            "volatility": round(port_vol, 4),
            "sharpe_ratio": round(sharpe, 4),
            "frontier": frontier,
        }

        logger.info(
            "Portfolio optimized (%s): return=%.2f%%, vol=%.2f%%, sharpe=%.2f",
            method, port_return * 100, port_vol * 100, sharpe,
        )
        return result

    def _max_sharpe(
        self, mean_returns: pd.Series, cov_matrix: pd.DataFrame, n_assets: int
    ) -> np.ndarray:
        """Find portfolio with maximum Sharpe ratio."""

        def neg_sharpe(weights):
            port_return = np.sum(mean_returns * weights)
            port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return -(port_return - self.risk_free_rate) / port_vol if port_vol > 0 else 0

        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
        bounds = tuple(
            (self.params["min_weight"], self.params["max_weight"])
            for _ in range(n_assets)
        )
        init = np.array([1 / n_assets] * n_assets)

        result = minimize(
            neg_sharpe, init,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )
        return result.x if result.success else init

    def _min_volatility(
        self, mean_returns: pd.Series, cov_matrix: pd.DataFrame, n_assets: int
    ) -> np.ndarray:
        """Find minimum volatility portfolio."""

        def portfolio_vol(weights):
            return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
        bounds = tuple(
            (self.params["min_weight"], self.params["max_weight"])
            for _ in range(n_assets)
        )
        init = np.array([1 / n_assets] * n_assets)

        result = minimize(
            portfolio_vol, init,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )
        return result.x if result.success else init

    def _risk_parity(
        self, mean_returns: pd.Series, cov_matrix: pd.DataFrame, n_assets: int
    ) -> np.ndarray:
        """Risk parity: equal risk contribution from each asset."""

        def risk_contribution_error(weights):
            port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            marginal_contrib = np.dot(cov_matrix, weights)
            risk_contrib = weights * marginal_contrib / port_vol
            target_risk = port_vol / n_assets
            return np.sum((risk_contrib - target_risk) ** 2)

        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
        bounds = tuple(
            (self.params["min_weight"], self.params["max_weight"])
            for _ in range(n_assets)
        )
        init = np.array([1 / n_assets] * n_assets)

        result = minimize(
            risk_contribution_error, init,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )
        return result.x if result.success else init

    def _efficient_frontier(
        self, mean_returns: pd.Series, cov_matrix: pd.DataFrame, n_assets: int,
        n_points: int = 50,
    ) -> list[dict]:
        """Calculate efficient frontier points."""
        frontier = []
        n_random = self.params["num_random_portfolios"]

        # Generate random portfolios
        for _ in range(n_random):
            weights = np.random.dirichlet(np.ones(n_assets))
            # Clip to bounds
            weights = np.clip(weights, self.params["min_weight"], self.params["max_weight"])
            weights /= weights.sum()

            port_return = float(np.sum(mean_returns * weights))
            port_vol = float(np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))))
            sharpe = (port_return - self.risk_free_rate) / port_vol if port_vol > 0 else 0

            frontier.append({
                "volatility": round(port_vol, 6),
                "expected_return": round(port_return, 6),
                "sharpe_ratio": round(sharpe, 4),
            })

        # Sort by volatility
        frontier.sort(key=lambda x: x["volatility"])

        # Subsample for response
        if len(frontier) > n_points:
            step = len(frontier) // n_points
            frontier = frontier[::step][:n_points]

        return frontier
