"""
Portfolio optimization routes.
"""

from fastapi import APIRouter, HTTPException
import pandas as pd

from schemas.api_schemas import PortfolioRequest, PortfolioResponse, FrontierPoint
from api.main import app_state
from portfolio.optimizer import PortfolioOptimizer

router = APIRouter()


@router.post("/optimize", response_model=PortfolioResponse)
async def optimize_portfolio(request: PortfolioRequest):
    """Optimize portfolio allocation."""
    db = app_state["db"]
    if not db:
        raise HTTPException(500, "Database not initialized")

    # Get returns for all requested symbols
    returns_dict = {}
    for symbol in request.symbols:
        df = db.get_stock_prices(symbol)
        if df.empty:
            continue
        returns_dict[symbol] = df["adj_close"].pct_change().dropna()

    if len(returns_dict) < 2:
        raise HTTPException(
            400,
            f"Need at least 2 stocks with data. Found: {list(returns_dict.keys())}",
        )

    # Align dates
    returns_df = pd.DataFrame(returns_dict).dropna()

    if len(returns_df) < 30:
        raise HTTPException(400, "Not enough data points for optimization (need >= 30)")

    # Optimize
    optimizer = PortfolioOptimizer()
    result = optimizer.optimize(
        returns_df,
        method=request.method,
        risk_level=request.risk_level,
    )

    return PortfolioResponse(
        method=result["method"],
        weights=result["weights"],
        expected_return=result["expected_return"],
        volatility=result["volatility"],
        sharpe_ratio=result["sharpe_ratio"],
        frontier=[FrontierPoint(**p) for p in result["frontier"]],
    )
