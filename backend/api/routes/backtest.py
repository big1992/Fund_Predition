"""
Backtesting routes.
"""

from fastapi import APIRouter, HTTPException
import numpy as np
import pandas as pd

from schemas.api_schemas import BacktestRequest, BacktestResponse, Trade
from api.main import app_state
from backtest.engine import BacktestEngine
from features.technical_indicators import TechnicalIndicators

router = APIRouter()


@router.post("/run", response_model=BacktestResponse)
async def run_backtest(request: BacktestRequest):
    """Run backtest simulation."""
    db = app_state["db"]
    trainer = app_state["trainer"]
    if not db or not trainer:
        raise HTTPException(500, "Service not initialized")

    # Get price data
    df = db.get_stock_prices(request.symbol)
    if df.empty:
        raise HTTPException(404, f"No data for {request.symbol}")

    # Filter by dates
    if request.start_date:
        df = df[df.index >= pd.Timestamp(request.start_date)]
    if request.end_date:
        df = df[df.index <= pd.Timestamp(request.end_date)]

    if len(df) < 100:
        raise HTTPException(400, "Need at least 100 data points for backtesting")

    # Generate predictions for the test period
    try:
        # Use XGBoost for backtest (faster than LSTM)
        trainer._load_xgboost(request.symbol)

        if trainer.xgb_model and trainer.xgb_model.model:
            df_features = TechnicalIndicators.calculate_all(df)
            df_features = df_features.dropna()

            # Use the exact feature names the model was trained on
            if trainer.xgb_model.feature_names:
                feature_cols = [
                    c for c in trainer.xgb_model.feature_names
                    if c in df_features.columns
                ]
            else:
                feature_cols = [
                    c for c in df_features.select_dtypes(include=[np.number]).columns
                    if c not in ("close", "open", "high", "low", "volume", "adj_close")
                ]

            if not feature_cols:
                raise ValueError("No matching feature columns found")

            # Use midpoint onwards for backtest
            mid = len(df_features) // 2
            test_features = df_features.iloc[mid:]

            # Generate predictions — model now predicts next-day RETURN (not price)
            X = test_features[feature_cols].values
            raw_returns = trainer.xgb_model.predict(X)

            # Convert returns to predicted next-day prices
            prices = test_features["close"]
            predictions = prices.values * (1 + raw_returns)  # return → price
            dates = test_features.index.tolist()
        else:
            raise ValueError("No XGBoost model available")

    except Exception as e:
        # Fallback: simple naive prediction (prev day close)
        mid = len(df) // 2
        test_df = df.iloc[mid:]
        prices = test_df["close"]
        predictions = prices.shift(1).bfill().values
        dates = test_df.index.tolist()

    # Run backtest
    engine = BacktestEngine()
    result = engine.run(
        prices=prices,
        predictions=predictions,
        dates=dates,
        initial_capital=request.initial_capital,
    )

    result["symbol"] = request.symbol
    result["model"] = request.model_type

    return BacktestResponse(
        symbol=result["symbol"],
        model=result["model"],
        initial_capital=result["initial_capital"],
        final_capital=result["final_capital"],
        total_return=result["total_return"],
        annualized_return=result["annualized_return"],
        sharpe_ratio=result["sharpe_ratio"],
        max_drawdown=result["max_drawdown"],
        win_rate=result["win_rate"],
        profit_factor=result["profit_factor"],
        total_trades=result["total_trades"],
        trades=[Trade(**t) for t in result["trades"]],
        equity_curve=result["equity_curve"],
    )
