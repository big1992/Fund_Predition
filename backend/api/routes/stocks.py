"""
Stock data routes: list, prices, indicators, collect.
"""

from fastapi import APIRouter, Query, HTTPException
from typing import Optional

from config.settings import STOCK_UNIVERSE
from schemas.api_schemas import (
    StockInfo, StockPriceResponse, StockPrice,
    IndicatorResponse, IndicatorData,
    CollectRequest, CollectResponse,
)
from api.main import app_state
from data.collector import StockDataCollector
from features.technical_indicators import TechnicalIndicators

router = APIRouter()


@router.get("", response_model=list[StockInfo])
async def list_stocks(sector: Optional[str] = None):
    """Get list of tracked stocks."""
    stocks = []
    for symbol, info in STOCK_UNIVERSE.items():
        if sector and info["sector"].lower() != sector.lower():
            continue
        stocks.append(StockInfo(symbol=symbol, name=info["name"], sector=info["sector"], market=info.get("market", "")))
    return stocks


@router.get("/{symbol}/prices", response_model=StockPriceResponse)
async def get_prices(
    symbol: str,
    start: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
):
    """Get historical prices for a stock."""
    db = app_state["db"]
    if not db:
        raise HTTPException(500, "Database not initialized")

    df = db.get_stock_prices(symbol, start, end)
    if df.empty:
        raise HTTPException(404, f"No data found for {symbol}")

    prices = []
    for idx, row in df.iterrows():
        prices.append(StockPrice(
            date=idx.date() if hasattr(idx, 'date') else idx,
            open=round(row["open"], 2),
            high=round(row["high"], 2),
            low=round(row["low"], 2),
            close=round(row["close"], 2),
            adj_close=round(row["adj_close"], 2),
            volume=int(row["volume"]),
        ))

    return StockPriceResponse(symbol=symbol, prices=prices, count=len(prices))


@router.get("/{symbol}/indicators", response_model=IndicatorResponse)
async def get_indicators(
    symbol: str,
    start: Optional[str] = Query(None),
    end: Optional[str] = Query(None),
):
    """Get technical indicators for a stock."""
    db = app_state["db"]
    if not db:
        raise HTTPException(500, "Database not initialized")

    # Get price data
    df = db.get_stock_prices(symbol, start, end)
    if df.empty:
        raise HTTPException(404, f"No data found for {symbol}")

    # Calculate indicators
    df_ind = TechnicalIndicators.calculate_all(df)
    df_ind = df_ind.dropna(subset=["sma_20"])  # Drop early rows without enough data

    indicators = []
    for idx, row in df_ind.iterrows():
        indicators.append(IndicatorData(
            date=idx.date() if hasattr(idx, 'date') else idx,
            close=round(float(row.get("close", 0)), 2),
            sma_20=_safe_round(row.get("sma_20")),
            sma_50=_safe_round(row.get("sma_50")),
            ema_12=_safe_round(row.get("ema_12")),
            ema_26=_safe_round(row.get("ema_26")),
            rsi_14=_safe_round(row.get("rsi_14")),
            macd=_safe_round(row.get("macd")),
            macd_signal=_safe_round(row.get("macd_signal")),
            macd_histogram=_safe_round(row.get("macd_histogram")),
            bb_upper=_safe_round(row.get("bb_upper")),
            bb_middle=_safe_round(row.get("bb_middle")),
            bb_lower=_safe_round(row.get("bb_lower")),
            atr_14=_safe_round(row.get("atr_14")),
            obv=_safe_round(row.get("obv")),
            adx=_safe_round(row.get("adx")),
            stoch_k=_safe_round(row.get("stoch_k")),
            stoch_d=_safe_round(row.get("stoch_d")),
            williams_r=_safe_round(row.get("williams_r")),
            daily_return=_safe_round(row.get("daily_return"), 6),
            volume_sma_20=_safe_round(row.get("volume_sma_20")),
        ))

    return IndicatorResponse(symbol=symbol, indicators=indicators, count=len(indicators))


@router.post("/collect", response_model=CollectResponse)
async def collect_data(request: CollectRequest):
    """Trigger data collection for stocks."""
    db = app_state["db"]
    if not db:
        raise HTTPException(500, "Database not initialized")

    collector = StockDataCollector(db)

    symbols = request.symbols if request.symbols else list(STOCK_UNIVERSE.keys())
    results = collector.collect_multiple(symbols, period=request.period)

    total_records = sum(len(df) for df in results.values())

    return CollectResponse(
        status="success",
        symbols_collected=len(results),
        records_stored=total_records,
        message=f"Collected data for {len(results)} symbols ({total_records} records)",
    )


def _safe_round(val, decimals=2):
    """Safely round a value that might be None or NaN."""
    import pandas as pd
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    return round(float(val), decimals)
