"""
Fund routes: list funds, NAV history.
"""

from fastapi import APIRouter, Query, HTTPException
from typing import Optional

from config.settings import FUND_PROFILES, STOCK_UNIVERSE
from schemas.api_schemas import FundInfo, FundHolding, FundNAVResponse, FundNAV
from api.main import app_state
from data.collector import FundNAVCalculator

router = APIRouter()


@router.get("", response_model=list[FundInfo])
async def list_funds():
    """Get list of all fund profiles."""
    funds = []
    for fund_key, profile in FUND_PROFILES.items():
        holdings = []
        for symbol, weight in profile["holdings"].items():
            name = STOCK_UNIVERSE.get(symbol, {}).get("name", symbol)
            holdings.append(FundHolding(symbol=symbol, name=name, weight=round(weight, 2)))

        funds.append(FundInfo(
            fund_name=fund_key,
            display_name=profile["name"],
            style=profile["style"],
            description=profile["description"],
            holdings=holdings,
            benchmark=profile["benchmark"],
        ))
    return funds


@router.get("/{fund_name}/nav", response_model=FundNAVResponse)
async def get_fund_nav(
    fund_name: str,
    start: Optional[str] = Query(None),
    end: Optional[str] = Query(None),
):
    """Get NAV history for a fund."""
    if fund_name not in FUND_PROFILES:
        raise HTTPException(404, f"Fund '{fund_name}' not found")

    db = app_state["db"]
    if not db:
        raise HTTPException(500, "Database not initialized")

    # Try to get from DB first
    df = db.get_fund_nav(fund_name, start, end)

    # If no data, calculate it
    if df.empty:
        calculator = FundNAVCalculator(db)
        df = calculator.calculate_nav(fund_name)
        if start:
            df = df[df["date"] >= start]
        if end:
            df = df[df["date"] <= end]

    if df.empty:
        raise HTTPException(404, f"No NAV data for {fund_name}. Collect stock data first.")

    nav_history = []
    for _, row in df.iterrows():
        nav_history.append(FundNAV(
            date=row["date"].date() if hasattr(row["date"], 'date') else row["date"],
            nav=round(float(row["nav"]), 4),
            daily_return=round(float(row.get("daily_return", 0)), 6),
        ))

    current_nav = float(df["nav"].iloc[-1]) if len(df) > 0 else None
    first_nav = float(df["nav"].iloc[0]) if len(df) > 0 else None
    total_return = ((current_nav - first_nav) / first_nav) if first_nav and first_nav > 0 else None

    return FundNAVResponse(
        fund_name=fund_name,
        nav_history=nav_history,
        current_nav=round(current_nav, 4) if current_nav else None,
        total_return=round(total_return, 4) if total_return else None,
        count=len(nav_history),
    )
