"""
Pydantic schemas for API request/response validation.
"""

import re
from pydantic import BaseModel, Field, field_validator, model_validator
from datetime import date, datetime
from typing import Optional, Literal


# ============== Stock Schemas ==============

class StockInfo(BaseModel):
    symbol: str
    name: str
    sector: str
    market: Optional[str] = None


class StockPrice(BaseModel):
    date: date
    open: float
    high: float
    low: float
    close: float
    adj_close: float
    volume: int


class StockPriceResponse(BaseModel):
    symbol: str
    prices: list[StockPrice]
    count: int


class IndicatorData(BaseModel):
    date: date
    close: float
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    ema_12: Optional[float] = None
    ema_26: Optional[float] = None
    rsi_14: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_histogram: Optional[float] = None
    bb_upper: Optional[float] = None
    bb_middle: Optional[float] = None
    bb_lower: Optional[float] = None
    atr_14: Optional[float] = None
    obv: Optional[float] = None
    adx: Optional[float] = None
    stoch_k: Optional[float] = None
    stoch_d: Optional[float] = None
    williams_r: Optional[float] = None
    daily_return: Optional[float] = None
    volume_sma_20: Optional[float] = None


class IndicatorResponse(BaseModel):
    symbol: str
    indicators: list[IndicatorData]
    count: int


class CollectRequest(BaseModel):
    symbols: list[str] = Field(default=[], description="Stock symbols to collect. Empty = all")
    period: Literal["1y", "2y", "5y", "max"] = Field(default="5y", description="Data period: 1y, 2y, 5y, max")


class CollectResponse(BaseModel):
    status: str
    symbols_collected: int
    records_stored: int
    message: str


# ============== Fund Schemas ==============

class FundHolding(BaseModel):
    symbol: str
    name: str
    weight: float


class FundInfo(BaseModel):
    fund_name: str
    display_name: str
    style: str
    description: str
    holdings: list[FundHolding]
    benchmark: str


class FundNAV(BaseModel):
    date: date
    nav: float
    daily_return: Optional[float] = None


class FundNAVResponse(BaseModel):
    fund_name: str
    nav_history: list[FundNAV]
    current_nav: Optional[float] = None
    total_return: Optional[float] = None
    count: int


# ============== Prediction Schemas ==============

class PredictionPoint(BaseModel):
    date: date
    predicted_price: float
    confidence: float  # 0-100


class PredictionResponse(BaseModel):
    symbol: str
    model: str
    current_price: float
    predictions: list[PredictionPoint]
    signal: Literal["BUY", "SELL", "HOLD"]
    signal_reason: str
    confidence: float


class TrainRequest(BaseModel):
    symbols: list[str] = Field(default=[], description="Symbols to train. Empty = all")
    model_type: Literal["lstm", "xgboost", "autogluon", "all"] = "all"
    walk_forward: bool = Field(default=False, description="Use walk-forward validation")

    @field_validator("symbols")
    @classmethod
    def validate_symbols(cls, values: list[str]) -> list[str]:
        symbol_re = re.compile(r"^[A-Z0-9][A-Z0-9.\-]{0,14}$")
        normalized = []
        for symbol in values:
            s = symbol.strip().upper()
            if not symbol_re.match(s):
                raise ValueError(f"Invalid symbol format: {symbol}")
            normalized.append(s)
        return normalized


class TrainResponse(BaseModel):
    status: str
    models_trained: list[str]
    metrics: dict
    message: str


# ============== Portfolio Schemas ==============

class PortfolioRequest(BaseModel):
    symbols: list[str] = Field(min_length=2, description="At least 2 symbols")
    method: Literal["max_sharpe", "min_volatility", "risk_parity"] = "max_sharpe"
    risk_level: float = Field(default=0.5, ge=0.0, le=1.0)


class FrontierPoint(BaseModel):
    volatility: float
    expected_return: float
    sharpe_ratio: float


class PortfolioResponse(BaseModel):
    method: str
    weights: dict[str, float]
    expected_return: float
    volatility: float
    sharpe_ratio: float
    frontier: list[FrontierPoint]


# ============== Backtest Schemas ==============

class BacktestRequest(BaseModel):
    symbol: str
    model_type: Literal["lstm", "xgboost", "ensemble"] = "ensemble"
    initial_capital: float = 1_000_000
    start_date: Optional[date] = None
    end_date: Optional[date] = None

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, value: str) -> str:
        s = value.strip().upper()
        if not re.match(r"^[A-Z0-9][A-Z0-9.\-]{0,14}$", s):
            raise ValueError("Invalid symbol format")
        return s

    @field_validator("initial_capital")
    @classmethod
    def validate_initial_capital(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("initial_capital must be > 0")
        return value

    @model_validator(mode="after")
    def validate_date_range(self):
        if self.start_date and self.end_date and self.start_date > self.end_date:
            raise ValueError("start_date must be <= end_date")
        return self


class Trade(BaseModel):
    date: date
    action: Literal["BUY", "SELL"]
    price: float
    shares: int
    value: float


class BacktestResponse(BaseModel):
    symbol: str
    model: str
    initial_capital: float
    final_capital: float
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    trades: list[Trade]
    equity_curve: list[dict]


# ============== Model Performance Schemas ==============

class ModelMetrics(BaseModel):
    model_name: str
    symbol: str
    rmse: float
    mae: float
    mape: float
    directional_accuracy: float
    r_squared: float


class FeatureImportance(BaseModel):
    feature: str
    importance: float


class ModelPerformanceResponse(BaseModel):
    models: list[ModelMetrics]


class FeatureImportanceResponse(BaseModel):
    symbol: str
    model: str
    features: list[FeatureImportance]


# ============== System Schemas ==============

class HealthResponse(BaseModel):
    status: str
    version: str
    uptime_seconds: float
    models_loaded: list[str]
    data_available: bool


class ErrorDetail(BaseModel):
    field: Optional[str] = None
    message: str


class ErrorResponse(BaseModel):
    error_code: str
    message: str
    details: list[ErrorDetail] = []
