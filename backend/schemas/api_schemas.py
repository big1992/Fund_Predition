"""
Pydantic schemas for API request/response validation.
"""

import re
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from datetime import date, datetime
from typing import Optional, Literal

class APIBaseModel(BaseModel):
    model_config = ConfigDict(protected_namespaces=())


# ============== Stock Schemas ==============

class StockInfo(APIBaseModel):
    symbol: str
    name: str
    sector: str
    market: Optional[str] = None


class StockPrice(APIBaseModel):
    date: date
    open: float
    high: float
    low: float
    close: float
    adj_close: float
    volume: int


class StockPriceResponse(APIBaseModel):
    symbol: str
    prices: list[StockPrice]
    count: int


class IndicatorData(APIBaseModel):
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


class IndicatorResponse(APIBaseModel):
    symbol: str
    indicators: list[IndicatorData]
    count: int


class CollectRequest(APIBaseModel):
    symbols: list[str] = Field(default=[], description="Stock symbols to collect. Empty = all")
    period: Literal["1y", "2y", "5y", "max"] = Field(default="5y", description="Data period: 1y, 2y, 5y, max")


class CollectResponse(APIBaseModel):
    status: str
    symbols_collected: int
    records_stored: int
    message: str


# ============== Fund Schemas ==============

class FundHolding(APIBaseModel):
    symbol: str
    name: str
    weight: float


class FundInfo(APIBaseModel):
    fund_name: str
    display_name: str
    style: str
    description: str
    holdings: list[FundHolding]
    benchmark: str


class FundNAV(APIBaseModel):
    date: date
    nav: float
    daily_return: Optional[float] = None


class FundNAVResponse(APIBaseModel):
    fund_name: str
    nav_history: list[FundNAV]
    current_nav: Optional[float] = None
    total_return: Optional[float] = None
    count: int


# ============== Prediction Schemas ==============

class PredictionPoint(APIBaseModel):
    date: date
    predicted_price: float
    predicted_low: Optional[float] = None
    predicted_high: Optional[float] = None
    uncertainty_pct: Optional[float] = None
    confidence: float  # 0-100


class PredictionResponse(APIBaseModel):
    symbol: str
    model: str
    current_price: float
    predictions: list[PredictionPoint]
    signal: Literal["BUY", "SELL", "HOLD"]
    signal_reason: str
    confidence: float
    model_disagreement_pct: Optional[float] = None
    confidence_band: Optional[Literal["low", "medium", "high"]] = None


class TrainRequest(APIBaseModel):
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


class TrainResponse(APIBaseModel):
    status: str
    models_trained: list[str]
    metrics: dict
    message: str


# ============== Portfolio Schemas ==============

class PortfolioRequest(APIBaseModel):
    symbols: list[str] = Field(min_length=2, description="At least 2 symbols")
    method: Literal["max_sharpe", "min_volatility", "risk_parity"] = "max_sharpe"
    risk_level: float = Field(default=0.5, ge=0.0, le=1.0)


class FrontierPoint(APIBaseModel):
    volatility: float
    expected_return: float
    sharpe_ratio: float


class PortfolioResponse(APIBaseModel):
    method: str
    weights: dict[str, float]
    expected_return: float
    volatility: float
    sharpe_ratio: float
    frontier: list[FrontierPoint]


# ============== Backtest Schemas ==============

class BacktestRequest(APIBaseModel):
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


class Trade(APIBaseModel):
    date: date
    action: Literal["BUY", "SELL"]
    price: float
    shares: int
    value: float


class BacktestResponse(APIBaseModel):
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

class ModelMetrics(APIBaseModel):
    model_name: str
    symbol: str
    rmse: float
    mae: float
    mape: float
    directional_accuracy: float
    r_squared: float
    baseline_type: Optional[str] = None
    total_return: Optional[float] = None
    max_drawdown: Optional[float] = None
    description: Optional[str] = None


class FeatureImportance(APIBaseModel):
    feature: str
    importance: float


class ModelPerformanceResponse(APIBaseModel):
    models: list[ModelMetrics]


class FeatureImportanceResponse(APIBaseModel):
    symbol: str
    model: str
    features: list[FeatureImportance]


class ValidationReportCard(APIBaseModel):
    model_name: str
    status: str
    rmse: Optional[float] = None
    mae: Optional[float] = None
    mape: Optional[float] = None
    directional_accuracy: Optional[float] = None
    r_squared: Optional[float] = None
    baseline_name: Optional[str] = None
    baseline_mape: Optional[float] = None
    mape_improvement_pct: Optional[float] = None
    caveats: list[str] = []


class ValidationReportResponse(APIBaseModel):
    symbol: str
    generated_at_utc: str
    cards: list[ValidationReportCard]


class ModelRegistryEntry(APIBaseModel):
    id: int
    symbol: str
    model_name: str
    version: int
    training_run_id: Optional[int] = None
    train_start_date: Optional[str] = None
    train_end_date: Optional[str] = None
    feature_version: Optional[str] = None
    artifact_path: Optional[str] = None
    created_at: Optional[str] = None
    is_deployed: bool = False
    is_best_run: bool = False
    metrics: dict = {}
    params: dict = {}


class ModelRegistryResponse(APIBaseModel):
    symbol: str
    entries: list[ModelRegistryEntry]


class DriftStatusEntry(APIBaseModel):
    symbol: str
    model_name: str
    status: str
    drift_score: float
    should_retrain: bool
    reasons: list[str] = []
    metrics: dict = {}
    created_at: Optional[str] = None


class DriftStatusResponse(APIBaseModel):
    symbol: str
    entries: list[DriftStatusEntry]


# ============== System Schemas ==============

class HealthResponse(APIBaseModel):
    status: str
    version: str
    uptime_seconds: float
    models_loaded: list[str]
    data_available: bool


class ErrorDetail(APIBaseModel):
    field: Optional[str] = None
    message: str


class ErrorResponse(APIBaseModel):
    error_code: str
    message: str
    details: list[ErrorDetail] = []
