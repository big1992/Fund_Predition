"""
Global configuration for the Fund Prediction System.
Uses Pydantic BaseSettings for environment variable support.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from pathlib import Path
from typing import Optional
import os


BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Application settings."""
    model_config = SettingsConfigDict(
        env_prefix="",
        env_file=".env",
        protected_namespaces=("settings_",),
    )

    # App
    app_name: str = "Fund Prediction System"
    app_version: str = "1.0.0"
    debug: bool = True

    # Database
    db_path: str = str(BASE_DIR / "database" / "fund_prediction.db")

    # Model storage
    model_dir: str = str(BASE_DIR / "saved_models")

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:3000"]

    # OpenAI
    openai_api_key: Optional[str] = None

    # NewsAPI
    newsapi_key: Optional[str] = None


# ---------- Stock Universe ----------

STOCK_UNIVERSE = {
    # ---- Thai Stocks (SET) ----
    "PTT.BK": {"name": "PTT PCL", "sector": "Energy", "market": "SET"},
    "ADVANC.BK": {"name": "Advanced Info Service", "sector": "Telecom", "market": "SET"},
    "SCC.BK": {"name": "Siam Cement Group", "sector": "Construction", "market": "SET"},
    "CPALL.BK": {"name": "CP ALL PCL", "sector": "Commerce", "market": "SET"},
    "AOT.BK": {"name": "Airports of Thailand", "sector": "Transport", "market": "SET"},
    "BDMS.BK": {"name": "Bangkok Dusit Medical", "sector": "Healthcare", "market": "SET"},
    "KBANK.BK": {"name": "Kasikornbank", "sector": "Banking", "market": "SET"},
    "SCB.BK": {"name": "SCB X PCL", "sector": "Banking", "market": "SET"},
    "GULF.BK": {"name": "Gulf Energy Development", "sector": "Energy", "market": "SET"},
    "DELTA.BK": {"name": "Delta Electronics Thai", "sector": "Electronics", "market": "SET"},
    "TRUE.BK": {"name": "True Corporation", "sector": "Telecom", "market": "SET"},
    "BEM.BK": {"name": "Bangkok Expressway", "sector": "Transport", "market": "SET"},
    "MINT.BK": {"name": "Minor International", "sector": "Hospitality", "market": "SET"},
    "CPN.BK": {"name": "Central Pattana", "sector": "Property", "market": "SET"},
    "HMPRO.BK": {"name": "Home Product Center", "sector": "Commerce", "market": "SET"},

    # ---- US Stocks (NASDAQ / NYSE) ----
    "AAPL": {"name": "Apple Inc.", "sector": "Technology", "market": "NASDAQ"},
    "MSFT": {"name": "Microsoft Corp.", "sector": "Technology", "market": "NASDAQ"},
    "GOOGL": {"name": "Alphabet Inc.", "sector": "Technology", "market": "NASDAQ"},
    "AMZN": {"name": "Amazon.com Inc.", "sector": "Consumer", "market": "NASDAQ"},
    "NVDA": {"name": "NVIDIA Corp.", "sector": "Semiconductors", "market": "NASDAQ"},
    "TSLA": {"name": "Tesla Inc.", "sector": "Automotive", "market": "NASDAQ"},
    "META": {"name": "Meta Platforms", "sector": "Technology", "market": "NASDAQ"},
    "JPM": {"name": "JPMorgan Chase", "sector": "Banking", "market": "NYSE"},
    "V": {"name": "Visa Inc.", "sector": "Fintech", "market": "NYSE"},
    "JNJ": {"name": "Johnson & Johnson", "sector": "Healthcare", "market": "NYSE"},
    "XOM": {"name": "Exxon Mobil", "sector": "Energy", "market": "NYSE"},
    "WMT": {"name": "Walmart Inc.", "sector": "Retail", "market": "NYSE"},
    "TSM": {"name": "Taiwan Semiconductor", "sector": "Semiconductors", "market": "NYSE"},
    "BRK-B": {"name": "Berkshire Hathaway B", "sector": "Diversified", "market": "NYSE"},
    "COIN": {"name": "Coinbase Global", "sector": "Crypto/Fintech", "market": "NASDAQ"},
}

# ---------- Fund Profiles ----------

FUND_PROFILES = {
    "TH-Growth": {
        "name": "Thai Growth Fund",
        "style": "Growth",
        "description": "เน้นหุ้นเติบโตสูง เทคโนโลยีและพลังงาน",
        "holdings": {
            "ADVANC.BK": 0.25,
            "DELTA.BK": 0.25,
            "TRUE.BK": 0.15,
            "GULF.BK": 0.20,
            "AOT.BK": 0.15,
        },
        "benchmark": "SET",
    },
    "TH-Value": {
        "name": "Thai Value Fund",
        "style": "Value",
        "description": "เน้นหุ้นมูลค่า พื้นฐานดี ราคาต่ำกว่ามูลค่า",
        "holdings": {
            "PTT.BK": 0.25,
            "SCC.BK": 0.20,
            "KBANK.BK": 0.25,
            "SCB.BK": 0.15,
            "HMPRO.BK": 0.15,
        },
        "benchmark": "SET",
    },
    "TH-Dividend": {
        "name": "Thai Dividend Fund",
        "style": "Dividend",
        "description": "เน้นหุ้นปันผลสูง รายได้สม่ำเสมอ",
        "holdings": {
            "CPALL.BK": 0.20,
            "CPN.BK": 0.20,
            "BDMS.BK": 0.25,
            "BEM.BK": 0.15,
            "MINT.BK": 0.20,
        },
        "benchmark": "SET",
    },
    "TH-BlueChip": {
        "name": "Thai Blue Chip Fund",
        "style": "Large Cap",
        "description": "เน้นหุ้นขนาดใหญ่ สภาพคล่องสูง",
        "holdings": {
            "PTT.BK": 0.20,
            "ADVANC.BK": 0.20,
            "SCC.BK": 0.20,
            "CPALL.BK": 0.20,
            "KBANK.BK": 0.20,
        },
        "benchmark": "SET50",
    },
}

# ---------- ML Hyperparameters ----------

LSTM_PARAMS = {
    "sequence_length": 60,        # 60-day lookback
    "lstm_units_1": 128,
    "lstm_units_2": 64,
    "dropout": 0.2,
    "dense_units": 32,
    "learning_rate": 0.001,
    "epochs": 100,
    "batch_size": 32,
    "early_stopping_patience": 10,
    "prediction_days": 5,
}

XGBOOST_PARAMS = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.01,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "prediction_days": 5,
}

AUTOGLUON_PARAMS = {
    "time_limit": 120,            # 2 minutes per symbol
    "preset": "medium_quality",   # "best_quality", "medium_quality", "good_quality"
    "eval_metric": "root_mean_squared_error",
    "prediction_days": 5,
    "excluded_model_types": ["FASTAI"],  # FastAI has compatibility issues; 9 other models work fine
}

ENSEMBLE_PARAMS = {
    "lstm_weight": 0.3,
    "xgboost_weight": 0.35,
    "autogluon_weight": 0.35,
}

# ---------- Data Settings ----------

DATA_SETTINGS = {
    "default_period": "5y",       # 5 years of historical data
    "train_ratio": 0.70,
    "val_ratio": 0.15,
    "test_ratio": 0.15,
    "validation_gap_days": 0,     # embargo gap between train/val/test targets
    "walk_forward_window": 30,    # retrain every 30 days
    "walk_forward_folds": 5,      # number of walk-forward folds
}

# ---------- Backtesting ----------

BACKTEST_SETTINGS = {
    "initial_capital": 1_000_000,  # 1M THB
    "buy_threshold": 0.005,        # predict +0.5% → BUY
    "sell_threshold": -0.005,      # predict -0.5% → SELL
    "rsi_overbought": 70,
    "rsi_oversold": 30,
    "commission": 0.001,           # 0.1% per trade
}

# ---------- Portfolio ----------

PORTFOLIO_SETTINGS = {
    "min_weight": 0.05,           # 5% minimum per stock
    "max_weight": 0.40,           # 40% maximum per stock
    "risk_free_rate": 0.02,       # 2% risk-free rate (Thai gov bond)
    "num_random_portfolios": 5000,
}


# Singleton
settings = Settings()
