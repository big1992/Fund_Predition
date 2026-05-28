"""
Settings routes — read/update system configuration via UI.
"""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional

from config.settings import (
    settings, LSTM_PARAMS, XGBOOST_PARAMS, ENSEMBLE_PARAMS,
    DATA_SETTINGS, BACKTEST_SETTINGS, PORTFOLIO_SETTINGS,
)

router = APIRouter()


class AllSettings(BaseModel):
    """Response model for all settings."""
    # App
    app_name: str
    app_version: str
    debug: bool
    openai_api_key_set: bool  # don't expose the actual key

    # LSTM
    lstm_sequence_length: int
    lstm_units_1: int
    lstm_units_2: int
    lstm_dropout: float
    lstm_dense_units: int
    lstm_learning_rate: float
    lstm_epochs: int
    lstm_batch_size: int
    lstm_early_stopping: int
    lstm_prediction_days: int

    # XGBoost
    xgb_n_estimators: int
    xgb_max_depth: int
    xgb_learning_rate: float
    xgb_subsample: float
    xgb_colsample_bytree: float
    xgb_reg_alpha: float
    xgb_reg_lambda: float

    # Ensemble
    ensemble_lstm_weight: float
    ensemble_xgb_weight: float

    # Data
    data_period: str
    data_train_ratio: float
    data_val_ratio: float
    data_test_ratio: float

    # Backtest
    bt_initial_capital: int
    bt_buy_threshold: float
    bt_sell_threshold: float
    bt_commission: float

    # Portfolio
    pf_min_weight: float
    pf_max_weight: float
    pf_risk_free_rate: float
    pf_num_portfolios: int


class UpdateSettings(BaseModel):
    """Request model for updating settings."""
    # LSTM
    lstm_sequence_length: Optional[int] = None
    lstm_units_1: Optional[int] = None
    lstm_units_2: Optional[int] = None
    lstm_dropout: Optional[float] = None
    lstm_dense_units: Optional[int] = None
    lstm_learning_rate: Optional[float] = None
    lstm_epochs: Optional[int] = None
    lstm_batch_size: Optional[int] = None
    lstm_early_stopping: Optional[int] = None
    lstm_prediction_days: Optional[int] = None

    # XGBoost
    xgb_n_estimators: Optional[int] = None
    xgb_max_depth: Optional[int] = None
    xgb_learning_rate: Optional[float] = None
    xgb_subsample: Optional[float] = None
    xgb_colsample_bytree: Optional[float] = None
    xgb_reg_alpha: Optional[float] = None
    xgb_reg_lambda: Optional[float] = None

    # Ensemble
    ensemble_lstm_weight: Optional[float] = None
    ensemble_xgb_weight: Optional[float] = None

    # Data
    data_train_ratio: Optional[float] = None
    data_val_ratio: Optional[float] = None
    data_test_ratio: Optional[float] = None

    # Backtest
    bt_initial_capital: Optional[int] = None
    bt_buy_threshold: Optional[float] = None
    bt_sell_threshold: Optional[float] = None
    bt_commission: Optional[float] = None

    # Portfolio
    pf_min_weight: Optional[float] = None
    pf_max_weight: Optional[float] = None
    pf_risk_free_rate: Optional[float] = None
    pf_num_portfolios: Optional[int] = None

    # OpenAI
    openai_api_key: Optional[str] = None


def _build_response() -> AllSettings:
    return AllSettings(
        app_name=settings.app_name,
        app_version=settings.app_version,
        debug=settings.debug,
        openai_api_key_set=bool(settings.openai_api_key),
        # LSTM
        lstm_sequence_length=LSTM_PARAMS["sequence_length"],
        lstm_units_1=LSTM_PARAMS["lstm_units_1"],
        lstm_units_2=LSTM_PARAMS["lstm_units_2"],
        lstm_dropout=LSTM_PARAMS["dropout"],
        lstm_dense_units=LSTM_PARAMS["dense_units"],
        lstm_learning_rate=LSTM_PARAMS["learning_rate"],
        lstm_epochs=LSTM_PARAMS["epochs"],
        lstm_batch_size=LSTM_PARAMS["batch_size"],
        lstm_early_stopping=LSTM_PARAMS["early_stopping_patience"],
        lstm_prediction_days=LSTM_PARAMS["prediction_days"],
        # XGBoost
        xgb_n_estimators=XGBOOST_PARAMS["n_estimators"],
        xgb_max_depth=XGBOOST_PARAMS["max_depth"],
        xgb_learning_rate=XGBOOST_PARAMS["learning_rate"],
        xgb_subsample=XGBOOST_PARAMS["subsample"],
        xgb_colsample_bytree=XGBOOST_PARAMS["colsample_bytree"],
        xgb_reg_alpha=XGBOOST_PARAMS["reg_alpha"],
        xgb_reg_lambda=XGBOOST_PARAMS["reg_lambda"],
        # Ensemble
        ensemble_lstm_weight=ENSEMBLE_PARAMS["lstm_weight"],
        ensemble_xgb_weight=ENSEMBLE_PARAMS["xgboost_weight"],
        # Data
        data_period=DATA_SETTINGS["default_period"],
        data_train_ratio=DATA_SETTINGS["train_ratio"],
        data_val_ratio=DATA_SETTINGS["val_ratio"],
        data_test_ratio=DATA_SETTINGS["test_ratio"],
        # Backtest
        bt_initial_capital=BACKTEST_SETTINGS["initial_capital"],
        bt_buy_threshold=BACKTEST_SETTINGS["buy_threshold"],
        bt_sell_threshold=BACKTEST_SETTINGS["sell_threshold"],
        bt_commission=BACKTEST_SETTINGS["commission"],
        # Portfolio
        pf_min_weight=PORTFOLIO_SETTINGS["min_weight"],
        pf_max_weight=PORTFOLIO_SETTINGS["max_weight"],
        pf_risk_free_rate=PORTFOLIO_SETTINGS["risk_free_rate"],
        pf_num_portfolios=PORTFOLIO_SETTINGS["num_random_portfolios"],
    )


@router.get("", response_model=AllSettings)
async def get_settings():
    """Get all current settings."""
    return _build_response()


@router.put("", response_model=AllSettings)
async def update_settings(req: UpdateSettings):
    """Update settings (in-memory, resets on restart)."""
    import os
    from api.main import app_state

    updates = req.model_dump(exclude_none=True)

    # LSTM params
    lstm_map = {
        "lstm_sequence_length": "sequence_length",
        "lstm_units_1": "lstm_units_1", "lstm_units_2": "lstm_units_2",
        "lstm_dropout": "dropout", "lstm_dense_units": "dense_units",
        "lstm_learning_rate": "learning_rate",
        "lstm_epochs": "epochs", "lstm_batch_size": "batch_size",
        "lstm_early_stopping": "early_stopping_patience",
        "lstm_prediction_days": "prediction_days",
    }
    for key, param in lstm_map.items():
        if key in updates:
            LSTM_PARAMS[param] = updates[key]

    # XGBoost params
    xgb_map = {
        "xgb_n_estimators": "n_estimators", "xgb_max_depth": "max_depth",
        "xgb_learning_rate": "learning_rate", "xgb_subsample": "subsample",
        "xgb_colsample_bytree": "colsample_bytree",
        "xgb_reg_alpha": "reg_alpha", "xgb_reg_lambda": "reg_lambda",
    }
    for key, param in xgb_map.items():
        if key in updates:
            XGBOOST_PARAMS[param] = updates[key]

    # Ensemble
    if "ensemble_lstm_weight" in updates:
        ENSEMBLE_PARAMS["lstm_weight"] = updates["ensemble_lstm_weight"]
    if "ensemble_xgb_weight" in updates:
        ENSEMBLE_PARAMS["xgboost_weight"] = updates["ensemble_xgb_weight"]

    # Data
    data_map = {
        "data_train_ratio": "train_ratio",
        "data_val_ratio": "val_ratio",
        "data_test_ratio": "test_ratio",
    }
    for key, param in data_map.items():
        if key in updates:
            DATA_SETTINGS[param] = updates[key]

    # Backtest
    bt_map = {
        "bt_initial_capital": "initial_capital",
        "bt_buy_threshold": "buy_threshold",
        "bt_sell_threshold": "sell_threshold",
        "bt_commission": "commission",
    }
    for key, param in bt_map.items():
        if key in updates:
            BACKTEST_SETTINGS[param] = updates[key]

    # Portfolio
    pf_map = {
        "pf_min_weight": "min_weight",
        "pf_max_weight": "max_weight",
        "pf_risk_free_rate": "risk_free_rate",
        "pf_num_portfolios": "num_random_portfolios",
    }
    for key, param in pf_map.items():
        if key in updates:
            PORTFOLIO_SETTINGS[param] = updates[key]

    # OpenAI key
    if "openai_api_key" in updates and updates["openai_api_key"]:
        os.environ["OPENAI_API_KEY"] = updates["openai_api_key"]
        settings.openai_api_key = updates["openai_api_key"]
        # Re-init AI explainer
        from ai.explainer import AIExplainer
        app_state["ai_explainer"] = AIExplainer()

    return _build_response()
