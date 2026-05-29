"""
Settings routes - read/update system configuration via UI.
"""

import json
import os
from pathlib import Path
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

from config.settings import (
    BACKTEST_SETTINGS,
    DATA_SETTINGS,
    ENSEMBLE_PARAMS,
    LSTM_PARAMS,
    PORTFOLIO_SETTINGS,
    XGBOOST_PARAMS,
    settings,
)

router = APIRouter()
OVERRIDES_PATH = Path(settings.db_path).parent / "settings_overrides.json"


class AllSettings(BaseModel):
    app_name: str
    app_version: str
    debug: bool
    openai_api_key_set: bool

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

    xgb_n_estimators: int
    xgb_max_depth: int
    xgb_learning_rate: float
    xgb_subsample: float
    xgb_colsample_bytree: float
    xgb_reg_alpha: float
    xgb_reg_lambda: float

    ensemble_lstm_weight: float
    ensemble_xgb_weight: float

    data_period: str
    data_train_ratio: float
    data_val_ratio: float
    data_test_ratio: float
    data_validation_gap_days: int

    bt_initial_capital: int
    bt_buy_threshold: float
    bt_sell_threshold: float
    bt_commission: float

    pf_min_weight: float
    pf_max_weight: float
    pf_risk_free_rate: float
    pf_num_portfolios: int


class UpdateSettings(BaseModel):
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

    xgb_n_estimators: Optional[int] = None
    xgb_max_depth: Optional[int] = None
    xgb_learning_rate: Optional[float] = None
    xgb_subsample: Optional[float] = None
    xgb_colsample_bytree: Optional[float] = None
    xgb_reg_alpha: Optional[float] = None
    xgb_reg_lambda: Optional[float] = None

    ensemble_lstm_weight: Optional[float] = None
    ensemble_xgb_weight: Optional[float] = None

    data_train_ratio: Optional[float] = None
    data_val_ratio: Optional[float] = None
    data_test_ratio: Optional[float] = None
    data_validation_gap_days: Optional[int] = None

    bt_initial_capital: Optional[int] = None
    bt_buy_threshold: Optional[float] = None
    bt_sell_threshold: Optional[float] = None
    bt_commission: Optional[float] = None

    pf_min_weight: Optional[float] = None
    pf_max_weight: Optional[float] = None
    pf_risk_free_rate: Optional[float] = None
    pf_num_portfolios: Optional[int] = None

    openai_api_key: Optional[str] = None


def _build_response() -> AllSettings:
    return AllSettings(
        app_name=settings.app_name,
        app_version=settings.app_version,
        debug=settings.debug,
        openai_api_key_set=bool(settings.openai_api_key),
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
        xgb_n_estimators=XGBOOST_PARAMS["n_estimators"],
        xgb_max_depth=XGBOOST_PARAMS["max_depth"],
        xgb_learning_rate=XGBOOST_PARAMS["learning_rate"],
        xgb_subsample=XGBOOST_PARAMS["subsample"],
        xgb_colsample_bytree=XGBOOST_PARAMS["colsample_bytree"],
        xgb_reg_alpha=XGBOOST_PARAMS["reg_alpha"],
        xgb_reg_lambda=XGBOOST_PARAMS["reg_lambda"],
        ensemble_lstm_weight=ENSEMBLE_PARAMS["lstm_weight"],
        ensemble_xgb_weight=ENSEMBLE_PARAMS["xgboost_weight"],
        data_period=DATA_SETTINGS["default_period"],
        data_train_ratio=DATA_SETTINGS["train_ratio"],
        data_val_ratio=DATA_SETTINGS["val_ratio"],
        data_test_ratio=DATA_SETTINGS["test_ratio"],
        data_validation_gap_days=DATA_SETTINGS.get("validation_gap_days", 0),
        bt_initial_capital=BACKTEST_SETTINGS["initial_capital"],
        bt_buy_threshold=BACKTEST_SETTINGS["buy_threshold"],
        bt_sell_threshold=BACKTEST_SETTINGS["sell_threshold"],
        bt_commission=BACKTEST_SETTINGS["commission"],
        pf_min_weight=PORTFOLIO_SETTINGS["min_weight"],
        pf_max_weight=PORTFOLIO_SETTINGS["max_weight"],
        pf_risk_free_rate=PORTFOLIO_SETTINGS["risk_free_rate"],
        pf_num_portfolios=PORTFOLIO_SETTINGS["num_random_portfolios"],
    )


def _collect_overrides() -> dict:
    return {
        "lstm": dict(LSTM_PARAMS),
        "xgboost": dict(XGBOOST_PARAMS),
        "ensemble": dict(ENSEMBLE_PARAMS),
        "data": dict(DATA_SETTINGS),
        "backtest": dict(BACKTEST_SETTINGS),
        "portfolio": dict(PORTFOLIO_SETTINGS),
        "openai_api_key": settings.openai_api_key or "",
    }


def _save_overrides() -> None:
    OVERRIDES_PATH.parent.mkdir(parents=True, exist_ok=True)
    OVERRIDES_PATH.write_text(
        json.dumps(_collect_overrides(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _load_overrides() -> None:
    if not OVERRIDES_PATH.exists():
        return
    try:
        payload = json.loads(OVERRIDES_PATH.read_text(encoding="utf-8"))
    except Exception:
        return

    if isinstance(payload.get("lstm"), dict):
        LSTM_PARAMS.update(payload["lstm"])
    if isinstance(payload.get("xgboost"), dict):
        XGBOOST_PARAMS.update(payload["xgboost"])
    if isinstance(payload.get("ensemble"), dict):
        ENSEMBLE_PARAMS.update(payload["ensemble"])
    if isinstance(payload.get("data"), dict):
        DATA_SETTINGS.update(payload["data"])
    if isinstance(payload.get("backtest"), dict):
        BACKTEST_SETTINGS.update(payload["backtest"])
    if isinstance(payload.get("portfolio"), dict):
        PORTFOLIO_SETTINGS.update(payload["portfolio"])

    api_key = payload.get("openai_api_key")
    if isinstance(api_key, str) and api_key.strip():
        settings.openai_api_key = api_key.strip()
        os.environ["OPENAI_API_KEY"] = settings.openai_api_key


_load_overrides()


@router.get("", response_model=AllSettings)
async def get_settings():
    return _build_response()


@router.put("", response_model=AllSettings)
async def update_settings(req: UpdateSettings):
    from api.main import app_state

    updates = req.model_dump(exclude_none=True)

    lstm_map = {
        "lstm_sequence_length": "sequence_length",
        "lstm_units_1": "lstm_units_1",
        "lstm_units_2": "lstm_units_2",
        "lstm_dropout": "dropout",
        "lstm_dense_units": "dense_units",
        "lstm_learning_rate": "learning_rate",
        "lstm_epochs": "epochs",
        "lstm_batch_size": "batch_size",
        "lstm_early_stopping": "early_stopping_patience",
        "lstm_prediction_days": "prediction_days",
    }
    for key, param in lstm_map.items():
        if key in updates:
            LSTM_PARAMS[param] = updates[key]

    xgb_map = {
        "xgb_n_estimators": "n_estimators",
        "xgb_max_depth": "max_depth",
        "xgb_learning_rate": "learning_rate",
        "xgb_subsample": "subsample",
        "xgb_colsample_bytree": "colsample_bytree",
        "xgb_reg_alpha": "reg_alpha",
        "xgb_reg_lambda": "reg_lambda",
    }
    for key, param in xgb_map.items():
        if key in updates:
            XGBOOST_PARAMS[param] = updates[key]

    if "ensemble_lstm_weight" in updates:
        ENSEMBLE_PARAMS["lstm_weight"] = updates["ensemble_lstm_weight"]
    if "ensemble_xgb_weight" in updates:
        ENSEMBLE_PARAMS["xgboost_weight"] = updates["ensemble_xgb_weight"]

    data_map = {
        "data_train_ratio": "train_ratio",
        "data_val_ratio": "val_ratio",
        "data_test_ratio": "test_ratio",
        "data_validation_gap_days": "validation_gap_days",
    }
    for key, param in data_map.items():
        if key in updates:
            DATA_SETTINGS[param] = updates[key]

    bt_map = {
        "bt_initial_capital": "initial_capital",
        "bt_buy_threshold": "buy_threshold",
        "bt_sell_threshold": "sell_threshold",
        "bt_commission": "commission",
    }
    for key, param in bt_map.items():
        if key in updates:
            BACKTEST_SETTINGS[param] = updates[key]

    pf_map = {
        "pf_min_weight": "min_weight",
        "pf_max_weight": "max_weight",
        "pf_risk_free_rate": "risk_free_rate",
        "pf_num_portfolios": "num_random_portfolios",
    }
    for key, param in pf_map.items():
        if key in updates:
            PORTFOLIO_SETTINGS[param] = updates[key]

    if "openai_api_key" in updates and updates["openai_api_key"]:
        settings.openai_api_key = updates["openai_api_key"]
        os.environ["OPENAI_API_KEY"] = updates["openai_api_key"]
        from ai.explainer import AIExplainer
        app_state["ai_explainer"] = AIExplainer()

    _save_overrides()
    return _build_response()
