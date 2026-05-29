"""
Shared training orchestration for sync/async API paths.
"""

from pathlib import Path
from typing import Callable, Optional

from config.settings import (
    AUTOGLUON_PARAMS,
    DATA_SETTINGS,
    ENSEMBLE_PARAMS,
    LSTM_PARAMS,
    STOCK_UNIVERSE,
    XGBOOST_PARAMS,
    settings,
)
from data.storage import DatabaseManager


def _strip_large_training_artifacts(metrics: dict) -> dict:
    stripped = {}
    for model_key, model_val in metrics.items():
        if isinstance(model_val, dict):
            out = {}
            for k, v in model_val.items():
                if k == "training" and isinstance(v, dict):
                    out[k] = {
                        tk: tv
                        for tk, tv in v.items()
                        if not tk.endswith("_history") and tk != "leaderboard"
                    }
                else:
                    out[k] = v
            stripped[model_key] = out
        else:
            stripped[model_key] = model_val
    return stripped


def _artifact_paths_for_symbol(symbol: str) -> dict:
    safe = symbol.replace(".", "_")
    model_dir = Path(settings.model_dir)
    return {
        "lstm": str(model_dir / f"lstm_{safe}.keras"),
        "xgboost": str(model_dir / f"xgb_{safe}.joblib"),
        "autogluon": str(model_dir / f"ag_{safe}"),
        "ensemble": str(model_dir / f"ensemble_weights_{safe}.joblib"),
        "naive_last_price": None,
        "mean_return": None,
        "buy_and_hold": None,
        "benchmark_hold": None,
    }


def _current_params() -> dict:
    return {
        "lstm_sequence_length": LSTM_PARAMS.get("sequence_length"),
        "lstm_units_1": LSTM_PARAMS.get("lstm_units_1"),
        "lstm_units_2": LSTM_PARAMS.get("lstm_units_2"),
        "lstm_dropout": LSTM_PARAMS.get("dropout"),
        "lstm_dense_units": LSTM_PARAMS.get("dense_units"),
        "lstm_learning_rate": LSTM_PARAMS.get("learning_rate"),
        "lstm_epochs": LSTM_PARAMS.get("epochs"),
        "lstm_batch_size": LSTM_PARAMS.get("batch_size"),
        "lstm_early_stopping": LSTM_PARAMS.get("early_stopping_patience"),
        "xgb_n_estimators": XGBOOST_PARAMS.get("n_estimators"),
        "xgb_max_depth": XGBOOST_PARAMS.get("max_depth"),
        "xgb_learning_rate": XGBOOST_PARAMS.get("learning_rate"),
        "xgb_subsample": XGBOOST_PARAMS.get("subsample"),
        "xgb_colsample_bytree": XGBOOST_PARAMS.get("colsample_bytree"),
        "xgb_reg_alpha": XGBOOST_PARAMS.get("reg_alpha"),
        "xgb_reg_lambda": XGBOOST_PARAMS.get("reg_lambda"),
        "ag_time_limit": AUTOGLUON_PARAMS.get("time_limit"),
        "ag_preset": AUTOGLUON_PARAMS.get("preset"),
        "ensemble_lstm_weight": ENSEMBLE_PARAMS.get("lstm_weight"),
        "ensemble_xgb_weight": ENSEMBLE_PARAMS.get("xgboost_weight"),
        "ensemble_ag_weight": ENSEMBLE_PARAMS.get("autogluon_weight"),
        "data_train_ratio": DATA_SETTINGS.get("train_ratio"),
        "data_val_ratio": DATA_SETTINGS.get("val_ratio"),
        "data_test_ratio": DATA_SETTINGS.get("test_ratio"),
    }


def _train_single_symbol(trainer, df, symbol: str, model_type: str, walk_forward: bool) -> tuple[dict, list[str]]:
    if walk_forward and model_type == "all":
        metrics = trainer.train_walk_forward(df, symbol)
        models = [f"lstm_{symbol}", f"xgb_{symbol}", f"ag_{symbol}"]
    elif model_type == "all":
        metrics = trainer.train_all(df, symbol)
        models = [f"lstm_{symbol}", f"xgb_{symbol}", f"ag_{symbol}"]
    elif model_type == "lstm":
        metrics = trainer.evaluate_baselines(df, symbol)
        metrics["lstm"] = trainer.train_lstm(df, symbol)
        models = [f"lstm_{symbol}"]
    elif model_type == "xgboost":
        metrics = trainer.evaluate_baselines(df, symbol)
        metrics["xgboost"] = trainer.train_xgboost(df, symbol)
        models = [f"xgb_{symbol}"]
    elif model_type == "autogluon":
        metrics = trainer.evaluate_baselines(df, symbol)
        metrics["autogluon"] = trainer.train_autogluon(df, symbol)
        models = [f"ag_{symbol}"]
    else:
        metrics = {}
        models = []
    return metrics, models


def refresh_loaded_models(app_state: dict) -> None:
    model_dir = Path(settings.model_dir)
    loaded = []
    if model_dir.exists():
        for f in model_dir.iterdir():
            if f.suffix in (".keras", ".joblib"):
                loaded.append(f.stem)
            elif f.is_dir() and f.name.startswith("ag_"):
                loaded.append(f.name)
    app_state["models_loaded"] = loaded


def run_training_for_symbols(
    *,
    trainer,
    db,
    model_type: str,
    walk_forward: bool,
    symbols: Optional[list[str]] = None,
    on_symbol_start: Optional[Callable[[str, int, int], None]] = None,
    cancel_requested: Optional[Callable[[], bool]] = None,
) -> dict:
    symbol_list = symbols if symbols else list(STOCK_UNIVERSE.keys())
    all_metrics = {}
    models_trained = []

    for idx, symbol in enumerate(symbol_list):
        if cancel_requested and cancel_requested():
            break
        if on_symbol_start:
            on_symbol_start(symbol, idx, len(symbol_list))

        df = db.get_stock_prices(symbol)
        if df.empty:
            continue

        try:
            metrics, trained = _train_single_symbol(trainer, df, symbol, model_type, walk_forward)
            all_metrics[symbol] = metrics
            models_trained.extend(trained)
            try:
                db2 = DatabaseManager(settings.db_path)
                stripped_metrics = _strip_large_training_artifacts(metrics)
                params_now = _current_params()
                run_id = None
                if len(df.index) > 0:
                    train_start = str(df.index.min())
                    train_end = str(df.index.max())
                else:
                    train_start = None
                    train_end = None
                run_id = db2.save_training_run(
                    symbol=symbol,
                    model_type=model_type,
                    metrics=stripped_metrics,
                    params=params_now,
                )
                db2.save_model_registry_entries(
                    symbol=symbol,
                    metrics_by_model=stripped_metrics,
                    params=params_now,
                    training_run_id=run_id,
                    train_start_date=train_start,
                    train_end_date=train_end,
                    artifact_paths=_artifact_paths_for_symbol(symbol),
                    feature_version="techind_v1",
                )
            except Exception:
                pass
        except Exception as e:
            all_metrics[symbol] = {"error": str(e)}

    return {
        "status": "success",
        "models_trained": models_trained,
        "metrics": all_metrics,
        "message": f"Trained {len(models_trained)} models for {len(all_metrics)} symbols",
    }
