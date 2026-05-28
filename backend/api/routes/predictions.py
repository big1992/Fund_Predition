"""
Prediction routes: get predictions, train models, model performance.
"""

from fastapi import APIRouter, Query, HTTPException
from typing import Optional

from config.settings import STOCK_UNIVERSE
from schemas.api_schemas import (
    PredictionResponse, PredictionPoint,
    TrainRequest, TrainResponse,
    ModelPerformanceResponse, ModelMetrics,
    FeatureImportanceResponse, FeatureImportance,
)
from api.main import app_state

router = APIRouter()


@router.get("/{symbol}", response_model=PredictionResponse)
async def get_prediction(
    symbol: str,
    model: str = Query("ensemble", pattern="^(lstm|xgboost|autogluon|ensemble)$", description="Model: lstm, xgboost, autogluon, ensemble"),
    days: int = Query(5, ge=1, le=30, description="Prediction days ahead"),
):
    """Get price predictions for a stock."""
    db = app_state["db"]
    trainer = app_state["trainer"]
    if not db or not trainer:
        raise HTTPException(500, "Service not initialized")

    # Get historical data
    df = db.get_stock_prices(symbol)
    if df.empty:
        raise HTTPException(404, f"No data for {symbol}. Run /api/stocks/collect first.")

    # Generate predictions
    result = trainer.predict(df, symbol, model_type=model, days=days)

    predictions = [
        PredictionPoint(
            date=p["date"],
            predicted_price=p["predicted_price"],
            confidence=p["confidence"],
        )
        for p in result["predictions"]
    ]

    return PredictionResponse(
        symbol=symbol,
        model=model,
        current_price=result["current_price"],
        predictions=predictions,
        signal=result["signal"],
        signal_reason=result["signal_reason"],
        confidence=result["confidence"],
    )


@router.post("/train", response_model=TrainResponse)
async def train_models(request: TrainRequest):
    """Train ML models."""
    db = app_state["db"]
    trainer = app_state["trainer"]
    if not db or not trainer:
        raise HTTPException(500, "Service not initialized")

    symbols = request.symbols if request.symbols else list(STOCK_UNIVERSE.keys())
    all_metrics = {}
    models_trained = []

    for symbol in symbols:
        df = db.get_stock_prices(symbol)
        if df.empty:
            continue

        try:
            if request.walk_forward and request.model_type == "all":
                # Walk-forward validation (expanding window)
                metrics = trainer.train_walk_forward(df, symbol)
                models_trained.extend([f"lstm_{symbol}", f"xgb_{symbol}", f"ag_{symbol}"])
            elif request.model_type == "all":
                metrics = trainer.train_all(df, symbol)
                models_trained.extend([f"lstm_{symbol}", f"xgb_{symbol}", f"ag_{symbol}"])
            elif request.model_type == "lstm":
                metrics = {"lstm": trainer.train_lstm(df, symbol)}
                models_trained.append(f"lstm_{symbol}")
            elif request.model_type == "xgboost":
                metrics = {"xgboost": trainer.train_xgboost(df, symbol)}
                models_trained.append(f"xgb_{symbol}")
            elif request.model_type == "autogluon":
                metrics = {"autogluon": trainer.train_autogluon(df, symbol)}
                models_trained.append(f"ag_{symbol}")
            else:
                metrics = {}

            all_metrics[symbol] = metrics

            # Save training history to database
            from config.settings import (
                settings, LSTM_PARAMS, XGBOOST_PARAMS, AUTOGLUON_PARAMS,
                ENSEMBLE_PARAMS, DATA_SETTINGS
            )
            from data.storage import DatabaseManager
            
            # Record current params from actual dicts
            current_params = {
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
            
            # Strip large arrays from metrics before saving
            def _strip_arrays(m):
                stripped = {}
                for model_key, model_val in m.items():
                    if isinstance(model_val, dict):
                        sv = {}
                        for k, v in model_val.items():
                            if k == "training" and isinstance(v, dict):
                                sv[k] = {tk: tv for tk, tv in v.items()
                                         if not tk.endswith("_history") and tk != "leaderboard"}
                            else:
                                sv[k] = v
                        stripped[model_key] = sv
                    else:
                        stripped[model_key] = model_val
                return stripped
            
            try:
                db = DatabaseManager(settings.db_path)
                db.save_training_run(
                    symbol=symbol,
                    model_type=request.model_type,
                    metrics=_strip_arrays(metrics),
                    params=current_params,
                )
            except Exception:
                pass

        except Exception as e:
            all_metrics[symbol] = {"error": str(e)}

    # Update loaded models
    from pathlib import Path
    from config.settings import settings
    model_dir = Path(settings.model_dir)
    if model_dir.exists():
        loaded = []
        for f in model_dir.iterdir():
            if f.suffix in (".keras", ".joblib"):
                loaded.append(f.stem)
            elif f.is_dir() and f.name.startswith("ag_"):
                loaded.append(f.name)
        app_state["models_loaded"] = loaded

    return TrainResponse(
        status="success",
        models_trained=models_trained,
        metrics=all_metrics,
        message=f"Trained {len(models_trained)} models for {len(all_metrics)} symbols",
    )


@router.get("/models/performance", response_model=ModelPerformanceResponse)
async def get_model_performance():
    """Get performance metrics for all trained models.
    Falls back to database training_history if in-memory metrics are empty (e.g., after server restart).
    """
    trainer = app_state["trainer"]
    if not trainer:
        raise HTTPException(500, "Trainer not initialized")

    models = []

    # Try in-memory metrics first (available during current session)
    metrics_data = trainer.get_all_metrics()

    if metrics_data:
        for key, metrics in metrics_data.items():
            if "error" in metrics:
                continue
            parts = key.split("_", 1)
            model_name = parts[0] if len(parts) > 0 else "unknown"
            symbol = parts[1] if len(parts) > 1 else "unknown"

            models.append(ModelMetrics(
                model_name=model_name,
                symbol=symbol,
                rmse=metrics.get("rmse", 0),
                mae=metrics.get("mae", 0),
                mape=metrics.get("mape", 0),
                directional_accuracy=metrics.get("directional_accuracy", 0),
                r_squared=metrics.get("r_squared", 0),
            ))
    else:
        # Fallback: load latest metrics from database training_history
        from config.settings import settings, STOCK_UNIVERSE
        from data.storage import DatabaseManager
        import json

        try:
            db = DatabaseManager(settings.db_path)
            for symbol in STOCK_UNIVERSE.keys():
                history = db.get_training_history(symbol, limit=1)
                if not history:
                    continue

                latest = history[0]
                metrics_detail = latest.get("metrics_detail", {})

                # LSTM metrics
                lstm = metrics_detail.get("lstm", {})
                if lstm and "error" not in lstm and lstm.get("mape") is not None:
                    models.append(ModelMetrics(
                        model_name="lstm",
                        symbol=symbol,
                        rmse=lstm.get("rmse", 0),
                        mae=lstm.get("mae", 0),
                        mape=lstm.get("mape", 0),
                        directional_accuracy=lstm.get("directional_accuracy", 0),
                        r_squared=lstm.get("r_squared", 0),
                    ))

                # XGBoost metrics
                xgb = metrics_detail.get("xgboost", {})
                if xgb and "error" not in xgb and xgb.get("mape") is not None:
                    models.append(ModelMetrics(
                        model_name="xgboost",
                        symbol=symbol,
                        rmse=xgb.get("rmse", 0),
                        mae=xgb.get("mae", 0),
                        mape=xgb.get("mape", 0),
                        directional_accuracy=xgb.get("directional_accuracy", 0),
                        r_squared=xgb.get("r_squared", 0),
                    ))

                # AutoGluon metrics
                ag = metrics_detail.get("autogluon", {})
                if ag and "error" not in ag and ag.get("mape") is not None:
                    models.append(ModelMetrics(
                        model_name="autogluon",
                        symbol=symbol,
                        rmse=ag.get("rmse", 0),
                        mae=ag.get("mae", 0),
                        mape=ag.get("mape", 0),
                        directional_accuracy=ag.get("directional_accuracy", 0),
                        r_squared=ag.get("r_squared", 0),
                    ))
        except Exception as e:
            logger.warning("Could not load metrics from DB: %s", e)

    return ModelPerformanceResponse(models=models)


@router.get("/models/features/{symbol}", response_model=FeatureImportanceResponse)
async def get_feature_importance(symbol: str):
    """Get feature importance from XGBoost model."""
    trainer = app_state["trainer"]
    if not trainer:
        raise HTTPException(500, "Trainer not initialized")

    # Load XGBoost model
    try:
        trainer._load_xgboost(symbol)
    except Exception:
        raise HTTPException(404, f"XGBoost model not found for {symbol}")

    if not trainer.xgb_model or not trainer.xgb_model.feature_importances:
        raise HTTPException(404, f"No feature importance data for {symbol}")

    importances = trainer.xgb_model.get_feature_importance(top_n=15)

    return FeatureImportanceResponse(
        symbol=symbol,
        model="xgboost",
        features=[FeatureImportance(**f) for f in importances],
    )


@router.get("/models/autogluon-leaderboard/{symbol}")
async def get_autogluon_leaderboard(symbol: str):
    """Get AutoGluon leaderboard (all models tried during training)."""
    trainer = app_state["trainer"]
    if not trainer:
        raise HTTPException(500, "Trainer not initialized")

    try:
        trainer._load_autogluon(symbol)
    except Exception:
        raise HTTPException(404, f"AutoGluon model not found for {symbol}")

    if not trainer.ag_model or not trainer.ag_model.predictor:
        raise HTTPException(404, f"No AutoGluon model for {symbol}")

    leaderboard = trainer.ag_model.get_leaderboard()
    feature_importance = trainer.ag_model.get_feature_importance(top_n=15)

    return {
        "symbol": symbol,
        "model": "autogluon",
        "best_model": trainer.ag_model.predictor.model_best if trainer.ag_model.predictor else None,
        "leaderboard": leaderboard,
        "feature_importance": feature_importance,
    }


@router.get("/models/history/{symbol}")
async def get_training_history(symbol: str, limit: int = 10):
    """Get training history for a symbol from database."""
    from config.settings import settings
    from data.storage import DatabaseManager
    db = DatabaseManager(settings.db_path)
    history = db.get_training_history(symbol, limit=limit)
    return {"symbol": symbol, "history": history}


@router.get("/models/best-run/{symbol}")
async def get_best_training_run(symbol: str):
    """Get the best training run for a symbol (lowest composite score)."""
    from config.settings import settings
    from data.storage import DatabaseManager
    db = DatabaseManager(settings.db_path)
    best = db.get_best_training_run(symbol)
    if not best:
        return {"symbol": symbol, "best_run": None, "message": "No training history found"}
    return {
        "symbol": symbol,
        "best_run": best,
        "message": f"Best run (id={best['id']}): composite_score = {best.get('composite_score', 0):.2f}",
    }


# ======================== BACKGROUND TRAINING ========================

@router.post("/train/async")
async def train_async(request: TrainRequest):
    """Start training in background. Returns task_id for status polling."""
    import threading
    from api.task_manager import task_manager

    db = app_state["db"]
    trainer = app_state["trainer"]
    if not db or not trainer:
        raise HTTPException(500, "Service not initialized")

    symbols = request.symbols if request.symbols else list(STOCK_UNIVERSE.keys())
    symbol = symbols[0] if symbols else "unknown"
    task_id = task_manager.create_task(symbol, request.model_type)

    def _run_training():
        try:
            task_manager.update_task(task_id, status="running", progress=10,
                                    message=f"🔄 กำลังเตรียมข้อมูล {symbol}...")
            all_metrics = {}
            models_trained = []

            for i, sym in enumerate(symbols):
                pct = 10 + int((i / max(len(symbols), 1)) * 70)
                task_manager.update_task(task_id, progress=pct,
                                        message=f"🧠 Training {sym} ({i+1}/{len(symbols)})...")

                df = db.get_stock_prices(sym)
                if df.empty:
                    continue

                try:
                    if request.walk_forward and request.model_type == "all":
                        metrics = trainer.train_walk_forward(df, sym)
                        models_trained.extend([f"lstm_{sym}", f"xgb_{sym}", f"ag_{sym}"])
                    elif request.model_type == "all":
                        metrics = trainer.train_all(df, sym)
                        models_trained.extend([f"lstm_{sym}", f"xgb_{sym}", f"ag_{sym}"])
                    elif request.model_type == "lstm":
                        metrics = {"lstm": trainer.train_lstm(df, sym)}
                        models_trained.append(f"lstm_{sym}")
                    elif request.model_type == "xgboost":
                        metrics = {"xgboost": trainer.train_xgboost(df, sym)}
                        models_trained.append(f"xgb_{sym}")
                    elif request.model_type == "autogluon":
                        metrics = {"autogluon": trainer.train_autogluon(df, sym)}
                        models_trained.append(f"ag_{sym}")
                    else:
                        metrics = {}

                    all_metrics[sym] = metrics

                    # Save to DB
                    from config.settings import (
                        settings as cfg, LSTM_PARAMS, XGBOOST_PARAMS,
                        AUTOGLUON_PARAMS, ENSEMBLE_PARAMS, DATA_SETTINGS,
                    )
                    from data.storage import DatabaseManager

                    current_params = {
                        "lstm_sequence_length": LSTM_PARAMS.get("sequence_length"),
                        "lstm_units_1": LSTM_PARAMS.get("lstm_units_1"),
                        "lstm_units_2": LSTM_PARAMS.get("lstm_units_2"),
                        "lstm_dropout": LSTM_PARAMS.get("dropout"),
                        "lstm_learning_rate": LSTM_PARAMS.get("learning_rate"),
                        "lstm_epochs": LSTM_PARAMS.get("epochs"),
                        "lstm_batch_size": LSTM_PARAMS.get("batch_size"),
                        "xgb_n_estimators": XGBOOST_PARAMS.get("n_estimators"),
                        "xgb_max_depth": XGBOOST_PARAMS.get("max_depth"),
                        "xgb_learning_rate": XGBOOST_PARAMS.get("learning_rate"),
                        "ag_time_limit": AUTOGLUON_PARAMS.get("time_limit"),
                        "ag_preset": AUTOGLUON_PARAMS.get("preset"),
                        "ensemble_lstm_weight": ENSEMBLE_PARAMS.get("lstm_weight"),
                        "ensemble_xgb_weight": ENSEMBLE_PARAMS.get("xgboost_weight"),
                        "ensemble_ag_weight": ENSEMBLE_PARAMS.get("autogluon_weight"),
                    }

                    def _strip(m):
                        out = {}
                        for mk, mv in m.items():
                            if isinstance(mv, dict):
                                out[mk] = {k: v for k, v in mv.items()
                                           if not (k == "training" and isinstance(v, dict)
                                                   and any(tk.endswith("_history") or tk == "leaderboard" for tk in v))}
                            else:
                                out[mk] = mv
                        return out

                    try:
                        db2 = DatabaseManager(cfg.db_path)
                        db2.save_training_run(symbol=sym, model_type=request.model_type,
                                             metrics=_strip(metrics), params=current_params)
                    except Exception:
                        pass

                except Exception as e:
                    all_metrics[sym] = {"error": str(e)}

            # Update loaded models
            from pathlib import Path
            from config.settings import settings as cfg
            model_dir = Path(cfg.model_dir)
            if model_dir.exists():
                loaded = []
                for f in model_dir.iterdir():
                    if f.suffix in (".keras", ".joblib"):
                        loaded.append(f.stem)
                    elif f.is_dir() and f.name.startswith("ag_"):
                        loaded.append(f.name)
                app_state["models_loaded"] = loaded

            task_manager.complete_task(task_id, {
                "status": "success",
                "models_trained": models_trained,
                "metrics": all_metrics,
                "message": f"Trained {len(models_trained)} models for {len(all_metrics)} symbols",
            })

        except Exception as e:
            task_manager.fail_task(task_id, str(e))

    thread = threading.Thread(target=_run_training, daemon=True)
    thread.start()

    return {"task_id": task_id, "status": "started", "symbol": symbol}


@router.get("/train/status/{task_id}")
async def get_train_status(task_id: str):
    """Get training task status."""
    from api.task_manager import task_manager
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(404, f"Task {task_id} not found")
    return task


@router.get("/train/active")
async def get_active_tasks():
    """Get all active training tasks."""
    from api.task_manager import task_manager
    return {"tasks": task_manager.get_active_tasks(), "all": task_manager.get_all_tasks()}


@router.delete("/train/dismiss/{task_id}")
async def dismiss_task(task_id: str):
    """Dismiss a completed/failed task."""
    from api.task_manager import task_manager
    task_manager.dismiss_task(task_id)
    return {"status": "dismissed", "task_id": task_id}
