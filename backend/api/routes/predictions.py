"""
Prediction routes: get predictions, train models, model performance.
"""

import logging
import threading
from fastapi import APIRouter, Query, HTTPException

from config.settings import STOCK_UNIVERSE, settings
from schemas.api_schemas import (
    FeatureImportance,
    FeatureImportanceResponse,
    ModelMetrics,
    ModelPerformanceResponse,
    PredictionPoint,
    PredictionResponse,
    TrainRequest,
    TrainResponse,
)
from api.main import app_state
from api.task_manager import task_manager
from data.storage import DatabaseManager
from services.training_service import refresh_loaded_models, run_training_for_symbols

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/{symbol}", response_model=PredictionResponse)
async def get_prediction(
    symbol: str,
    model: str = Query(
        "ensemble",
        pattern="^(lstm|xgboost|autogluon|ensemble)$",
        description="Model: lstm, xgboost, autogluon, ensemble",
    ),
    days: int = Query(5, ge=1, le=30, description="Prediction days ahead"),
):
    db = app_state["db"]
    trainer = app_state["trainer"]
    if not db or not trainer:
        raise HTTPException(500, "Service not initialized")

    df = db.get_stock_prices(symbol)
    if df.empty:
        raise HTTPException(404, f"No data for {symbol}. Run /api/stocks/collect first.")

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
    db = app_state["db"]
    trainer = app_state["trainer"]
    if not db or not trainer:
        raise HTTPException(500, "Service not initialized")

    symbols = request.symbols if request.symbols else list(STOCK_UNIVERSE.keys())
    result = run_training_for_symbols(
        trainer=trainer,
        db=db,
        model_type=request.model_type,
        walk_forward=request.walk_forward,
        symbols=symbols,
    )
    refresh_loaded_models(app_state)
    return TrainResponse(
        status=result["status"],
        models_trained=result["models_trained"],
        metrics=result["metrics"],
        message=result["message"],
    )


@router.get("/models/performance", response_model=ModelPerformanceResponse)
async def get_model_performance():
    trainer = app_state["trainer"]
    if not trainer:
        raise HTTPException(500, "Trainer not initialized")

    models = []
    metrics_data = trainer.get_all_metrics()
    if metrics_data:
        for key, metrics in metrics_data.items():
            if "error" in metrics:
                continue
            parts = key.split("_", 1)
            model_name = parts[0] if len(parts) > 0 else "unknown"
            symbol = parts[1] if len(parts) > 1 else "unknown"
            models.append(
                ModelMetrics(
                    model_name=model_name,
                    symbol=symbol,
                    rmse=metrics.get("rmse", 0),
                    mae=metrics.get("mae", 0),
                    mape=metrics.get("mape", 0),
                    directional_accuracy=metrics.get("directional_accuracy", 0),
                    r_squared=metrics.get("r_squared", 0),
                )
            )
    else:
        try:
            db = DatabaseManager(settings.db_path)
            for symbol in STOCK_UNIVERSE.keys():
                history = db.get_training_history(symbol, limit=1)
                if not history:
                    continue
                latest = history[0]
                metrics_detail = latest.get("metrics_detail", {})

                for model_name, row_key in (("lstm", "lstm"), ("xgboost", "xgboost"), ("autogluon", "autogluon")):
                    m = metrics_detail.get(row_key, {})
                    if m and "error" not in m and m.get("mape") is not None:
                        models.append(
                            ModelMetrics(
                                model_name=model_name,
                                symbol=symbol,
                                rmse=m.get("rmse", 0),
                                mae=m.get("mae", 0),
                                mape=m.get("mape", 0),
                                directional_accuracy=m.get("directional_accuracy", 0),
                                r_squared=m.get("r_squared", 0),
                            )
                        )
        except Exception as e:
            logger.warning("Could not load metrics from DB: %s", e)

    return ModelPerformanceResponse(models=models)


@router.get("/models/features/{symbol}", response_model=FeatureImportanceResponse)
async def get_feature_importance(symbol: str):
    trainer = app_state["trainer"]
    if not trainer:
        raise HTTPException(500, "Trainer not initialized")
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
    trainer = app_state["trainer"]
    if not trainer:
        raise HTTPException(500, "Trainer not initialized")
    try:
        trainer._load_autogluon(symbol)
    except Exception:
        raise HTTPException(404, f"AutoGluon model not found for {symbol}")
    if not trainer.ag_model or not trainer.ag_model.predictor:
        raise HTTPException(404, f"No AutoGluon model for {symbol}")

    return {
        "symbol": symbol,
        "model": "autogluon",
        "best_model": trainer.ag_model.predictor.model_best if trainer.ag_model.predictor else None,
        "leaderboard": trainer.ag_model.get_leaderboard(),
        "feature_importance": trainer.ag_model.get_feature_importance(top_n=15),
    }


@router.get("/models/history/{symbol}")
async def get_training_history(symbol: str, limit: int = 10):
    db = DatabaseManager(settings.db_path)
    history = db.get_training_history(symbol, limit=limit)
    return {"symbol": symbol, "history": history}


@router.get("/models/best-run/{symbol}")
async def get_best_training_run(symbol: str):
    db = DatabaseManager(settings.db_path)
    best = db.get_best_training_run(symbol)
    if not best:
        return {"symbol": symbol, "best_run": None, "message": "No training history found"}
    return {
        "symbol": symbol,
        "best_run": best,
        "message": f"Best run (id={best['id']}): composite_score = {best.get('composite_score', 0):.2f}",
    }


@router.post("/train/async")
async def train_async(request: TrainRequest):
    db = app_state["db"]
    trainer = app_state["trainer"]
    if not db or not trainer:
        raise HTTPException(500, "Service not initialized")

    symbols = request.symbols if request.symbols else list(STOCK_UNIVERSE.keys())
    symbol = symbols[0] if symbols else "unknown"
    task_id = task_manager.create_task(symbol, request.model_type)

    def _run_training():
        try:
            task_manager.update_task(task_id, status="running", progress=10, message=f"Starting training for {symbol}...")

            def _on_symbol_start(sym: str, idx: int, total: int):
                pct = 10 + int((idx / max(total, 1)) * 70)
                task_manager.update_task(task_id, progress=pct, message=f"Training {sym} ({idx + 1}/{total})...")

            result = run_training_for_symbols(
                trainer=trainer,
                db=db,
                model_type=request.model_type,
                walk_forward=request.walk_forward,
                symbols=symbols,
                on_symbol_start=_on_symbol_start,
                cancel_requested=lambda: task_manager.is_cancel_requested(task_id),
            )

            if task_manager.is_cancel_requested(task_id):
                task_manager.cancel_task(task_id)
                return

            refresh_loaded_models(app_state)
            task_manager.complete_task(task_id, result)
        except Exception as e:
            task_manager.fail_task(task_id, str(e))

    thread = threading.Thread(target=_run_training, daemon=True)
    thread.start()
    return {"task_id": task_id, "status": "started", "symbol": symbol}


@router.get("/train/status/{task_id}")
async def get_train_status(task_id: str):
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(404, f"Task {task_id} not found")
    return task


@router.get("/train/active")
async def get_active_tasks():
    return {"tasks": task_manager.get_active_tasks(), "all": task_manager.get_all_tasks()}


@router.post("/train/cancel/{task_id}")
async def cancel_train_task(task_id: str):
    accepted = task_manager.request_cancel(task_id)
    if not accepted:
        raise HTTPException(404, f"Active task {task_id} not found or cannot be cancelled")
    return {"status": "cancelling", "task_id": task_id}


@router.delete("/train/dismiss/{task_id}")
async def dismiss_task(task_id: str):
    task_manager.dismiss_task(task_id)
    return {"status": "dismissed", "task_id": task_id}

