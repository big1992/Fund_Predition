"""
Prediction routes: get predictions, train models, model performance.
"""

import logging
import threading
import numpy as np
from datetime import datetime, timezone
from fastapi import APIRouter, Query, HTTPException, Response

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
    ModelRegistryEntry,
    ModelRegistryResponse,
    DriftStatusEntry,
    DriftStatusResponse,
    AlertEntry,
    AlertResponse,
    ValidationReportCard,
    ValidationReportResponse,
)
from api.main import app_state
from api.task_manager import task_manager
from data.storage import DatabaseManager
from services.training_service import refresh_loaded_models, run_training_for_symbols
from services.validation_report import build_validation_report
from services.drift_monitor import evaluate_simple_drift
from services.alert_rules import build_watchlist_alerts

router = APIRouter()
logger = logging.getLogger(__name__)


def _mini_pdf_from_lines(lines: list[str]) -> bytes:
    safe_lines = [ln.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)") for ln in lines]
    text_stream = "BT /F1 10 Tf 40 800 Td " + " Tj T* ".join(f"({ln})" for ln in safe_lines[:80]) + " Tj ET"
    objects = []
    objects.append("1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj")
    objects.append("2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj")
    objects.append("3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >> endobj")
    objects.append("4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj")
    objects.append(f"5 0 obj << /Length {len(text_stream)} >> stream\n{text_stream}\nendstream endobj")
    body = "%PDF-1.4\n"
    offsets = []
    for obj in objects:
        offsets.append(len(body.encode("latin-1")))
        body += obj + "\n"
    xref_start = len(body.encode("latin-1"))
    body += f"xref\n0 {len(objects)+1}\n0000000000 65535 f \n"
    for off in offsets:
        body += f"{off:010d} 00000 n \n"
    body += f"trailer << /Size {len(objects)+1} /Root 1 0 R >>\nstartxref\n{xref_start}\n%%EOF"
    return body.encode("latin-1", errors="ignore")


def _build_report_payload(symbol: str, trainer, db) -> dict:
    df = db.get_stock_prices(symbol)
    if df.empty:
        raise HTTPException(404, f"No data for {symbol}")
    pred = trainer.predict(df, symbol, model_type="ensemble", days=5)
    metrics = _symbol_metrics_from_runtime(trainer, symbol) or _symbol_metrics_from_db(symbol)
    report = build_validation_report(symbol=symbol, metrics_by_name=metrics) if metrics else {"cards": []}
    drift = db.get_latest_drift_status(symbol)
    alerts = db.get_alert_events(symbol, limit=10)
    return {
        "symbol": symbol,
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "data_start": str(df.index.min()) if len(df.index) > 0 else None,
        "data_end": str(df.index.max()) if len(df.index) > 0 else None,
        "prediction": pred,
        "validation_cards": report.get("cards", []),
        "drift_entries": drift,
        "alerts": alerts,
        "disclaimer": "Research analytics only. Not personalized investment advice.",
    }


def _metrics_row(model_name: str, symbol: str, metrics: dict) -> ModelMetrics:
    """Convert stored model/baseline metrics into the API response shape."""
    return ModelMetrics(
        model_name=model_name,
        symbol=symbol,
        rmse=metrics.get("rmse", 0),
        mae=metrics.get("mae", 0),
        mape=metrics.get("mape", 0),
        directional_accuracy=metrics.get("directional_accuracy", 0),
        r_squared=metrics.get("r_squared", 0),
        baseline_type=metrics.get("baseline_type"),
        total_return=metrics.get("total_return"),
        max_drawdown=metrics.get("max_drawdown"),
        description=metrics.get("description"),
    )


def _symbol_metrics_from_runtime(trainer, symbol: str) -> dict[str, dict]:
    out = {}
    suffix = f"_{symbol}"
    for key, val in trainer.get_all_metrics().items():
        if not isinstance(val, dict) or not key.endswith(suffix):
            continue
        name = key[: -len(suffix)]
        out[name] = val
    return out


def _symbol_metrics_from_db(symbol: str) -> dict[str, dict]:
    db = DatabaseManager(settings.db_path)
    history = db.get_training_history(symbol, limit=1)
    if not history:
        return {}
    metrics_detail = history[0].get("metrics_detail", {})
    return metrics_detail if isinstance(metrics_detail, dict) else {}


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
            predicted_low=p.get("predicted_low"),
            predicted_high=p.get("predicted_high"),
            uncertainty_pct=p.get("uncertainty_pct"),
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
        model_disagreement_pct=result.get("model_disagreement_pct"),
        confidence_band=result.get("confidence_band"),
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
            parts = key.rsplit("_", 1)
            model_name = parts[0] if len(parts) > 0 else "unknown"
            symbol = parts[1] if len(parts) > 1 else "unknown"
            models.append(_metrics_row(model_name, symbol, metrics))
    else:
        try:
            db = DatabaseManager(settings.db_path)
            for symbol in STOCK_UNIVERSE.keys():
                history = db.get_training_history(symbol, limit=1)
                if not history:
                    continue
                latest = history[0]
                metrics_detail = latest.get("metrics_detail", {})

                for model_name, m in metrics_detail.items():
                    if not isinstance(m, dict) or "error" in m or m.get("mape") is None:
                        continue
                    models.append(_metrics_row(model_name, symbol, m))
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


@router.get("/models/report-card/{symbol}", response_model=ValidationReportResponse)
async def get_validation_report_card(symbol: str):
    trainer = app_state["trainer"]
    if not trainer:
        raise HTTPException(500, "Trainer not initialized")

    metrics = _symbol_metrics_from_runtime(trainer, symbol)
    if not metrics:
        metrics = _symbol_metrics_from_db(symbol)
    if not metrics:
        raise HTTPException(404, f"No training metrics found for {symbol}")

    report = build_validation_report(symbol=symbol, metrics_by_name=metrics)
    return ValidationReportResponse(
        symbol=report["symbol"],
        generated_at_utc=report["generated_at_utc"],
        cards=[ValidationReportCard(**card) for card in report["cards"]],
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


@router.get("/models/registry/{symbol}", response_model=ModelRegistryResponse)
async def get_model_registry(symbol: str, limit: int = 100):
    db = DatabaseManager(settings.db_path)
    rows = db.get_model_registry(symbol, limit=limit)
    return ModelRegistryResponse(
        symbol=symbol,
        entries=[ModelRegistryEntry(**r) for r in rows],
    )


@router.get("/models/drift/{symbol}", response_model=DriftStatusResponse)
async def get_model_drift_status(symbol: str):
    db = DatabaseManager(settings.db_path)
    history = db.get_training_history(symbol, limit=1)
    if not history:
        raise HTTPException(404, f"No training history found for {symbol}")

    latest = history[0]
    metrics_detail = latest.get("metrics_detail", {}) if isinstance(latest, dict) else {}

    df = db.get_stock_prices(symbol)
    if df.empty or "close" not in df.columns:
        raise HTTPException(404, f"No price data for {symbol}")
    returns = df["close"].pct_change().dropna().tail(30).values.astype(float)

    out = []
    for model_name, m in metrics_detail.items():
        if not isinstance(m, dict) or m.get("error"):
            continue
        train_mape = m.get("mape")
        drift = evaluate_simple_drift(train_mape=train_mape, latest_returns=np.asarray(returns))
        db.save_drift_status(
            symbol=symbol,
            model_name=model_name,
            status=drift["status"],
            drift_score=drift["drift_score"],
            should_retrain=drift["should_retrain"],
            reasons=drift["reasons"],
            metrics=drift["metrics"],
        )

    out = db.get_latest_drift_status(symbol)
    return DriftStatusResponse(
        symbol=symbol,
        entries=[DriftStatusEntry(**e) for e in out],
    )


@router.get("/alerts/{symbol}", response_model=AlertResponse)
async def get_watchlist_alerts(symbol: str, limit: int = 20):
    db = app_state["db"]
    trainer = app_state["trainer"]
    if not db or not trainer:
        raise HTTPException(500, "Service not initialized")

    df = db.get_stock_prices(symbol)
    if df.empty:
        raise HTTPException(404, f"No price data for {symbol}")

    pred = trainer.predict(df, symbol, model_type="ensemble", days=5)
    preds = pred.get("predictions", [])
    current_price = float(pred.get("current_price", 0.0) or 0.0)
    avg_interval_pct = 0.0
    if preds and current_price > 0:
        widths = []
        for p in preds:
            lo = p.get("predicted_low")
            hi = p.get("predicted_high")
            if isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
                widths.append((float(hi) - float(lo)) / max(current_price, 1e-9))
        if widths:
            avg_interval_pct = float(np.mean(widths))

    drift_entries = db.get_latest_drift_status(symbol)
    metrics = _symbol_metrics_from_runtime(trainer, symbol) or _symbol_metrics_from_db(symbol)
    report = build_validation_report(symbol=symbol, metrics_by_name=metrics) if metrics else {"cards": []}
    cards = report.get("cards", [])

    previous_signal = None
    prior_alerts = db.get_alert_events(symbol, limit=50)
    for a in prior_alerts:
        if a.get("alert_type") == "signal_change":
            payload = a.get("payload", {})
            previous_signal = payload.get("current_signal")
            if previous_signal:
                break

    alerts = build_watchlist_alerts(
        symbol=symbol,
        current_signal=pred.get("signal", "HOLD"),
        previous_signal=previous_signal,
        avg_interval_pct=avg_interval_pct,
        drift_entries=drift_entries,
        validation_cards=cards,
    )

    for a in alerts:
        db.save_alert_event(
            symbol=symbol,
            alert_type=a["alert_type"],
            severity=a["severity"],
            message=a["message"],
            payload=a.get("payload", {}),
        )

    latest = db.get_alert_events(symbol, limit=limit)
    return AlertResponse(
        symbol=symbol,
        alerts=[AlertEntry(**a) for a in latest],
    )


@router.get("/reports/{symbol}/csv")
async def export_research_report_csv(symbol: str):
    db = app_state["db"]
    trainer = app_state["trainer"]
    if not db or not trainer:
        raise HTTPException(500, "Service not initialized")
    p = _build_report_payload(symbol, trainer, db)
    pred = p.get("prediction", {})
    rows = ["section,key,value"]
    rows.append(f"meta,symbol,{symbol}")
    rows.append(f"meta,generated_at_utc,{p.get('generated_at_utc')}")
    rows.append(f"meta,data_range,{p.get('data_start')} to {p.get('data_end')}")
    rows.append(f"meta,signal,{pred.get('signal','HOLD')}")
    rows.append(f"meta,confidence,{pred.get('confidence',0)}")
    rows.append(f"meta,model_disagreement_pct,{pred.get('model_disagreement_pct')}")
    for i, pp in enumerate(pred.get("predictions", []), start=1):
        rows.append(f"forecast,day_{i}_date,{pp.get('date')}")
        rows.append(f"forecast,day_{i}_base,{pp.get('predicted_price')}")
        rows.append(f"forecast,day_{i}_low,{pp.get('predicted_low')}")
        rows.append(f"forecast,day_{i}_high,{pp.get('predicted_high')}")
        rows.append(f"forecast,day_{i}_confidence,{pp.get('confidence')}")
    for c in p.get("validation_cards", []):
        rows.append(f"validation,{c.get('model_name')}_status,{c.get('status')}")
        rows.append(f"validation,{c.get('model_name')}_mape,{c.get('mape')}")
        rows.append(f"validation,{c.get('model_name')}_dir_acc,{c.get('directional_accuracy')}")
    for d in p.get("drift_entries", []):
        rows.append(f"drift,{d.get('model_name')}_status,{d.get('status')}")
        rows.append(f"drift,{d.get('model_name')}_score,{d.get('drift_score')}")
        rows.append(f"drift,{d.get('model_name')}_retrain,{d.get('should_retrain')}")
    for a in p.get("alerts", []):
        rows.append(f"alert,{a.get('alert_type')},{str(a.get('message','')).replace(',', ';')}")
    rows.append(f"compliance,disclaimer,{p.get('disclaimer')}")
    body = "\n".join(rows)
    return Response(
        content=body,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{symbol.replace(".","_")}_research_report.csv"'},
    )


@router.get("/reports/{symbol}/pdf")
async def export_research_report_pdf(symbol: str):
    db = app_state["db"]
    trainer = app_state["trainer"]
    if not db or not trainer:
        raise HTTPException(500, "Service not initialized")
    p = _build_report_payload(symbol, trainer, db)
    pred = p.get("prediction", {})
    lines = [
        f"Fund Prediction Research Report - {symbol}",
        f"Generated (UTC): {p.get('generated_at_utc')}",
        f"Data range: {p.get('data_start')} to {p.get('data_end')}",
        "",
        f"Signal: {pred.get('signal','HOLD')} | Confidence: {pred.get('confidence',0)} | Band: {pred.get('confidence_band')}",
        f"Model disagreement: {pred.get('model_disagreement_pct')}",
        "",
        "Forecast (downside/base/upside):",
    ]
    for pp in pred.get("predictions", [])[:10]:
        lines.append(f"{pp.get('date')}  {pp.get('predicted_low')} / {pp.get('predicted_price')} / {pp.get('predicted_high')}")
    lines.append("")
    lines.append("Validation status:")
    for c in p.get("validation_cards", [])[:8]:
        lines.append(f"{c.get('model_name')}: {c.get('status')} | MAPE={c.get('mape')} | DirAcc={c.get('directional_accuracy')}")
    lines.append("")
    lines.append("Drift monitor:")
    for d in p.get("drift_entries", [])[:8]:
        lines.append(f"{d.get('model_name')}: {d.get('status')} score={d.get('drift_score')} retrain={d.get('should_retrain')}")
    lines.append("")
    lines.append("Recent alerts:")
    for a in p.get("alerts", [])[:8]:
        lines.append(f"[{a.get('severity')}] {a.get('alert_type')}: {a.get('message')}")
    lines.append("")
    lines.append("Backtest summary: Run Model Performance backtest for latest simulation snapshot.")
    lines.append(f"Disclaimer: {p.get('disclaimer')}")
    pdf_bytes = _mini_pdf_from_lines(lines)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{symbol.replace(".","_")}_research_report.pdf"'},
    )


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
