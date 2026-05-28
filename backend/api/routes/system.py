"""
System routes: health check, status, scheduler control.
"""

import time
from fastapi import APIRouter, HTTPException
from schemas.api_schemas import HealthResponse
from api.main import app_state
from config.settings import settings

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    db = app_state["db"]
    data_available = False
    if db:
        stats = db.get_data_stats()
        data_available = stats.get("total_price_records", 0) > 0

    return HealthResponse(
        status="healthy",
        version=settings.app_version,
        uptime_seconds=round(time.time() - app_state["start_time"], 1),
        models_loaded=app_state.get("models_loaded", []),
        data_available=data_available,
    )


@router.get("/scheduler/status")
async def scheduler_status():
    """Get auto-retrain scheduler status."""
    scheduler = app_state.get("scheduler")
    if not scheduler:
        return {"running": False, "message": "Scheduler not initialized"}
    return scheduler.get_status()


@router.post("/scheduler/start")
async def scheduler_start(interval_hours: int = 168):
    """Start auto-retrain scheduler."""
    scheduler = app_state.get("scheduler")
    if not scheduler:
        raise HTTPException(500, "Scheduler not initialized")
    scheduler.interval_hours = interval_hours
    scheduler.start()
    return {"status": "started", "interval_hours": interval_hours}


@router.post("/scheduler/stop")
async def scheduler_stop():
    """Stop auto-retrain scheduler."""
    scheduler = app_state.get("scheduler")
    if not scheduler:
        raise HTTPException(500, "Scheduler not initialized")
    scheduler.stop()
    return {"status": "stopped"}


@router.post("/scheduler/retrain-now")
async def scheduler_retrain_now():
    """Trigger immediate retrain of all models."""
    scheduler = app_state.get("scheduler")
    if not scheduler:
        raise HTTPException(500, "Scheduler not initialized")
    await scheduler.retrain_now()
    return {"status": "retrain_complete", "timestamp": scheduler.last_retrain.isoformat()}
