"""
FastAPI application entry point.
"""

import sys
import os
import time
import logging
from pathlib import Path
from contextlib import asynccontextmanager

# ── Suppress TF noise and eagerly initialise TensorFlow on the MAIN thread ──
# On Windows, TF's DLL initialisation fails if first triggered inside a
# uvicorn threadpool worker.  Importing (and running a tiny op) here forces
# the DLL to load before any request handler runs.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
try:
    import tensorflow as tf
    # Run a trivial op to force full GPU/CPU device enumeration now
    _ = tf.constant(0)
except Exception as _tf_err:
    # If TF is broken we log it clearly and continue (other models still work)
    import warnings
    warnings.warn(f"TensorFlow failed to initialise: {_tf_err}")

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Load .env
load_dotenv()

# Add backend to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import settings
from data.storage import DatabaseManager
from models.trainer import ModelTrainer
from models.scheduler import RetrainScheduler
from ai.explainer import AIExplainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Global state
app_state = {
    "start_time": time.time(),
    "db": None,
    "trainer": None,
    "models_loaded": [],
    "scheduler": None,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources on startup, cleanup on shutdown."""
    logger.info("🚀 Starting %s v%s", settings.app_name, settings.app_version)

    # Init database
    app_state["db"] = DatabaseManager(settings.db_path)
    logger.info("📦 Database initialized: %s", settings.db_path)

    # Init trainer
    app_state["trainer"] = ModelTrainer()
    logger.info("🤖 ML Trainer initialized")

    # Init scheduler (not started by default — user controls via API)
    app_state["scheduler"] = RetrainScheduler(
        trainer=app_state["trainer"],
        db=app_state["db"],
    )
    logger.info("⏰ Retrain scheduler initialized (start via API)")

    # Check for saved models
    model_dir = Path(settings.model_dir)
    if model_dir.exists():
        for f in model_dir.iterdir():
            if f.suffix in (".keras", ".joblib"):
                app_state["models_loaded"].append(f.stem)
            elif f.is_dir() and f.name.startswith("ag_"):
                app_state["models_loaded"].append(f.name)
        if app_state["models_loaded"]:
            logger.info("📂 Found %d saved models", len(app_state["models_loaded"]))

    # Init AI explainer
    app_state["ai_explainer"] = AIExplainer()
    if app_state["ai_explainer"].is_available():
        logger.info("🧠 AI Explainer ready (GPT-4o-mini)")
    else:
        logger.warning("⚠️ AI Explainer unavailable — set OPENAI_API_KEY in .env")

    yield

    # Shutdown
    if app_state["scheduler"]:
        app_state["scheduler"].stop()
    logger.info("👋 Shutting down %s", settings.app_name)


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="ระบบวิเคราะห์และพยากรณ์กองทุนรวมหุ้นไทย",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
from api.routes.system import router as system_router
from api.routes.stocks import router as stocks_router
from api.routes.funds import router as funds_router
from api.routes.predictions import router as predictions_router
from api.routes.portfolio import router as portfolio_router
from api.routes.backtest import router as backtest_router
from api.routes.ai import router as ai_router
from api.routes.settings import router as settings_router
from api.routes.news import router as news_router

app.include_router(system_router, prefix="/api", tags=["System"])
app.include_router(stocks_router, prefix="/api/stocks", tags=["Stocks"])
app.include_router(funds_router, prefix="/api/funds", tags=["Funds"])
app.include_router(predictions_router, prefix="/api/predictions", tags=["Predictions"])
app.include_router(portfolio_router, prefix="/api/portfolio", tags=["Portfolio"])
app.include_router(backtest_router, prefix="/api/backtest", tags=["Backtest"])
app.include_router(ai_router, prefix="/api/ai", tags=["AI"])
app.include_router(settings_router, prefix="/api/settings", tags=["Settings"])
app.include_router(news_router, prefix="/api/news", tags=["News"])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host=settings.api_host, port=settings.api_port, reload=True)
