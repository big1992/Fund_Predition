"""
Auto-retrain scheduler for stock prediction models.
Runs periodic model retraining in background using asyncio.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional

from config.settings import STOCK_UNIVERSE, DATA_SETTINGS

logger = logging.getLogger(__name__)


class RetrainScheduler:
    """Background scheduler that retrains models periodically."""

    def __init__(self, trainer, db, interval_hours: int = 168):  # default: weekly
        self.trainer = trainer
        self.db = db
        self.interval_hours = interval_hours
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self.last_retrain: Optional[datetime] = None
        self.retrain_history = []

    def start(self):
        """Start the background scheduler."""
        if self._running:
            logger.info("Scheduler already running")
            return
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info("🔄 Auto-retrain scheduler started (every %dh)", self.interval_hours)

    def stop(self):
        """Stop the background scheduler."""
        self._running = False
        if self._task:
            self._task.cancel()
            self._task = None
        logger.info("⏹️ Auto-retrain scheduler stopped")

    async def _run_loop(self):
        """Main scheduler loop."""
        while self._running:
            try:
                await asyncio.sleep(self.interval_hours * 3600)
                if self._running:
                    await self._retrain_all()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Scheduler error: %s", e)
                await asyncio.sleep(600)  # wait 10min on error

    async def _retrain_all(self):
        """Retrain all models."""
        logger.info("🔄 Auto-retrain triggered at %s", datetime.now().isoformat())
        results = {}

        for symbol in STOCK_UNIVERSE.keys():
            try:
                df = self.db.get_stock_prices(symbol)
                if df.empty or len(df) < 200:
                    continue

                # Run training in thread pool (CPU-bound)
                loop = asyncio.get_event_loop()
                metrics = await loop.run_in_executor(
                    None, self.trainer.train_all, df, symbol
                )
                results[symbol] = "success"
                logger.info("✅ Retrained %s", symbol)

            except Exception as e:
                results[symbol] = f"error: {e}"
                logger.error("❌ Retrain %s failed: %s", symbol, e)

        self.last_retrain = datetime.now()
        self.retrain_history.append({
            "timestamp": self.last_retrain.isoformat(),
            "results": results,
        })
        # Keep last 10 entries
        if len(self.retrain_history) > 10:
            self.retrain_history = self.retrain_history[-10:]

        logger.info("🔄 Auto-retrain complete: %d symbols processed", len(results))

    async def retrain_now(self):
        """Trigger immediate retrain."""
        await self._retrain_all()

    def get_status(self) -> dict:
        """Return scheduler status."""
        return {
            "running": self._running,
            "interval_hours": self.interval_hours,
            "last_retrain": self.last_retrain.isoformat() if self.last_retrain else None,
            "next_retrain": (
                (self.last_retrain + timedelta(hours=self.interval_hours)).isoformat()
                if self.last_retrain else None
            ),
            "history_count": len(self.retrain_history),
        }
