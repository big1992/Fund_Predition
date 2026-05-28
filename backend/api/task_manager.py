"""
Background training task manager with SQLite persistence.
"""

import json
import logging
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from config.settings import settings

logger = logging.getLogger(__name__)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class TrainingTask:
    """Represents a single training task."""

    def __init__(self, task_id: str, symbol: str, model_type: str):
        self.task_id = task_id
        self.symbol = symbol
        self.model_type = model_type
        self.status = "pending"  # pending, running, cancelling, cancelled, completed, failed
        self.progress = 0
        self.message = "Preparing training data..."
        self.result = None
        self.error = None
        self.created_at = _utc_now()
        self.completed_at = None
        self.cancel_requested = False

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "symbol": self.symbol,
            "model_type": self.model_type,
            "status": self.status,
            "progress": self.progress,
            "message": self.message,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "cancel_requested": self.cancel_requested,
        }

    @staticmethod
    def from_row(row: sqlite3.Row) -> "TrainingTask":
        task = TrainingTask(row["task_id"], row["symbol"], row["model_type"])
        task.status = row["status"]
        task.progress = row["progress"]
        task.message = row["message"]
        task.error = row["error"]
        task.created_at = row["created_at"]
        task.completed_at = row["completed_at"]
        task.cancel_requested = bool(row["cancel_requested"])
        try:
            task.result = json.loads(row["result_json"]) if row["result_json"] else None
        except Exception:
            task.result = None
        return task


class TaskManager:
    """Manages background training tasks and persists them to SQLite."""

    def __init__(self):
        self._tasks: dict[str, TrainingTask] = {}
        self._lock = threading.Lock()
        self._db_path = settings.db_path
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self._load_recent_tasks()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        conn = self._conn()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS training_tasks (
                    task_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    progress INTEGER NOT NULL DEFAULT 0,
                    message TEXT,
                    result_json TEXT,
                    error TEXT,
                    cancel_requested INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    completed_at TEXT
                )
                """
            )
            conn.commit()
        finally:
            conn.close()

    def _persist_task(self, task: TrainingTask) -> None:
        conn = self._conn()
        try:
            conn.execute(
                """
                INSERT INTO training_tasks (
                    task_id, symbol, model_type, status, progress, message,
                    result_json, error, cancel_requested, created_at, completed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(task_id) DO UPDATE SET
                    status=excluded.status,
                    progress=excluded.progress,
                    message=excluded.message,
                    result_json=excluded.result_json,
                    error=excluded.error,
                    cancel_requested=excluded.cancel_requested,
                    completed_at=excluded.completed_at
                """,
                (
                    task.task_id,
                    task.symbol,
                    task.model_type,
                    task.status,
                    task.progress,
                    task.message,
                    json.dumps(task.result, default=str) if task.result is not None else None,
                    task.error,
                    1 if task.cancel_requested else 0,
                    task.created_at,
                    task.completed_at,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def _load_recent_tasks(self, limit: int = 100) -> None:
        conn = self._conn()
        try:
            rows = conn.execute(
                """
                SELECT * FROM training_tasks
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        finally:
            conn.close()

        with self._lock:
            for row in rows:
                task = TrainingTask.from_row(row)
                if task.status == "cancelling":
                    task.status = "cancelled"
                    task.message = "Training cancelled after restart."
                    task.completed_at = task.completed_at or _utc_now()
                    task.cancel_requested = False
                    self._persist_task(task)
                self._tasks[task.task_id] = task

    def create_task(self, symbol: str, model_type: str) -> str:
        task_id = str(uuid.uuid4())[:8]
        task = TrainingTask(task_id, symbol, model_type)
        with self._lock:
            self._tasks[task_id] = task
            self._persist_task(task)
        logger.info("Created training task %s for %s", task_id, symbol)
        return task_id

    def get_task(self, task_id: str) -> Optional[dict]:
        with self._lock:
            task = self._tasks.get(task_id)
            return task.to_dict() if task else None

    def get_active_tasks(self) -> list[dict]:
        with self._lock:
            return [
                t.to_dict()
                for t in self._tasks.values()
                if t.status in ("pending", "running", "cancelling")
            ]

    def get_all_tasks(self) -> list[dict]:
        with self._lock:
            return sorted(
                [t.to_dict() for t in self._tasks.values()],
                key=lambda t: t["created_at"],
                reverse=True,
            )

    def update_task(
        self,
        task_id: str,
        status: str = None,
        progress: int = None,
        message: str = None,
    ) -> None:
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return
            if status:
                task.status = status
            if progress is not None:
                task.progress = progress
            if message:
                task.message = message
            self._persist_task(task)

    def complete_task(self, task_id: str, result: dict) -> None:
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return
            task.status = "completed"
            task.progress = 100
            task.message = "Training completed successfully."
            task.result = result
            task.completed_at = _utc_now()
            self._persist_task(task)
        logger.info("Task %s completed", task_id)

    def fail_task(self, task_id: str, error: str) -> None:
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return
            task.status = "failed"
            task.message = f"Error: {error}"
            task.error = error
            task.completed_at = _utc_now()
            self._persist_task(task)
        logger.error("Task %s failed: %s", task_id, error)

    def request_cancel(self, task_id: str) -> bool:
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return False
            if task.status not in ("pending", "running", "cancelling"):
                return False
            task.cancel_requested = True
            if task.status != "pending":
                task.status = "cancelling"
            task.message = "Cancellation requested..."
            self._persist_task(task)
            return True

    def is_cancel_requested(self, task_id: str) -> bool:
        with self._lock:
            task = self._tasks.get(task_id)
            return bool(task and task.cancel_requested)

    def cancel_task(self, task_id: str) -> None:
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return
            task.status = "cancelled"
            task.progress = min(task.progress, 99)
            task.message = "Training cancelled."
            task.completed_at = _utc_now()
            task.cancel_requested = False
            self._persist_task(task)

    def dismiss_task(self, task_id: str) -> None:
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return
            if task.status not in ("completed", "failed", "cancelled"):
                return
            del self._tasks[task_id]
        conn = self._conn()
        try:
            conn.execute("DELETE FROM training_tasks WHERE task_id = ?", (task_id,))
            conn.commit()
        finally:
            conn.close()


task_manager = TaskManager()

