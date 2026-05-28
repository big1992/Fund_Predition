"""
In-memory background task manager for training jobs.
Tracks status, progress, and results so users can navigate away.
"""

import uuid
import threading
import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


class TrainingTask:
    """Represents a single training task."""

    def __init__(self, task_id: str, symbol: str, model_type: str):
        self.task_id = task_id
        self.symbol = symbol
        self.model_type = model_type
        self.status = "pending"  # pending, running, completed, failed
        self.progress = 0  # 0-100
        self.message = "กำลังเตรียมข้อมูล..."
        self.result = None
        self.error = None
        self.created_at = datetime.now().isoformat()
        self.completed_at = None

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
        }


class TaskManager:
    """Manages background training tasks."""

    def __init__(self):
        self._tasks: dict[str, TrainingTask] = {}
        self._lock = threading.Lock()

    def create_task(self, symbol: str, model_type: str) -> str:
        """Create a new training task and return its ID."""
        task_id = str(uuid.uuid4())[:8]
        task = TrainingTask(task_id, symbol, model_type)
        with self._lock:
            self._tasks[task_id] = task
        logger.info("Created training task %s for %s", task_id, symbol)
        return task_id

    def get_task(self, task_id: str) -> Optional[dict]:
        """Get task status."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task:
                return task.to_dict()
        return None

    def get_active_tasks(self) -> list[dict]:
        """Get all active (non-completed) tasks."""
        with self._lock:
            return [
                t.to_dict() for t in self._tasks.values()
                if t.status in ("pending", "running")
            ]

    def get_all_tasks(self) -> list[dict]:
        """Get all tasks (for listing)."""
        with self._lock:
            return [t.to_dict() for t in self._tasks.values()]

    def update_task(self, task_id: str, status: str = None,
                    progress: int = None, message: str = None):
        """Update task progress."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task:
                if status:
                    task.status = status
                if progress is not None:
                    task.progress = progress
                if message:
                    task.message = message

    def complete_task(self, task_id: str, result: dict):
        """Mark task as completed with results."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task:
                task.status = "completed"
                task.progress = 100
                task.message = "✅ Training เสร็จสมบูรณ์"
                task.result = result
                task.completed_at = datetime.now().isoformat()
        logger.info("Task %s completed", task_id)

    def fail_task(self, task_id: str, error: str):
        """Mark task as failed."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task:
                task.status = "failed"
                task.message = f"❌ Error: {error}"
                task.error = error
                task.completed_at = datetime.now().isoformat()
        logger.error("Task %s failed: %s", task_id, error)

    def dismiss_task(self, task_id: str):
        """Remove a completed/failed task."""
        with self._lock:
            if task_id in self._tasks and self._tasks[task_id].status in ("completed", "failed"):
                del self._tasks[task_id]


# Global singleton
task_manager = TaskManager()
