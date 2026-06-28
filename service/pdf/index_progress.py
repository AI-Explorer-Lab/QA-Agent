from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from threading import Lock
from typing import Any
from uuid import uuid4


STEP_ORDER = ["upload", "collect", "ocr", "chunking", "embedding", "database"]
STEP_LABELS = {
    "upload": "上传文件",
    "collect": "收集文档",
    "ocr": "OCR / PDF 解析",
    "chunking": "chunking",
    "embedding": "embedding",
    "database": "入库",
}

EVENT_TO_STEP = {
    "index.collect_documents": "collect",
    "index.ocr": "ocr",
    "index.chunking": "chunking",
    "index.embedding": "embedding",
    "index.database": "database",
}

STATUS_PROGRESS = {
    "pending": 0,
    "queued": 5,
    "started": 8,
    "running": 50,
    "completed": 100,
    "skipped": 100,
    "failed": 100,
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _initial_steps() -> list[dict[str, Any]]:
    return [
        {
            "key": key,
            "label": STEP_LABELS[key],
            "status": "pending",
            "progress": 0,
            "detail": "等待后端开始",
            "updated_at": "",
        }
        for key in STEP_ORDER
    ]


class IndexProgressTracker:
    def __init__(self) -> None:
        self._lock = Lock()
        self._tasks: dict[str, dict[str, Any]] = {}

    def create(self, *, collection_name: str, file_name: str = "") -> dict[str, Any]:
        task_id = str(uuid4())
        task = {
            "task_id": task_id,
            "status": "pending",
            "collection_name": collection_name,
            "file_name": file_name,
            "created_at": _now(),
            "updated_at": _now(),
            "steps": _initial_steps(),
            "result": None,
            "error": "",
        }
        with self._lock:
            self._tasks[task_id] = task
        return deepcopy(task)

    def get(self, task_id: str) -> dict[str, Any] | None:
        with self._lock:
            task = self._tasks.get(task_id)
            return deepcopy(task) if task else None

    def mark_queued(self, task_id: str, detail: str = "Waiting for indexing worker") -> dict[str, Any] | None:
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return None
            task["status"] = "queued"
            task["updated_at"] = _now()
            for step in task["steps"]:
                if step["status"] == "pending":
                    step["detail"] = detail
                    step["updated_at"] = task["updated_at"]
                    break
            return deepcopy(task)

    def update_step(
        self,
        task_id: str,
        step_key: str,
        *,
        status: str,
        progress: int | None = None,
        detail: str = "",
        fields: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return None
            task["status"] = "running" if status in {"started", "running"} else task.get("status", "running")
            task["updated_at"] = _now()
            for step in task["steps"]:
                if step["key"] != step_key:
                    continue
                step["status"] = status
                step["progress"] = int(progress if progress is not None else STATUS_PROGRESS.get(status, 0))
                step["detail"] = detail or step.get("detail") or ""
                step["updated_at"] = task["updated_at"]
                if fields:
                    step["fields"] = fields
                break
            return deepcopy(task)

    def complete(self, task_id: str, result: dict[str, Any]) -> dict[str, Any] | None:
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return None
            task["status"] = "completed"
            task["result"] = result
            task["updated_at"] = _now()
            if isinstance(result.get("pipeline_steps"), list) and result["pipeline_steps"]:
                task["steps"] = result["pipeline_steps"]
                return deepcopy(task)
            for step in task["steps"]:
                if step["status"] in {"pending", "started", "running"}:
                    step["status"] = "completed"
                    step["progress"] = 100
                    step["detail"] = step.get("detail") or "后端已完成"
                    step["updated_at"] = task["updated_at"]
            return deepcopy(task)

    def fail(self, task_id: str, error: str) -> dict[str, Any] | None:
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return None
            task["status"] = "failed"
            task["error"] = error
            task["updated_at"] = _now()
            for step in task["steps"]:
                if step["status"] in {"started", "running"}:
                    step["status"] = "failed"
                    step["progress"] = 100
                    step["detail"] = error
                    step["updated_at"] = task["updated_at"]
                    break
            return deepcopy(task)


_TRACKER = IndexProgressTracker()


def get_index_progress_tracker() -> IndexProgressTracker:
    return _TRACKER
