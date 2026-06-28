from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from threading import Lock
from typing import Awaitable, Callable

from service.pdf.index_progress import get_index_progress_tracker

IndexJobRunner = Callable[[], Awaitable[None]]


@dataclass(frozen=True)
class DocumentIndexJob:
    task_id: str
    collection_name: str
    runner: IndexJobRunner


class DocumentIndexQueue:
    """Small in-process queue boundary for PDF indexing jobs.

    The API layer enqueues jobs and returns a task immediately. A single worker
    consumes jobs in FIFO order, making the queue replaceable by a real external
    worker later without changing controller behavior.
    """

    def __init__(self) -> None:
        self._queue: asyncio.Queue[DocumentIndexJob] | None = None
        self._worker_task: asyncio.Task[None] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._lock = Lock()
        self._pending: dict[str, str] = {}
        self._active: dict[str, str] = {}
        self._logger = logging.getLogger(__name__)

    def start(self) -> None:
        loop = asyncio.get_running_loop()
        if self._loop is not None and self._loop is not loop and self.is_running:
            raise RuntimeError("Document index queue is already running on another event loop.")
        if self._queue is None or self._loop is not loop:
            self._queue = asyncio.Queue()
            self._loop = loop
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = loop.create_task(self._worker(), name="document-index-worker")

    @property
    def is_running(self) -> bool:
        return self._worker_task is not None and not self._worker_task.done()

    async def stop(self) -> None:
        task = self._worker_task
        if task is None or task.done():
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    async def enqueue(self, *, task_id: str, collection_name: str, runner: IndexJobRunner) -> dict[str, object]:
        self.start()
        assert self._queue is not None
        collection = (collection_name or "default").strip() or "default"
        with self._lock:
            self._pending[task_id] = collection
        get_index_progress_tracker().mark_queued(task_id)
        await self._queue.put(DocumentIndexJob(task_id=task_id, collection_name=collection, runner=runner))
        return {"task_id": task_id, "collection_name": collection, "queue_size": self._queue.qsize()}

    def has_collection_work(self, collection_name: str) -> bool:
        return self.get_collection_work(collection_name) is not None

    def get_collection_work(self, collection_name: str) -> dict[str, object] | None:
        collection = (collection_name or "default").strip() or "default"
        with self._lock:
            task_ids = [
                task_id
                for task_id, active_collection in self._active.items()
                if active_collection == collection
            ]
            task_ids.extend(
                task_id
                for task_id, pending_collection in self._pending.items()
                if pending_collection == collection
            )
        tracker = get_index_progress_tracker()
        for task_id in task_ids:
            task = tracker.get(task_id)
            if task and task.get("status") in {"pending", "queued", "running"}:
                return task
        return None

    async def join(self) -> None:
        if self._queue is None:
            return
        await self._queue.join()

    async def _worker(self) -> None:
        assert self._queue is not None
        tracker = get_index_progress_tracker()
        while True:
            job = await self._queue.get()
            try:
                with self._lock:
                    self._pending.pop(job.task_id, None)
                    self._active[job.task_id] = job.collection_name
                await job.runner()
            except asyncio.CancelledError:
                tracker.fail(job.task_id, "Indexing worker stopped before task completed.")
                raise
            except Exception as exc:
                self._logger.exception("document index job failed", extra={"task_id": job.task_id})
                tracker.fail(job.task_id, str(exc))
            finally:
                with self._lock:
                    self._active.pop(job.task_id, None)
                self._queue.task_done()


_INDEX_QUEUE = DocumentIndexQueue()


def get_document_index_queue() -> DocumentIndexQueue:
    return _INDEX_QUEUE