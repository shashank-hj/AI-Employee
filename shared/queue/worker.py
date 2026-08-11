"""Generic background worker for :class:`RedisTaskQueue` (shared).

Runs one asyncio task that polls the queue and dispatches each job to the handler
registered for its ``task`` name. Mirrors the lifecycle used by the memory-writer
worker: ``start()`` / ``stop()`` / ``enqueue()``.
"""

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

import structlog

from shared.queue.queue import RedisTaskQueue

logger = structlog.get_logger(__name__)

DEFAULT_RETRY_DELAY = 1.0


class RedisTaskWorker:
    """Consumes jobs from a :class:`RedisTaskQueue` and dispatches by task name."""

    def __init__(
        self,
        queue: RedisTaskQueue,
        retry_delay: float = DEFAULT_RETRY_DELAY,
    ) -> None:
        self._queue = queue
        self._retry_delay = retry_delay
        self._handlers: dict[str, Callable[[dict[str, Any]], Awaitable[None]]] = {}
        self._task: asyncio.Task | None = None
        self._stop_event = asyncio.Event()

    def register(self, task: str, handler: Callable[[dict[str, Any]], Awaitable[None]]) -> None:
        self._handlers[task] = handler

    def start(self) -> None:
        if self._task is not None and not self._task.done():
            logger.warning("task_worker_already_running", queue=self._queue.queue_key)
            return
        self._stop_event.clear()
        self._task = asyncio.create_task(self._run_loop())
        logger.info("task_worker_started", queue=self._queue.queue_key, tasks=list(self._handlers))

    async def stop(self, timeout: float = 5.0) -> None:
        if self._task is None or self._task.done():
            return
        self._stop_event.set()
        try:
            await asyncio.wait_for(self._task, timeout=timeout)
        except TimeoutError:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("task_worker_stopped", queue=self._queue.queue_key)

    async def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                job = await self._queue.poll()
                if job is None:
                    await asyncio.sleep(self._retry_delay)
                    continue
                await self._dispatch(job)
            except asyncio.CancelledError:
                logger.info("task_worker_loop_cancelled", queue=self._queue.queue_key)
                break
            except Exception as exc:
                logger.error("task_worker_loop_error", error=str(exc), queue=self._queue.queue_key)
                await asyncio.sleep(self._retry_delay)

    async def _dispatch(self, job: dict[str, Any]) -> None:
        task = job.get("task", "unknown")
        handler = self._handlers.get(task)
        if handler is None:
            logger.warning(
                "task_worker_no_handler",
                task=task,
                job_id=job.get("job_id"),
                queue=self._queue.queue_key,
            )
            return
        try:
            await handler(job.get("payload", {}))
        except Exception as exc:
            logger.error(
                "task_handler_failed",
                task=task,
                job_id=job.get("job_id"),
                error=str(exc),
            )
