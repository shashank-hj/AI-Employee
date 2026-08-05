"""M5 Memory Writer — background worker that extracts facts from completed conversations\n"
"and writes them to User Profile (M2) and Episodic Memory (M3).\n"
"\n"
"Runs as a fire-and-forget asyncio task alongside the orchestrator. Uses a Redis\n"
"list as a simple job queue for durability and retry resilience.\n"""

import asyncio
import json
from typing import Any

import redis.asyncio as aioredis
import structlog

from orchestrator.config import settings
from orchestrator.services.fact_extractor import FactExtractor
from orchestrator.services.memory_client import MemoryClient

logger = structlog.get_logger(__name__)

DEFAULT_QUEUE_KEY = "memory_writer:queue"
DEFAULT_POLL_TIMEOUT = 5  # seconds for BRPOP
DEFAULT_RETRY_DELAY = 2.0  # seconds between failed processing retries


class MemoryWriterWorker:
    """Background worker that consumes conversation jobs from Redis and writes memories."""

    def __init__(
        self,
        redis_client: aioredis.Redis,
        fact_extractor: FactExtractor,
        memory_client: MemoryClient,
        queue_key: str = DEFAULT_QUEUE_KEY,
        retry_delay: float = DEFAULT_RETRY_DELAY,
        enabled: bool = True,
    ) -> None:
        self._redis = redis_client
        self._extractor = fact_extractor
        self._memory = memory_client
        self._queue_key = queue_key
        self._retry_delay = retry_delay
        self._enabled = enabled
        self._task: asyncio.Task | None = None
        self._stop_event = asyncio.Event()

    # ── Lifecycle ──

    def start(self) -> None:
        if not self._enabled:
            logger.info("memory_writer_disabled")
            return
        if self._task is not None and not self._task.done():
            logger.warning("memory_writer_already_running")
            return
        self._stop_event.clear()
        self._task = asyncio.create_task(self._run_loop())
        logger.info("memory_writer_started", queue_key=self._queue_key)

    async def stop(self, timeout: float = 5.0) -> None:
        if self._task is None or self._task.done():
            return
        self._stop_event.set()
        try:
            await asyncio.wait_for(self._task, timeout=timeout)
        except asyncio.TimeoutError:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("memory_writer_stopped")

    async def enqueue(self, job: dict[str, Any]) -> bool:
        """Enqueue a conversation job for background processing.\n\n"
        "Returns True if successfully queued, False otherwise."
        """
        if not self._enabled:
            logger.debug("memory_writer_enqueue_skipped_disabled")
            return False
        try:
            payload = json.dumps(job, default=str)
            await self._redis.rpush(self._queue_key, payload)
            logger.info(
                "memory_writer_enqueued",
                request_id=job.get("request_id"),
                user_id=job.get("user_id"),
                queue_length=await self._redis.llen(self._queue_key),
            )
            return True
        except Exception as exc:
            logger.error("memory_writer_enqueue_failed", error=str(exc), request_id=job.get("request_id"))
            return False

    # ── Core loop ──

    async def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                payload = await self._redis.lpop(self._queue_key)
                if payload is None:
                    await asyncio.sleep(self._retry_delay)
                    continue

                job = json.loads(payload)
                await self._process_job(job)
            except asyncio.CancelledError:
                logger.info("memory_writer_loop_cancelled")
                break
            except json.JSONDecodeError as exc:
                logger.error("memory_writer_bad_payload", error=str(exc), payload=payload[:200])
            except asyncio.TimeoutError:
                await asyncio.sleep(self._retry_delay)
            except Exception as exc:
                logger.error("memory_writer_loop_error", error=str(exc))
                await asyncio.sleep(self._retry_delay)

    async def _process_job(self, job: dict[str, Any]) -> None:
        request_id = job.get("request_id", "unknown")
        user_id = job.get("user_id")
        session_id = job.get("session_id")

        if not user_id:
            logger.warning("memory_writer_no_user_id", request_id=request_id)
            return

        logger.info(
            "memory_writer_processing",
            request_id=request_id,
            user_id=user_id,
            session_id=session_id,
        )

        # 1. Extract facts via LLM
        facts = await self._extractor.extract(job)

        if not facts.display_name and not facts.preferences and not facts.facts and not facts.summary:
            logger.info("memory_writer_no_facts_extracted", request_id=request_id, user_id=user_id)
            return

        # 2. Update user profile (M2) — merge preferences & display_name
        profile_preferences: dict[str, Any] = {}
        if facts.preferences:
            profile_preferences.update(facts.preferences)
        if facts.sentiment:
            profile_preferences["sentiment"] = facts.sentiment
        if facts.topics:
            profile_preferences["recent_topics"] = facts.topics

        profile_metadata = {
            "last_session_id": session_id,
            "last_request_id": request_id,
        }

        await self._memory.update_profile(
            user_id=user_id,
            display_name=facts.display_name,
            preferences=profile_preferences if profile_preferences else None,
            metadata=profile_metadata,
        )

        # 3. Store episodic summary (M3) as long-term memory
        if facts.summary:
            await self._memory.store_long_term(
                user_id=user_id,
                content=facts.summary,
                memory_type="summary",
                importance=0.7,
                source=request_id,
                metadata={
                    "topics": facts.topics,
                    "sentiment": facts.sentiment,
                    "session_id": session_id,
                },
            )

        # 4. Store individual facts (M3)
        for fact in facts.facts:
            await self._memory.store_long_term(
                user_id=user_id,
                content=fact,
                memory_type="fact",
                importance=0.6,
                source=request_id,
                metadata={
                    "session_id": session_id,
                },
            )

        logger.info(
            "memory_writer_completed",
            request_id=request_id,
            user_id=user_id,
            facts_count=len(facts.facts),
            has_summary=bool(facts.summary),
        )
