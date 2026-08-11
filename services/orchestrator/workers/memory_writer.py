"""M5 Memory Writer — background worker that extracts facts from completed conversations\n"
"and writes them to User Profile (M2) and Episodic Memory (M3).\n"
"\n"
"Runs as a fire-and-forget asyncio task alongside the orchestrator. Uses the shared\n"
"Redis task queue for durability and retry resilience.\n"""

from typing import Any

import redis.asyncio as aioredis
import structlog

from orchestrator.services.fact_extractor import FactExtractor
from orchestrator.services.memory_client import MemoryClient
from shared.queue import RedisTaskQueue, RedisTaskWorker
from shared.usage.context import reset_usage_context, set_usage_context

logger = structlog.get_logger(__name__)

DEFAULT_QUEUE_KEY = "memory_writer:queue"
DEFAULT_POLL_TIMEOUT = 5  # seconds for BRPOP
DEFAULT_RETRY_DELAY = 2.0  # seconds between failed processing retries

TASK_NAME = "memory_writer"


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
        self._queue = RedisTaskQueue(redis_client, queue_key=queue_key)
        self._worker = RedisTaskWorker(self._queue, retry_delay=retry_delay)
        self._worker.register(TASK_NAME, self._process_job)

    # ── Lifecycle ──

    def start(self) -> None:
        if not self._enabled:
            logger.info("memory_writer_disabled")
            return
        self._worker.start()

    async def stop(self, timeout: float = 5.0) -> None:
        if self._enabled:
            await self._worker.stop(timeout=timeout)

    async def enqueue(self, job: dict[str, Any]) -> bool:
        """Enqueue a conversation job for background processing.\n\n"
        "Returns True if successfully queued, False otherwise."
        """
        if not self._enabled:
            logger.debug("memory_writer_enqueue_skipped_disabled")
            return False
        ok = await self._queue.enqueue(TASK_NAME, job)
        if ok:
            logger.info(
                "memory_writer_enqueued",
                request_id=job.get("request_id"),
                user_id=job.get("user_id"),
                queue_length=await self._queue.length(),
            )
        else:
            logger.error("memory_writer_enqueue_failed", request_id=job.get("request_id"))
        return ok

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
        token = set_usage_context(
            operation="fact_extraction",
            request_id=request_id,
            user_id=user_id,
            session_id=session_id,
        )
        try:
            facts = await self._extractor.extract(job)
        finally:
            reset_usage_context(token)

        if not (facts.display_name or facts.preferences or facts.facts or facts.summary):
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
