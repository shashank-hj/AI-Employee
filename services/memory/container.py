from functools import lru_cache

import redis.asyncio as aioredis
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from memory.config import settings
from memory.database.session import get_db
from memory.repositories.conversation import ConversationRepository
from memory.repositories.long_term import LongTermMemoryRepository
from memory.repositories.profile import ProfileRepository
from memory.services.memory_service import MemoryService
from memory.services.stores import (
    BaseEmbeddingService,
    MockEmbeddingService,
    OllamaEmbeddingService,
    SessionStore,
)
from memory.services.summarizer import (
    BaseSessionSummarizer,
    LLMSessionSummarizer,
    MockSessionSummarizer,
)
from shared.llm.base import LLMProvider
from shared.llm.ollama_provider import OllamaProvider
from shared.llm.sarvam_provider import SarvamProvider


@lru_cache
def get_redis_client() -> aioredis.Redis:
    return aioredis.from_url(settings.REDIS_URL, decode_responses=True)


@lru_cache
def get_session_store() -> SessionStore:
    return SessionStore(
        get_redis_client(),
        ttl_seconds=settings.MEMORY_SESSION_TTL_SECONDS,
    )


@lru_cache
def get_embedding_service() -> BaseEmbeddingService:
    if settings.EMBEDDING_PROVIDER == "ollama":
        return OllamaEmbeddingService(
            base_url=settings.OLLAMA_BASE_URL,
            model=settings.OLLAMA_EMBED_MODEL,
            timeout=settings.EMBEDDING_TIMEOUT,
        )
    return MockEmbeddingService()


def _build_summary_llm() -> LLMProvider | None:
    mode = settings.MEMORY_SUMMARY_MODE.lower().strip()
    if mode == "ollama":
        return OllamaProvider(
            base_url=settings.OLLAMA_BASE_URL,
            model=settings.MEMORY_SUMMARY_MODEL,
            timeout=settings.MEMORY_SUMMARY_TIMEOUT,
        )
    if mode == "sarvam" and settings.SARVAM_API_KEY:
        return SarvamProvider(
            api_key=settings.SARVAM_API_KEY,
            model=settings.SARVAM_MODEL,
            timeout=settings.MEMORY_SUMMARY_TIMEOUT,
        )
    return None


@lru_cache
def get_summarizer() -> BaseSessionSummarizer:
    llm = _build_summary_llm()
    if llm is not None:
        return LLMSessionSummarizer(llm=llm)
    return MockSessionSummarizer()


async def get_memory_service(db: AsyncSession = Depends(get_db)) -> MemoryService:
    return MemoryService(
        session_store=get_session_store(),
        long_term_repo=LongTermMemoryRepository(db),
        conversation_repo=ConversationRepository(db),
        profile_repo=ProfileRepository(db),
        embedding_service=get_embedding_service(),
        summarizer=get_summarizer(),
    )
