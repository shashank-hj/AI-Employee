from functools import lru_cache

import redis.asyncio as aioredis
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from memory.config import settings
from memory.database.session import get_db
from memory.repositories.long_term import LongTermMemoryRepository
from memory.repositories.conversation import ConversationRepository
from memory.repositories.profile import ProfileRepository
from memory.services.stores import SessionStore, MockEmbeddingService, OllamaEmbeddingService, BaseEmbeddingService
from memory.services.memory_service import MemoryService


@lru_cache()
def get_redis_client() -> aioredis.Redis:
    return aioredis.from_url(settings.REDIS_URL, decode_responses=True)


@lru_cache()
def get_session_store() -> SessionStore:
    return SessionStore(get_redis_client())


@lru_cache()
def get_embedding_service() -> BaseEmbeddingService:
    if settings.EMBEDDING_PROVIDER == "ollama":
        return OllamaEmbeddingService(
            base_url=settings.OLLAMA_BASE_URL,
            model=settings.OLLAMA_EMBED_MODEL,
            timeout=settings.EMBEDDING_TIMEOUT,
        )
    return MockEmbeddingService()


async def get_memory_service(db: AsyncSession = Depends(get_db)) -> MemoryService:
    return MemoryService(
        session_store=get_session_store(),
        long_term_repo=LongTermMemoryRepository(db),
        conversation_repo=ConversationRepository(db),
        profile_repo=ProfileRepository(db),
        embedding_service=get_embedding_service(),
    )
