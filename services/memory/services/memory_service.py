import uuid

import structlog

from memory.models.conversation import ConversationMessageModel
from memory.models.long_term import LongTermMemoryModel
from memory.models.profile import UserProfileModel
from memory.repositories.conversation import ConversationRepository
from memory.repositories.long_term import LongTermMemoryRepository
from memory.repositories.profile import ProfileRepository
from memory.schemas.conversation import ConversationMessageCreate, ConversationMessageResponse
from memory.schemas.long_term import LongTermMemoryCreate, LongTermMemoryResponse
from memory.schemas.profile import UserProfileCreate, UserProfileResponse, UserProfileUpdate
from memory.schemas.search import MemorySearchRequest, MemorySearchResult
from memory.schemas.session import (
    SessionCreate,
    SessionMessage,
    SessionResponse,
    SessionSummaryResponse,
)
from memory.services.stores import BaseEmbeddingService, SessionStore
from memory.services.summarizer import BaseSessionSummarizer
from shared.utils.exceptions import NotFoundException

logger = structlog.get_logger(__name__)


class MemoryService:
    def __init__(
        self,
        session_store: SessionStore,
        long_term_repo: LongTermMemoryRepository,
        conversation_repo: ConversationRepository,
        profile_repo: ProfileRepository,
        embedding_service: BaseEmbeddingService,
        summarizer: BaseSessionSummarizer | None = None,
    ) -> None:
        self._session = session_store
        self._lt_repo = long_term_repo
        self._conv_repo = conversation_repo
        self._prof_repo = profile_repo
        self._embedder = embedding_service
        self._summarizer = summarizer

    async def upsert_session(self, data: SessionCreate) -> SessionResponse:
        session_id = data.session_id or str(uuid.uuid4())
        data = data.model_copy(update={"session_id": session_id})

        message_count: int | None = None
        if data.messages:
            existing = await self._session.get(session_id)
            base = existing.message_count if existing is not None else 0
            for msg in data.messages:
                conv_msg = ConversationMessageModel(
                    session_id=session_id,
                    user_id=data.user_id,
                    role=msg.role,
                    content=msg.content,
                    sequence=0,
                )
                await self._conv_repo.create(conv_msg)
            message_count = base + len(data.messages)

        await self._session.upsert(data, message_count=message_count)
        logger.info(
            "session_upserted",
            session_id=session_id,
            message_count=message_count if message_count is not None else 0,
        )
        return await self.get_session(session_id)

    async def get_session(self, session_id: str) -> SessionResponse:
        state = await self._session.get(session_id)
        if state is None:
            raise NotFoundException(f"Session '{session_id}' not found or expired")

        messages = await self._conv_repo.get_by_session(session_id)
        return SessionResponse(
            session_id=state.session_id,
            user_id=state.user_id,
            messages=[SessionMessage(role=m.role, content=m.content) for m in messages],
            context=state.context,
            metadata=state.metadata,
            message_count=len(messages),
            created_at=state.created_at,
            updated_at=state.updated_at,
            ttl_seconds=state.ttl_seconds,
            expires_at=state.expires_at,
        )

    async def delete_session(self, session_id: str) -> None:
        deleted = await self._session.delete(session_id)
        if not deleted:
            raise NotFoundException(f"Session '{session_id}' not found")
        await self._conv_repo.delete_by_session(session_id)

    async def add_session_message(
        self,
        session_id: str,
        message: SessionMessage,
    ) -> SessionResponse:
        state = await self._session.get(session_id)
        if state is None:
            raise NotFoundException(f"Session '{session_id}' not found or expired")

        conv_msg = ConversationMessageModel(
            session_id=session_id,
            user_id=state.user_id,
            role=message.role,
            content=message.content,
            sequence=0,
        )
        await self._conv_repo.create(conv_msg)
        await self._session.add_message(session_id)
        return await self.get_session(session_id)

    async def update_session_state(
        self,
        session_id: str,
        context: dict | None = None,
        metadata: dict | None = None,
    ) -> SessionResponse:
        updated = await self._session.update_state(session_id, context=context, metadata=metadata)
        if updated is None:
            raise NotFoundException(f"Session '{session_id}' not found or expired")
        return await self.get_session(session_id)

    async def list_sessions(
        self,
        user_id: str | None = None,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[list[SessionResponse], int]:
        return await self._session.list(user_id=user_id, page=page, page_size=page_size)

    async def backfill_message_counts(self) -> dict:
        """Reconcile Redis session message_count with the real count in PostgreSQL.

        Scans every active session in Redis and overwrites its stored
        message_count with the actual number of rows in conversation_messages.
        """
        updated: list[str] = []
        errors: list[str] = []
        for session_id in await self._session.list_ids():
            try:
                real = await self._conv_repo.count_by_session(session_id)
                await self._session.set_message_count(session_id, real)
                updated.append(session_id)
            except Exception as exc:
                errors.append(session_id)
                logger.warning(
                    "backfill_message_count_failed",
                    session_id=session_id,
                    error=str(exc),
                )
        logger.info(
            "backfill_message_counts_complete",
            total=len(updated),
            errors=len(errors),
        )
        return {"updated": updated, "errors": errors, "total": len(updated)}

    async def clear_session_messages(self, session_id: str) -> SessionResponse:
        cleared = await self._session.clear_messages(session_id)
        if cleared is None:
            raise NotFoundException(f"Session '{session_id}' not found or expired")
        await self._conv_repo.delete_by_session(session_id)
        return await self.get_session(session_id)

    async def expire_session(self, session_id: str) -> None:
        deleted = await self._session.delete(session_id)
        if not deleted:
            raise NotFoundException(f"Session '{session_id}' not found")
        await self._conv_repo.delete_by_session(session_id)

    async def generate_session_summary(
        self,
        session_id: str,
        message_limit: int = 20,
        store: bool = True,
    ) -> SessionSummaryResponse:
        state = await self._session.get(session_id)
        if state is None:
            raise NotFoundException(f"Session '{session_id}' not found or expired")

        messages = await self._conv_repo.get_by_session(session_id)
        recent = messages[-message_limit:]
        transcript = "\n".join(f"{m.role}: {m.content}" for m in recent)
        summary = await self._summarizer.summarize(transcript) if self._summarizer else ""
        summary = summary.strip()

        if store and summary:
            await self._session.update_state(session_id, context={"summary": summary})
            if state.user_id:
                await self._lt_repo.create(LongTermMemoryModel(
                    user_id=state.user_id,
                    content=summary,
                    memory_type="summary",
                    importance=0.7,
                    embedding=await self._embedder.embed(summary),
                    metadata_={"session_id": session_id},
                    source="session_summary",
                ))

        logger.info(
            "session_summary_generated",
            session_id=session_id,
            message_count=len(recent),
            summary_length=len(summary),
        )
        return SessionSummaryResponse(
            session_id=session_id,
            summary=summary,
            message_count=len(recent),
        )

    async def store_long_term(self, data: LongTermMemoryCreate) -> LongTermMemoryResponse:
        embedding = await self._embedder.embed(data.content)
        memory = LongTermMemoryModel(
            user_id=data.user_id,
            content=data.content,
            memory_type=data.memory_type.value,
            importance=data.importance,
            embedding=embedding,
            metadata_=data.metadata,
            source=data.source,
        )
        created = await self._lt_repo.create(memory)
        logger.info(
            "long_term_stored",
            memory_id=str(created.id),
            memory_type=data.memory_type.value,
        )
        return self._lt_to_response(created)

    async def get_long_term(self, memory_id: str) -> LongTermMemoryResponse:
        memory = await self._lt_repo.get_by_id(memory_id)
        if memory is None:
            raise NotFoundException(f"Long-term memory '{memory_id}' not found")
        return self._lt_to_response(memory)

    async def list_long_term(
        self,
        user_id: str,
        memory_type: str | None = None,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[list[LongTermMemoryResponse], int]:
        memories, total = await self._lt_repo.list_by_user(user_id, memory_type, page, page_size)
        return [self._lt_to_response(m) for m in memories], total

    async def delete_long_term(self, memory_id: str) -> None:
        memory = await self._lt_repo.get_by_id(memory_id)
        if memory is None:
            raise NotFoundException(f"Long-term memory '{memory_id}' not found")
        await self._lt_repo.delete(memory)

    async def search_memories(self, request: MemorySearchRequest) -> list[MemorySearchResult]:
        embedding = await self._embedder.embed(request.query)
        results = await self._lt_repo.search_by_embedding(
            embedding=embedding,
            user_id=request.user_id,
            memory_type=request.memory_type,
            top_k=request.top_k,
            min_score=request.min_score,
            importance_min=request.importance_min,
            importance_max=request.importance_max,
            sort=request.sort,
        )
        logger.info("memory_search", query=request.query[:50], num_results=len(results))
        return [
            MemorySearchResult(
                id=str(m.id),
                user_id=m.user_id,
                content=m.content,
                memory_type=m.memory_type,
                importance=m.importance,
                score=round(score, 4),
                metadata=m.metadata_,
                source=m.source,
                created_at=m.created_at,
            )
            for m, score in results
        ]

    async def store_message(self, data: ConversationMessageCreate) -> ConversationMessageResponse:
        msg = ConversationMessageModel(
            session_id=data.session_id,
            user_id=data.user_id,
            role=data.role,
            content=data.content,
            sequence=data.sequence or 0,
        )
        created = await self._conv_repo.create(msg)
        try:
            await self._session.add_message(data.session_id)
        except Exception:
            pass
        return self._msg_to_response(created)

    async def get_conversation(self, session_id: str) -> list[ConversationMessageResponse]:
        messages = await self._conv_repo.get_by_session(session_id)
        return [self._msg_to_response(m) for m in messages]

    async def upsert_profile(self, data: UserProfileCreate) -> UserProfileResponse:
        profile = UserProfileModel(
            user_id=data.user_id,
            display_name=data.display_name,
            preferences=data.preferences,
            metadata_=data.metadata,
        )
        result = await self._prof_repo.upsert(profile)
        return self._prof_to_response(result)

    async def get_profile(self, user_id: str) -> UserProfileResponse:
        profile = await self._prof_repo.get_by_user_id(user_id)
        if profile is None:
            raise NotFoundException(f"Profile for user '{user_id}' not found")
        return self._prof_to_response(profile)

    async def list_profiles(
        self, page: int = 1, page_size: int = 20
    ) -> tuple[list[UserProfileResponse], int]:
        profiles, total = await self._prof_repo.list_all(page, page_size)
        return [self._prof_to_response(p) for p in profiles], total

    async def update_profile(self, user_id: str, data: UserProfileUpdate) -> UserProfileResponse:
        existing = await self._prof_repo.get_by_user_id(user_id)
        if existing is None:
            raise NotFoundException(f"Profile for user '{user_id}' not found")
        update_dict = data.model_dump(exclude_unset=True)
        for key, value in update_dict.items():
            setattr(existing, key, value)
        updated = await self._prof_repo.upsert(existing)
        return self._prof_to_response(updated)

    @staticmethod
    def _lt_to_response(m: LongTermMemoryModel) -> LongTermMemoryResponse:
        return LongTermMemoryResponse(
            id=str(m.id),
            user_id=m.user_id,
            content=m.content,
            memory_type=m.memory_type,
            importance=m.importance,
            metadata=m.metadata_,
            source=m.source,
            created_at=m.created_at,
            updated_at=m.updated_at,
        )

    @staticmethod
    def _msg_to_response(m: ConversationMessageModel) -> ConversationMessageResponse:
        return ConversationMessageResponse(
            id=str(m.id),
            session_id=m.session_id,
            user_id=m.user_id,
            role=m.role,
            content=m.content,
            sequence=m.sequence,
            created_at=m.created_at,
        )

    @staticmethod
    def _prof_to_response(p: UserProfileModel) -> UserProfileResponse:
        return UserProfileResponse(
            id=str(p.id),
            user_id=p.user_id,
            display_name=p.display_name,
            preferences=p.preferences or {},
            metadata=p.metadata_,
            created_at=p.created_at,
            updated_at=p.updated_at,
        )
