import structlog

from memory.models.long_term import LongTermMemoryModel
from memory.models.conversation import ConversationMessageModel
from memory.models.profile import UserProfileModel
from memory.repositories.long_term import LongTermMemoryRepository
from memory.repositories.conversation import ConversationRepository
from memory.repositories.profile import ProfileRepository
from memory.services.stores import SessionStore, BaseEmbeddingService
from memory.schemas.session import SessionCreate, SessionResponse, SessionMessage
from memory.schemas.long_term import LongTermMemoryCreate, LongTermMemoryResponse
from memory.schemas.conversation import ConversationMessageCreate, ConversationMessageResponse
from memory.schemas.profile import UserProfileCreate, UserProfileUpdate, UserProfileResponse
from memory.schemas.search import MemorySearchRequest, MemorySearchResult
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
    ) -> None:
        self._session = session_store
        self._lt_repo = long_term_repo
        self._conv_repo = conversation_repo
        self._prof_repo = profile_repo
        self._embedder = embedding_service

    async def upsert_session(self, data: SessionCreate) -> SessionResponse:
        response = await self._session.upsert(data)
        logger.info("session_upserted", session_id=response.session_id, message_count=response.message_count)
        return response

    async def get_session(self, session_id: str) -> SessionResponse:
        session = await self._session.get(session_id)
        if session is None:
            raise NotFoundException(f"Session '{session_id}' not found or expired")
        return session

    async def delete_session(self, session_id: str) -> None:
        deleted = await self._session.delete(session_id)
        if not deleted:
            raise NotFoundException(f"Session '{session_id}' not found")

    async def add_session_message(self, session_id: str, message: SessionMessage) -> SessionResponse:
        session = await self._session.add_message(session_id, message)
        if session is None:
            raise NotFoundException(f"Session '{session_id}' not found or expired")

        conv_msg = ConversationMessageModel(
            session_id=session_id,
            user_id=session.user_id,
            role=message.role,
            content=message.content,
            sequence=session.message_count,
        )
        await self._conv_repo.create(conv_msg)
        return session

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
        logger.info("long_term_stored", memory_id=str(created.id), memory_type=data.memory_type.value)
        return self._lt_to_response(created)

    async def get_long_term(self, memory_id: str) -> LongTermMemoryResponse:
        memory = await self._lt_repo.get_by_id(memory_id)
        if memory is None:
            raise NotFoundException(f"Long-term memory '{memory_id}' not found")
        return self._lt_to_response(memory)

    async def list_long_term(self, user_id: str, memory_type: str | None = None, page: int = 1, page_size: int = 20) -> tuple[list[LongTermMemoryResponse], int]:
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
