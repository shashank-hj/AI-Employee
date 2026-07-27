from typing import Optional

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from memory.models.conversation import ConversationMessageModel


class ConversationRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(self, message: ConversationMessageModel) -> ConversationMessageModel:
        if message.sequence == 0:
            max_seq = await self._get_max_sequence(message.session_id)
            message.sequence = max_seq + 1
        self._session.add(message)
        await self._session.flush()
        await self._session.refresh(message)
        return message

    async def _get_max_sequence(self, session_id: str) -> int:
        stmt = select(func.coalesce(func.max(ConversationMessageModel.sequence), 0)).where(
            ConversationMessageModel.session_id == session_id
        )
        result = await self._session.execute(stmt)
        return result.scalar() or 0

    async def get_by_session(self, session_id: str) -> list[ConversationMessageModel]:
        stmt = (
            select(ConversationMessageModel)
            .where(ConversationMessageModel.session_id == session_id)
            .order_by(ConversationMessageModel.sequence.asc())
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def delete_by_session(self, session_id: str) -> int:
        from sqlalchemy import delete
        stmt = delete(ConversationMessageModel).where(ConversationMessageModel.session_id == session_id)
        result = await self._session.execute(stmt)
        await self._session.flush()
        return result.rowcount or 0
