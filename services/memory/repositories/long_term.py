from typing import Optional

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from memory.models.long_term import LongTermMemoryModel


class LongTermMemoryRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(self, memory: LongTermMemoryModel) -> LongTermMemoryModel:
        self._session.add(memory)
        await self._session.flush()
        await self._session.refresh(memory)
        return memory

    async def get_by_id(self, memory_id: str) -> Optional[LongTermMemoryModel]:
        stmt = select(LongTermMemoryModel).where(LongTermMemoryModel.id == memory_id)
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def list_by_user(
        self,
        user_id: str,
        memory_type: Optional[str] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[list[LongTermMemoryModel], int]:
        conditions = [LongTermMemoryModel.user_id == user_id]
        if memory_type:
            conditions.append(LongTermMemoryModel.memory_type == memory_type)

        base = select(LongTermMemoryModel).where(*conditions)
        count_stmt = select(func.count()).select_from(base.subquery())
        total_result = await self._session.execute(count_stmt)
        total = total_result.scalar() or 0

        list_stmt = base.order_by(LongTermMemoryModel.created_at.desc()).offset((page - 1) * page_size).limit(page_size)
        result = await self._session.execute(list_stmt)
        return list(result.scalars().all()), total

    async def search_by_embedding(
        self,
        embedding: list[float],
        user_id: Optional[str] = None,
        memory_type: Optional[str] = None,
        top_k: int = 10,
        min_score: float = 0.0,
    ) -> list[tuple[LongTermMemoryModel, float]]:
        from sqlalchemy import text as sa_text

        params: dict = {"embedding": embedding, "top_k": top_k, "min_score": min_score}
        conditions = ["1=1"]
        if user_id:
            conditions.append("user_id = :user_id")
            params["user_id"] = user_id
        if memory_type:
            conditions.append("memory_type = :memory_type")
            params["memory_type"] = memory_type

        query = sa_text(f"""
            SELECT id, user_id, content, memory_type, importance, metadata, source,
                   created_at, updated_at,
                   1 - (embedding <=> :embedding) AS score
            FROM long_term_memories
            WHERE {" AND ".join(conditions)}
            AND embedding IS NOT NULL
            ORDER BY embedding <=> :embedding
            LIMIT :top_k
        """)

        result = await self._session.execute(query, params)
        rows = result.fetchall()
        return [
            (
                LongTermMemoryModel(
                    id=row.id,
                    user_id=row.user_id,
                    content=row.content,
                    memory_type=row.memory_type,
                    importance=row.importance,
                    metadata_=row.metadata,
                    source=row.source,
                    created_at=row.created_at,
                    updated_at=row.updated_at,
                ),
                max(0.0, min(1.0, float(row.score))),
            )
            for row in rows
            if float(row.score) >= min_score
        ]

    async def delete(self, memory: LongTermMemoryModel) -> None:
        await self._session.delete(memory)
        await self._session.flush()
