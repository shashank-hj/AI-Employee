
from sqlalchemy import func, select
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

    async def get_by_id(self, memory_id: str) -> LongTermMemoryModel | None:
        stmt = select(LongTermMemoryModel).where(LongTermMemoryModel.id == memory_id)
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def list_by_user(
        self,
        user_id: str,
        memory_type: str | None = None,
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

        list_stmt = (
            base.order_by(LongTermMemoryModel.created_at.desc())
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
        result = await self._session.execute(list_stmt)
        return list(result.scalars().all()), total

    async def search_by_embedding(
        self,
        embedding: list[float],
        user_id: str | None = None,
        memory_type: str | None = None,
        top_k: int = 10,
        min_score: float = 0.0,
        importance_min: float | None = None,
        importance_max: float | None = None,
        sort: str = "score",
    ) -> list[tuple[LongTermMemoryModel, float]]:
        from sqlalchemy import text as sa_text

        vector_literal = "[" + ",".join(str(float(x)) for x in embedding) + "]"
        params: dict = {"embedding": vector_literal, "top_k": top_k, "min_score": min_score}
        conditions = ["1=1"]
        if user_id:
            conditions.append("user_id = :user_id")
            params["user_id"] = user_id
        if memory_type:
            conditions.append("memory_type = :memory_type")
            params["memory_type"] = memory_type
        if importance_min is not None:
            conditions.append("importance >= :importance_min")
            params["importance_min"] = importance_min
        if importance_max is not None:
            conditions.append("importance <= :importance_max")
            params["importance_max"] = importance_max

        order_by = {
            "score": "embedding <=> :embedding",
            "importance": "importance DESC, embedding <=> :embedding",
            "created_at": "created_at DESC, embedding <=> :embedding",
        }.get(sort, "embedding <=> :embedding")

        query = sa_text(f"""
            SELECT id, user_id, content, memory_type, importance, metadata, source,
                   created_at, updated_at,
                   1 - (embedding <=> :embedding) AS score
            FROM long_term_memories
            WHERE {" AND ".join(conditions)}
            AND embedding IS NOT NULL
            ORDER BY {order_by}
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
