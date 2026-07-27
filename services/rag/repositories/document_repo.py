from typing import Optional

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from rag.models.document import DocumentModel, DocumentStatus


class DocumentRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(self, doc: DocumentModel) -> DocumentModel:
        self._session.add(doc)
        await self._session.flush()
        await self._session.refresh(doc)
        return doc

    async def update_status(self, doc: DocumentModel, status: str, chunks_count: int = 0) -> DocumentModel:
        doc.status = status
        doc.chunks_count = chunks_count
        await self._session.flush()
        await self._session.refresh(doc)
        return doc

    async def get_by_id(self, doc_id: str) -> Optional[DocumentModel]:
        stmt = select(DocumentModel).where(DocumentModel.id == doc_id)
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def list_all(self, page: int = 1, page_size: int = 20) -> tuple[list[DocumentModel], int]:
        base = select(DocumentModel)
        count_stmt = select(func.count()).select_from(base.subquery())
        total_result = await self._session.execute(count_stmt)
        total = total_result.scalar() or 0

        list_stmt = base.order_by(DocumentModel.created_at.desc()).offset((page - 1) * page_size).limit(page_size)
        result = await self._session.execute(list_stmt)
        return list(result.scalars().all()), total

    async def delete(self, doc: DocumentModel) -> None:
        await self._session.delete(doc)
        await self._session.flush()
