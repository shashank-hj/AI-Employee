from typing import Optional

from sqlalchemy import text as sa_text
from sqlalchemy.ext.asyncio import AsyncSession

from rag.models.chunk import DocumentChunkModel


class VectorStore:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def store_chunks(self, chunks: list[DocumentChunkModel]) -> list[DocumentChunkModel]:
        for chunk in chunks:
            self._session.add(chunk)
        await self._session.flush()
        return chunks

    async def search(
        self,
        embedding: list[float],
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> list[tuple[DocumentChunkModel, str, str, float]]:
        params: dict = {"embedding": embedding, "top_k": top_k}
        query = sa_text("""
            SELECT
                dc.id, dc.document_id, dc.chunk_index, dc.content, dc.metadata,
                dc.created_at, dc.updated_at,
                d.title AS document_title,
                1 - (dc.embedding <=> :embedding) AS score
            FROM document_chunks dc
            JOIN documents d ON d.id = dc.document_id
            WHERE dc.embedding IS NOT NULL
            ORDER BY dc.embedding <=> :embedding
            LIMIT :top_k
        """)

        result = await self._session.execute(query, params)
        rows = result.fetchall()

        output: list[tuple[DocumentChunkModel, str, str, float]] = []
        for row in rows:
            score = max(0.0, min(1.0, float(row.score)))
            if score >= min_score:
                chunk = DocumentChunkModel(
                    id=row.id,
                    document_id=row.document_id,
                    chunk_index=row.chunk_index,
                    content=row.content,
                    metadata_=row.metadata,
                    created_at=row.created_at,
                    updated_at=row.updated_at,
                )
                output.append((chunk, str(row.document_id), row.document_title, score))
        return output

    async def delete_by_document(self, document_id: str) -> int:
        from sqlalchemy import delete
        stmt = delete(DocumentChunkModel).where(DocumentChunkModel.document_id == document_id)
        result = await self._session.execute(stmt)
        await self._session.flush()
        return result.rowcount or 0
