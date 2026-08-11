
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

    @staticmethod
    def _format_vector(embedding: list[float]) -> str:
        return "[" + ",".join(str(v) for v in embedding) + "]"

    @staticmethod
    def _row_to_result(row) -> tuple[DocumentChunkModel, str, str, float]:
        score = max(0.0, min(1.0, float(row.score)))
        chunk = DocumentChunkModel(
            id=row.id,
            document_id=row.document_id,
            chunk_index=row.chunk_index,
            content=row.content,
            metadata_=row.metadata,
            created_at=row.created_at,
            updated_at=row.updated_at,
        )
        return (chunk, str(row.document_id), row.document_title, score)

    async def search(
        self,
        embedding: list[float],
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> list[tuple[DocumentChunkModel, str, str, float]]:
        vec_str = self._format_vector(embedding)
        params: dict = {"embedding": vec_str, "top_k": top_k}
        query = sa_text("""
            SELECT
                dc.id, dc.document_id, dc.chunk_index, dc.content, dc.metadata,
                dc.created_at, dc.updated_at,
                d.title AS document_title,
                1 - (dc.embedding <=> CAST(:embedding AS vector)) AS score
            FROM document_chunks dc
            JOIN documents d ON d.id = dc.document_id
            WHERE dc.embedding IS NOT NULL
            ORDER BY dc.embedding <=> CAST(:embedding AS vector)
            LIMIT :top_k
        """)

        result = await self._session.execute(query, params)
        rows = result.fetchall()

        output: list[tuple[DocumentChunkModel, str, str, float]] = []
        for row in rows:
            item = self._row_to_result(row)
            if item[3] >= min_score:
                output.append(item)
        return output

    async def search_lexical(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> list[tuple[DocumentChunkModel, str, str, float]]:
        """Full-text (lexical) retrieval via Postgres ``tsvector``.

        Complements vector search: exact term matches surface even when the
        embedding similarity is weak. ``websearch_to_tsquery`` parses the query
        robustly; malformed queries fall back to ``plainto_tsquery``.
        """
        params: dict = {"query": query, "top_k": top_k}
        base = """
            SELECT
                dc.id, dc.document_id, dc.chunk_index, dc.content, dc.metadata,
                dc.created_at, dc.updated_at,
                d.title AS document_title,
                ts_rank(to_tsvector('english', dc.content), {tsquery}) AS score
            FROM document_chunks dc
            JOIN documents d ON d.id = dc.document_id
            WHERE to_tsvector('english', dc.content) @@ {tsquery}
            ORDER BY score DESC
            LIMIT :top_k
        """
        try:
            sql = sa_text(base.format(tsquery="websearch_to_tsquery('english', :query)"))
            result = await self._session.execute(sql, params)
        except Exception:
            sql = sa_text(base.format(tsquery="plainto_tsquery('english', :query)"))
            result = await self._session.execute(sql, params)

        rows = result.fetchall()
        output: list[tuple[DocumentChunkModel, str, str, float]] = []
        for row in rows:
            item = self._row_to_result(row)
            if item[3] >= min_score:
                output.append(item)
        return output

    async def delete_by_document(self, document_id: str) -> int:
        from sqlalchemy import delete
        stmt = delete(DocumentChunkModel).where(DocumentChunkModel.document_id == document_id)
        result = await self._session.execute(stmt)
        await self._session.flush()
        return result.rowcount or 0
