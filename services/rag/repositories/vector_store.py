
import re

from sqlalchemy import text as sa_text
from sqlalchemy.ext.asyncio import AsyncSession

from rag.models.chunk import DocumentChunkModel

# Words that carry no lexical signal; they are dropped when building the
# full-text tsquery so that a query like "Explain Peninsular Plateau" does not
# require every word (incl. "explain") to appear verbatim in a chunk.
_LEXICAL_STOPWORDS = {
    "the", "and", "for", "are", "was", "were", "with", "that", "this", "what",
    "which", "how", "why", "when", "where", "who", "is", "in", "on",
    "to", "of", "a", "an", "explain", "describe", "tell", "about",
    "give", "does", "do", "can", "you", "me", "it", "be", "has", "have",
}


def _lexical_tsquery(query: str) -> str | None:
    """Build an OR-combined tsquery from the meaningful terms of ``query``.

    Returns ``None`` when there are no meaningful terms (so the caller can skip
    the search entirely). Terms are matched with OR semantics to keep lexical
    recall high (it complements vector search, which handles precision).
    """
    terms = [
        t for t in re.findall(r"[a-zA-Z0-9]+", (query or "").lower())
        if t not in _LEXICAL_STOPWORDS and len(t) > 1
    ]
    if not terms:
        return None
    return " | ".join(terms)


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
            # Cosine similarity (1 - distance) is often slightly negative for
            # unrelated text with nomic-embed. Only apply a real threshold when
            # the caller asks for one (min_score > 0); the default 0.0 means
            # "keep everything, ordered by similarity".
            if min_score <= 0.0 or item[3] >= min_score:
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
        embedding similarity is weak. Terms are OR-combined (after dropping
        stopwords) so a broad query still matches chunks that contain any of
        its meaningful words.
        """
        tsquery = _lexical_tsquery(query)
        if not tsquery:
            return []

        params: dict = {"query": tsquery, "top_k": top_k}
        sql = sa_text("""
            SELECT
                dc.id, dc.document_id, dc.chunk_index, dc.content, dc.metadata,
                dc.created_at, dc.updated_at,
                d.title AS document_title,
                ts_rank(to_tsvector('english', dc.content),
                        to_tsquery('english', :query)) AS score
            FROM document_chunks dc
            JOIN documents d ON d.id = dc.document_id
            WHERE to_tsvector('english', dc.content)
                @@ to_tsquery('english', :query)
            ORDER BY score DESC
            LIMIT :top_k
        """)
        result = await self._session.execute(sql, params)
        rows = result.fetchall()

        output: list[tuple[DocumentChunkModel, str, str, float]] = []
        for row in rows:
            item = self._row_to_result(row)
            if min_score <= 0.0 or item[3] >= min_score:
                output.append(item)
        return output

    async def delete_by_document(self, document_id: str) -> int:
        from sqlalchemy import delete
        stmt = delete(DocumentChunkModel).where(DocumentChunkModel.document_id == document_id)
        result = await self._session.execute(stmt)
        await self._session.flush()
        return result.rowcount or 0
