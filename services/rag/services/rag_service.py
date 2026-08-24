import re

import structlog

from rag.models.chunk import DocumentChunkModel
from rag.models.document import DocumentModel, DocumentStatus
from rag.repositories.document_repo import DocumentRepository
from rag.repositories.vector_store import VectorStore
from rag.schemas.documents import (
    Citation,
    DocumentResponse,
    DocumentUpload,
    QueryRequest,
    QueryResponse,
    SearchResult,
)
from rag.services.answer_generator import AnswerGenerator
from rag.services.pipeline import BaseEmbeddingProvider, DocumentIngester
from rag.services.query_refiner import QueryRefiner
from rag.services.translator import QueryTranslator
from shared.usage.pricing import estimate_tokens
from shared.usage.recorder import UsageRecorder
from shared.usage.records import UsageRecord
from shared.utils.exceptions import NotFoundException

logger = structlog.get_logger(__name__)


class Retriever:
    def __init__(self, vector_store: VectorStore, embedder: BaseEmbeddingProvider) -> None:
        self._store = vector_store
        self._embedder = embedder

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> list[SearchResult]:
        embeddings = await self._embedder.embed([query])
        rows = await self._store.search(embeddings[0], top_k=top_k, min_score=min_score)

        results: list[SearchResult] = []
        for chunk, doc_id, doc_title, score in rows:
            results.append(SearchResult(
                chunk_id=str(chunk.id),
                document_id=doc_id,
                document_title=doc_title,
                chunk_index=chunk.chunk_index,
                content=chunk.content,
                score=round(score, 4),
                metadata=chunk.metadata_,
            ))
        return results


RRF_CONSTANT = 60.0
RRF_SCORE_SCALE = 20.0


class HybridRetriever:
    """Fuses vector and lexical retrieval with Reciprocal Rank Fusion (RRF).

    Both sources are pooled (a superset of ``top_k``), ranked lists are merged by
    reciprocal rank, and the top fused candidates are returned with a normalized
    0..1 confidence score.
    """

    def __init__(self, vector_retriever: Retriever, vector_store: VectorStore) -> None:
        self._vector = vector_retriever
        self._store = vector_store

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> list[SearchResult]:
        pool_size = max(top_k * 3, 10)
        vector = await self._vector.retrieve(query, top_k=pool_size, min_score=min_score)
        lexical_rows = await self._store.search_lexical(query, top_k=pool_size, min_score=min_score)
        lexical = [
            SearchResult(
                chunk_id=str(chunk.id),
                document_id=doc_id,
                document_title=doc_title,
                chunk_index=chunk.chunk_index,
                content=chunk.content,
                score=round(score, 4),
                metadata=chunk.metadata_,
            )
            for chunk, doc_id, doc_title, score in lexical_rows
        ]

        fused: dict[str, dict] = {}
        for rank, result in enumerate(vector):
            entry = fused.setdefault(result.chunk_id, {"result": result, "rrf": 0.0})
            entry["rrf"] += 1.0 / (RRF_CONSTANT + rank + 1)
        for rank, result in enumerate(lexical):
            entry = fused.setdefault(result.chunk_id, {"result": result, "rrf": 0.0})
            entry["rrf"] += 1.0 / (RRF_CONSTANT + rank + 1)

        ranked = sorted(fused.values(), key=lambda e: e["rrf"], reverse=True)
        output: list[SearchResult] = []
        for entry in ranked[:top_k]:
            result: SearchResult = entry["result"]
            fused_score = min(1.0, entry["rrf"] * RRF_SCORE_SCALE)
            output.append(result.model_copy(update={"score": round(fused_score, 4)}))
        return output


_STOPWORDS = {
    "the", "and", "for", "are", "was", "with", "that", "this", "what", "which",
    "how", "why", "when", "where", "who", "is", "in", "on", "to", "of", "a", "an",
}


class Ranker:
    """Reranks fused results with a lightweight query-term overlap boost.

    The boost is deterministic and offline (no cross-encoder); it nudges chunks
    whose title/content contain several query terms above pure RRF ordering.
    """

    def __init__(self, keyword_boost: float = 0.08) -> None:
        self._keyword_boost = keyword_boost

    def rank(self, results: list[SearchResult], query: str) -> list[SearchResult]:
        terms = {t for t in re.findall(r"\w{3,}", (query or "").lower()) if t not in _STOPWORDS}
        if not terms:
            return sorted(results, key=lambda r: r.score, reverse=True)

        def boosted(r: SearchResult) -> SearchResult:
            text = (r.document_title + " " + r.content).lower()
            hits = sum(1 for term in terms if term in text)
            if hits == 0:
                return r
            new_score = min(1.0, r.score + self._keyword_boost * hits)
            return r.model_copy(update={"score": round(new_score, 4)})

        ranked = [boosted(r) for r in results]
        return sorted(ranked, key=lambda r: r.score, reverse=True)


class RAGService:
    def __init__(
        self,
        doc_repo: DocumentRepository,
        vector_store: VectorStore,
        ingester: DocumentIngester,
        embedder: BaseEmbeddingProvider,
        retriever: Retriever,
        ranker: Ranker,
        usage_recorder: UsageRecorder | None = None,
        query_refiner: QueryRefiner | None = None,
        query_translator: QueryTranslator | None = None,
        answer_generator: AnswerGenerator | None = None,
        answer_top_k: int = 5,
    ) -> None:
        self._doc_repo = doc_repo
        self._store = vector_store
        self._ingester = ingester
        self._embedder = embedder
        self._retriever = retriever
        self._ranker = ranker
        self._usage = usage_recorder
        self._refiner = query_refiner
        self._translator = query_translator
        self._answer_gen = answer_generator
        self._answer_top_k = answer_top_k

    @property
    def _embed_model(self) -> str:
        return getattr(self._embedder, "_model", "embedding")

    async def _record_embed(
        self,
        *,
        operation: str,
        text: str,
        status: str = "success",
        error: str | None = None,
    ) -> None:
        if self._usage is None:
            return
        try:
            await self._usage.record(UsageRecord(
                service="rag",
                category="embedding",
                operation=operation,
                model=self._embed_model,
                unit="tokens",
                input_units=estimate_tokens(text),
                status=status,
                error=error,
            ))
        except Exception as exc:  # pragma: no cover - usage must never break the caller
            logger.warning("embed_usage_record_failed", operation=operation, error=str(exc))

    async def ingest_document(self, data: DocumentUpload) -> DocumentResponse:
        doc = DocumentModel(
            title=data.title,
            content=data.content,
            source=data.source,
            content_type=data.content_type,
            status=DocumentStatus.PROCESSING,
            metadata_=data.metadata,
        )
        doc = await self._doc_repo.create(doc)
        logger.info("document_created", doc_id=str(doc.id), title=data.title)

        try:
            chunks_text = self._ingester.ingest(data.content)
            if not chunks_text:
                raise ValueError("No text was extracted from the document")

            embeddings = await self._embedder.embed(chunks_text)
            await self._record_embed(
                operation="embed_docs",
                text=data.content,
            )

            if len(embeddings) != len(chunks_text):
                raise ValueError(
                    f"Embedding provider returned {len(embeddings)} vectors "
                    f"for {len(chunks_text)} chunks; document ingestion aborted"
                )

            if all(not any(vec) for vec in embeddings):
                raise ValueError(
                    "All embeddings are zero vectors (embedding provider failed); "
                    "document ingestion aborted instead of storing garbage"
                )

            chunk_models: list[DocumentChunkModel] = []
            for i, (chunk_text, embedding) in enumerate(
                zip(chunks_text, embeddings, strict=True)
            ):
                chunk_models.append(DocumentChunkModel(
                    document_id=doc.id,
                    chunk_index=i,
                    content=chunk_text,
                    embedding=embedding,
                ))

            await self._store.store_chunks(chunk_models)
            doc = await self._doc_repo.update_status(doc, DocumentStatus.READY, len(chunk_models))
            logger.info("document_ingested", doc_id=str(doc.id), chunks_count=len(chunk_models))
        except Exception as exc:
            logger.error("document_ingestion_failed", doc_id=str(doc.id), error=str(exc))
            try:
                await self._doc_repo.update_status(doc, DocumentStatus.FAILED, 0)
            except Exception as status_exc:  # pragma: no cover
                logger.error(
                    "document_status_update_failed",
                    doc_id=str(doc.id),
                    error=str(status_exc),
                )

        return self._doc_to_response(doc)

    async def get_document(self, doc_id: str) -> DocumentResponse:
        doc = await self._get_or_raise(doc_id)
        return self._doc_to_response(doc)

    async def list_documents(
        self,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[list[DocumentResponse], int]:
        docs, total = await self._doc_repo.list_all(page, page_size)
        return [self._doc_to_response(d) for d in docs], total

    async def delete_document(self, doc_id: str) -> None:
        doc = await self._get_or_raise(doc_id)
        await self._store.delete_by_document(doc_id)
        await self._doc_repo.delete(doc)
        logger.info("document_deleted", doc_id=doc_id)

    async def query(self, request: QueryRequest) -> QueryResponse:
        search_query = request.query
        refined_query: str | None = None
        translated_query: str | None = None

        # ── Agentic refinement ──
        if self._refiner is not None:
            refined_query = await self._refiner.refine(request.query)
            search_query = refined_query or request.query

        # ── Cross-lingual retrieval ──
        if request.language and request.language != "en-IN" and self._translator is not None:
            translated = await self._translator.translate(
                search_query, target_language_code="en-IN"
            )
            if translated and translated != search_query:
                translated_query = translated
                search_query = translated

        try:
            results = await self._retriever.retrieve(
                query=search_query,
                top_k=request.top_k,
                min_score=request.min_score,
            )
        except Exception as exc:
            await self._record_embed(
                operation="embed_query",
                text=request.query,
                status="error",
                error=str(exc)[:500],
            )
            raise
        await self._record_embed(operation="embed_query", text=request.query)
        ranked = self._ranker.rank(results, request.query)

        # ── Answer generation (opencode synthesizes a sourced answer) ──
        answer: str | None = None
        sources: list = []
        if self._answer_gen is not None and ranked:
            try:
                answer_result = await self._answer_gen.generate(
                    request.query,
                    ranked[: self._answer_top_k],
                )
                answer = answer_result.answer
                sources = answer_result.sources
            except Exception as exc:
                logger.error("answer_generation_failed", query=request.query[:50], error=str(exc))

        logger.info(
            "query_executed",
            query=request.query[:50],
            search_query=search_query[:50],
            refined=bool(refined_query),
            translated=bool(translated_query),
            results_count=len(ranked),
            answer=bool(answer),
        )
        return QueryResponse(
            query=request.query,
            results=ranked,
            total_found=len(ranked),
            citations=self._build_citations(ranked, limit=3),
            answer=answer,
            sources=sources,
            refined_query=refined_query,
            translated_query=translated_query,
            language=request.language,
        )

    @staticmethod
    def _build_citations(results: list[SearchResult], limit: int = 3) -> list[Citation]:
        citations: list[Citation] = []
        for result in results[:limit]:
            snippet = " ".join(result.content.split())[:200]
            citations.append(Citation(
                document_id=result.document_id,
                document_title=result.document_title,
                chunk_index=result.chunk_index,
                content=snippet,
                score=result.score,
            ))
        return citations

    async def _get_or_raise(self, doc_id: str) -> DocumentModel:
        doc = await self._doc_repo.get_by_id(doc_id)
        if doc is None:
            raise NotFoundException(f"Document '{doc_id}' not found")
        return doc

    @staticmethod
    def _doc_to_response(d: DocumentModel) -> DocumentResponse:
        return DocumentResponse(
            id=str(d.id),
            title=d.title,
            source=d.source,
            content_type=d.content_type,
            status=d.status,
            chunks_count=d.chunks_count,
            metadata=d.metadata_,
            created_at=d.created_at,
            updated_at=d.updated_at,
        )
