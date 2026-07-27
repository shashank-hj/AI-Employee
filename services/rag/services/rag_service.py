import structlog

from rag.models.document import DocumentModel, DocumentStatus
from rag.models.chunk import DocumentChunkModel
from rag.repositories.document_repo import DocumentRepository
from rag.repositories.vector_store import VectorStore
from rag.services.pipeline import BaseEmbeddingProvider, DocumentIngester
from rag.schemas.documents import (
    DocumentUpload,
    DocumentResponse,
    QueryRequest,
    QueryResponse,
    SearchResult,
)
from shared.utils.exceptions import NotFoundException

logger = structlog.get_logger(__name__)


class Retriever:
    def __init__(self, vector_store: VectorStore, embedder: BaseEmbeddingProvider) -> None:
        self._store = vector_store
        self._embedder = embedder

    async def retrieve(self, query: str, top_k: int = 5, min_score: float = 0.0) -> list[SearchResult]:
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


class Ranker:
    def rank(self, results: list[SearchResult], query: str) -> list[SearchResult]:
        return sorted(results, key=lambda r: r.score, reverse=True)


class RAGService:
    def __init__(
        self,
        doc_repo: DocumentRepository,
        vector_store: VectorStore,
        ingester: DocumentIngester,
        embedder: BaseEmbeddingProvider,
        retriever: Retriever,
        ranker: Ranker,
    ) -> None:
        self._doc_repo = doc_repo
        self._store = vector_store
        self._ingester = ingester
        self._embedder = embedder
        self._retriever = retriever
        self._ranker = ranker

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
            embeddings = await self._embedder.embed(chunks_text)

            chunk_models: list[DocumentChunkModel] = []
            for i, (chunk_text, embedding) in enumerate(zip(chunks_text, embeddings)):
                chunk_models.append(DocumentChunkModel(
                    document_id=doc.id,
                    chunk_index=i,
                    content=chunk_text,
                    embedding=embedding,
                ))

            await self._store.store_chunks(chunk_models)
            doc = await self._doc_repo.update_status(doc, DocumentStatus.READY, len(chunk_models))
            logger.info("document_ingested", doc_id=str(doc.id), chunks_count=len(chunk_models))
        except Exception:
            await self._doc_repo.update_status(doc, DocumentStatus.FAILED, 0)
            raise

        return self._doc_to_response(doc)

    async def get_document(self, doc_id: str) -> DocumentResponse:
        doc = await self._get_or_raise(doc_id)
        return self._doc_to_response(doc)

    async def list_documents(self, page: int = 1, page_size: int = 20) -> tuple[list[DocumentResponse], int]:
        docs, total = await self._doc_repo.list_all(page, page_size)
        return [self._doc_to_response(d) for d in docs], total

    async def delete_document(self, doc_id: str) -> None:
        doc = await self._get_or_raise(doc_id)
        await self._store.delete_by_document(doc_id)
        await self._doc_repo.delete(doc)
        logger.info("document_deleted", doc_id=doc_id)

    async def query(self, request: QueryRequest) -> QueryResponse:
        results = await self._retriever.retrieve(
            query=request.query,
            top_k=request.top_k,
            min_score=request.min_score,
        )
        ranked = self._ranker.rank(results, request.query)
        logger.info("query_executed", query=request.query[:50], results_count=len(ranked))
        return QueryResponse(
            query=request.query,
            results=ranked,
            total_found=len(ranked),
        )

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
