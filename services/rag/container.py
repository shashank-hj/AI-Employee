from functools import lru_cache

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from rag.config import settings
from rag.database.session import get_db
from rag.repositories.document_repo import DocumentRepository
from rag.repositories.vector_store import VectorStore
from rag.services.pipeline import (
    BaseEmbeddingProvider,
    DocumentIngester,
    MockEmbeddingProvider,
    TextChunker,
)
from rag.services.rag_service import RAGService, Retriever, Ranker


@lru_cache()
def get_embedding_provider() -> BaseEmbeddingProvider:
    return MockEmbeddingProvider()


@lru_cache()
def get_chunker() -> TextChunker:
    return TextChunker(chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP)


@lru_cache()
def get_ingester() -> DocumentIngester:
    return DocumentIngester(get_chunker())


async def get_rag_service(db: AsyncSession = Depends(get_db)) -> RAGService:
    embedder = get_embedding_provider()
    doc_repo = DocumentRepository(db)
    vector_store = VectorStore(db)
    ingester = get_ingester()
    retriever = Retriever(vector_store, embedder)
    ranker = Ranker()

    return RAGService(
        doc_repo=doc_repo,
        vector_store=vector_store,
        ingester=ingester,
        embedder=embedder,
        retriever=retriever,
        ranker=ranker,
    )
