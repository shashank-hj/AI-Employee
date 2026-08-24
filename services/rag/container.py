import os
from functools import lru_cache

import structlog
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from rag.config import settings
from rag.database.session import get_db
from rag.repositories.document_repo import DocumentRepository
from rag.repositories.vector_store import VectorStore
from rag.services.answer_generator import AnswerGenerator
from rag.services.pipeline import (
    BaseEmbeddingProvider,
    DocumentIngester,
    MockEmbeddingProvider,
    TextChunker,
)
from rag.services.query_refiner import QueryRefiner
from rag.services.rag_service import HybridRetriever, RAGService, Ranker, Retriever
from rag.services.translator import NoopQueryTranslator, SpeechQueryTranslator
from shared.llm.base import LLMProvider
from shared.llm.embedding_provider import OllamaEmbeddingProvider
from shared.llm.ollama_provider import OllamaProvider
from shared.llm.opencode_provider import OpencodeProvider
from shared.llm.sarvam_provider import SarvamProvider
from shared.usage import UsageRecorder

logger = structlog.get_logger(__name__)


@lru_cache
def get_embedding_provider() -> BaseEmbeddingProvider:
    if settings.EMBEDDING_PROVIDER == "ollama":
        return OllamaEmbeddingProvider(
            base_url=settings.OLLAMA_BASE_URL,
            model=settings.OLLAMA_EMBED_MODEL,
            timeout=settings.EMBEDDING_TIMEOUT,
            batch_size=settings.EMBEDDING_BATCH_SIZE,
        )
    return MockEmbeddingProvider()


@lru_cache
def get_chunker() -> TextChunker:
    return TextChunker(chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP)


@lru_cache
def get_ingester() -> DocumentIngester:
    return DocumentIngester(get_chunker())


@lru_cache
def get_usage_recorder() -> UsageRecorder:
    """Recorder that persists embedding usage rows into the shared usage_events table."""
    if settings.USAGE_PRICING:
        os.environ["USAGE_PRICING"] = settings.USAGE_PRICING
    from rag.database.session import async_session

    return UsageRecorder(
        session_factory=async_session,
        service="rag",
        enabled=settings.USAGE_ENABLED,
    )


@lru_cache
def get_query_refiner() -> QueryRefiner | None:
    """Build an LLM-backed query refiner, or ``None`` when disabled."""
    if not settings.RAG_REFINE_ENABLED:
        return None

    provider_name = settings.RAG_REFINE_LLM.lower().strip()
    llm: LLMProvider
    if provider_name == "ollama":
        llm = OllamaProvider(
            base_url=settings.OLLAMA_BASE_URL,
            model=settings.RAG_REFINE_MODEL,
            timeout=settings.RAG_REFINE_TIMEOUT,
        )
    elif provider_name == "sarvam" and settings.RAG_TRANSLATE_API_KEY:
        llm = SarvamProvider(
            api_key=settings.RAG_TRANSLATE_API_KEY,
            model=settings.RAG_REFINE_MODEL,
            timeout=settings.RAG_REFINE_TIMEOUT,
        )
    else:
        return None
    return QueryRefiner(llm_provider=llm)


@lru_cache
def get_query_translator():
    """Build a query translator backed by the speech service, or a no-op."""
    if not settings.RAG_TRANSLATE_URL:
        return NoopQueryTranslator()
    return SpeechQueryTranslator(
        base_url=settings.RAG_TRANSLATE_URL,
        api_key=settings.RAG_TRANSLATE_API_KEY,
        timeout=settings.RAG_TRANSLATE_TIMEOUT,
    )


@lru_cache
def get_answer_generator() -> AnswerGenerator | None:
    """Build the LLM-backed answer generator, or ``None`` when disabled."""
    if not settings.RAG_ANSWER_ENABLED:
        return None

    provider_name = settings.RAG_ANSWER_LLM.lower().strip()
    if provider_name != "opencode":
        logger.warning(
            "answer_llm_unsupported",
            provider=provider_name,
            message="Only 'opencode' is supported for answer generation; disabled",
        )
        return None

    llm = OpencodeProvider(
        base_url=settings.OPENCODE_BASE_URL,
        model=settings.OPENCODE_MODEL,
        agent=settings.OPENCODE_AGENT,
        password=settings.OPENCODE_PASSWORD or None,
        username=settings.OPENCODE_USERNAME or "opencode",
        timeout=settings.RAG_ANSWER_TIMEOUT,
        max_tokens=settings.RAG_ANSWER_MAX_TOKENS,
        temperature=0.1,
    )
    return AnswerGenerator(llm=llm, usage_recorder=get_usage_recorder())


async def get_rag_service(db: AsyncSession = Depends(get_db)) -> RAGService:
    embedder = get_embedding_provider()
    doc_repo = DocumentRepository(db)
    vector_store = VectorStore(db)
    ingester = get_ingester()
    retriever = HybridRetriever(
        vector_retriever=Retriever(vector_store, embedder),
        vector_store=vector_store,
    )
    ranker = Ranker()

    return RAGService(
        doc_repo=doc_repo,
        vector_store=vector_store,
        ingester=ingester,
        embedder=embedder,
        retriever=retriever,
        ranker=ranker,
        usage_recorder=get_usage_recorder(),
        query_refiner=get_query_refiner(),
        query_translator=get_query_translator(),
        answer_generator=get_answer_generator(),
        answer_top_k=settings.RAG_ANSWER_TOP_K,
    )
