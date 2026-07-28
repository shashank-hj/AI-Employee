from functools import lru_cache

import structlog

from shared.llm import LLMProvider, SarvamProvider, OllamaProvider

from orchestrator.tools.registry import ToolRegistry
from orchestrator.tools.mock_tools import register_mock_tools
from orchestrator.tools.rag_client import RAGClient, MockRAGClient, HttpRAGClient
from orchestrator.planner.mock_planner import MockPlanner
from orchestrator.planner.llm_planner import LLMPlanner
from orchestrator.planner.base import BasePlanner
from orchestrator.context.builder import MockContextBuilder, ContextBuilder
from orchestrator.services.agent_service import AgentService
from orchestrator.services.mock_services import MockOrderService, MockCalendarService, MockPricingService
from orchestrator.config import settings

logger = structlog.get_logger(__name__)


@lru_cache()
def _build_rag_client() -> RAGClient:
    return HttpRAGClient(
        base_url=settings.RAG_URL,
        query_path=settings.RAG_QUERY_PATH,
        health_path=settings.RAG_HEALTH_PATH,
        timeout=settings.RAG_TIMEOUT,
        fallback_client=MockRAGClient(),
    )


@lru_cache()
def get_tool_registry() -> ToolRegistry:
    registry = ToolRegistry()
    register_mock_tools(
        registry,
        rag_client=_build_rag_client(),
        order_service=get_order_service(),
        calendar_service=get_calendar_service(),
        pricing_service=get_pricing_service(),
    )
    return registry


def _resolve_provider() -> LLMProvider | None:
    provider = settings.LLM_PROVIDER.lower().strip()

    if provider == "ollama":
        logger.info("llm_provider_selected", provider="OllamaProvider", model=settings.OLLAMA_MODEL)
        return OllamaProvider(
            base_url=settings.OLLAMA_BASE_URL,
            model=settings.OLLAMA_MODEL,
            timeout=settings.OLLAMA_TIMEOUT,
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS,
        )

    if provider == "sarvam":
        if not settings.SARVAM_API_KEY:
            logger.warning("sarvam_api_key_not_set", message="Falling back to MockPlanner")
            return None
        logger.info("llm_provider_selected", provider="SarvamProvider", model=settings.SARVAM_MODEL)
        return SarvamProvider(
            api_key=settings.SARVAM_API_KEY,
            base_url=settings.SARVAM_BASE_URL,
            model=settings.SARVAM_MODEL,
            timeout=settings.SARVAM_TIMEOUT,
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS,
        )

    if settings.SARVAM_API_KEY:
        logger.info("llm_provider_selected", provider="SarvamProvider", model=settings.SARVAM_MODEL)
        return SarvamProvider(
            api_key=settings.SARVAM_API_KEY,
            base_url=settings.SARVAM_BASE_URL,
            model=settings.SARVAM_MODEL,
            timeout=settings.SARVAM_TIMEOUT,
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS,
        )

    logger.warning("llm_provider_not_configured", message="Set LLM_PROVIDER=ollama or LLM_PROVIDER=sarvam; using MockPlanner")
    return None


@lru_cache()
def _build_classify_llm() -> LLMProvider | None:
    provider = _resolve_provider()
    if provider is None:
        return None
    if isinstance(provider, OllamaProvider):
        return OllamaProvider(
            base_url=settings.OLLAMA_BASE_URL,
            model=settings.OLLAMA_MODEL,
            timeout=settings.OLLAMA_TIMEOUT,
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_CLASSIFY_MAX_TOKENS,
        )
    return SarvamProvider(
        api_key=settings.SARVAM_API_KEY,
        base_url=settings.SARVAM_BASE_URL,
        model=settings.SARVAM_MODEL,
        timeout=settings.SARVAM_TIMEOUT,
        temperature=settings.LLM_TEMPERATURE,
        max_tokens=settings.LLM_CLASSIFY_MAX_TOKENS,
    )


@lru_cache()
def get_planner() -> BasePlanner:
    llm = _build_classify_llm()
    if llm is not None:
        model = settings.SARVAM_MODEL if settings.LLM_PROVIDER.lower() in ("sarvam", "") else settings.OLLAMA_MODEL
        logger.info("planner_initialized", provider=type(llm).__name__, model=model)
        return LLMPlanner(llm_provider=llm, fallback_intent=settings.LLM_FALLBACK_INTENT)
    logger.info("planner_initialized", provider="MockPlanner")
    return MockPlanner()


@lru_cache()
def get_context_builder() -> ContextBuilder:
    return MockContextBuilder()


@lru_cache()
def get_order_service() -> MockOrderService:
    return MockOrderService()


@lru_cache()
def get_calendar_service() -> MockCalendarService:
    return MockCalendarService()


@lru_cache()
def get_pricing_service() -> MockPricingService:
    return MockPricingService()


@lru_cache()
def _build_generate_llm() -> LLMProvider | None:
    provider = _resolve_provider()
    if provider is None:
        return None
    if isinstance(provider, OllamaProvider):
        return OllamaProvider(
            base_url=settings.OLLAMA_BASE_URL,
            model=settings.OLLAMA_MODEL,
            timeout=settings.OLLAMA_TIMEOUT,
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS,
        )
    return provider


@lru_cache()
def get_agent_service() -> AgentService:
    return AgentService(
        tool_registry=get_tool_registry(),
        planner=get_planner(),
        context_builder=get_context_builder(),
        llm_provider=_build_generate_llm(),
    )
