import os
from functools import lru_cache

import redis.asyncio as aioredis
import structlog

from orchestrator.config import settings
from orchestrator.context.builder import ContextBuilder, MemoryContextBuilder
from orchestrator.planner.base import BasePlanner
from orchestrator.planner.llm_planner import LLMPlanner
from orchestrator.planner.mock_planner import MockPlanner
from orchestrator.services.agent_service import AgentService
from orchestrator.services.approval_service import ApprovalService
from orchestrator.services.business_services import (
    EmailService,
    EscalationService,
    OrderService,
    PricingService,
    WeatherService,
)
from orchestrator.services.calendar.repository import PendingBookingRepository
from orchestrator.services.calendar_service import CalendarService
from orchestrator.services.fact_extractor import FactExtractor
from orchestrator.services.gmail_client import GmailClient
from orchestrator.services.memory_client import MemoryClient
from orchestrator.services.samvaad_client import SamvaadSessionManager
from orchestrator.services.speech_client import SpeechClient
from orchestrator.services.task_service import TaskService
from orchestrator.services.voice_service import VoiceService
from orchestrator.tools.calendar_tools import register_calendar_tools
from orchestrator.tools.mock_tools import register_mock_tools
from orchestrator.tools.rag_client import HttpRAGClient, MockRAGClient, RAGClient
from orchestrator.tools.registry import ToolRegistry
from orchestrator.workers.memory_writer import MemoryWriterWorker
from shared.llm import LLMProvider, OllamaProvider, OpencodeProvider, SarvamProvider
from shared.usage import UsageRecorder

logger = structlog.get_logger(__name__)


@lru_cache
def _build_rag_client() -> RAGClient:
    return HttpRAGClient(
        base_url=settings.RAG_URL,
        query_path=settings.RAG_QUERY_PATH,
        health_path=settings.RAG_HEALTH_PATH,
        timeout=settings.RAG_TIMEOUT,
        fallback_client=MockRAGClient(),
    )


@lru_cache
def get_tool_registry() -> ToolRegistry:
    registry = ToolRegistry()
    register_mock_tools(
        registry,
        rag_client=_build_rag_client(),
        order_service=get_order_service(),
        pricing_service=get_pricing_service(),
        email_service=get_email_service(),
        escalation_service=get_escalation_service(),
        weather_service=get_weather_service(),
        gmail_client=get_gmail_client(),
    )
    register_calendar_tools(registry, get_calendar_service())
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

    if provider == "opencode":
        logger.info(
            "llm_provider_selected",
            provider="OpencodeProvider",
            model=settings.OPENCODE_MODEL or "opencode-default",
            agent=settings.OPENCODE_AGENT,
        )
        return _opencode_provider()

    logger.warning(
        "llm_provider_not_configured",
        message="Set LLM_PROVIDER=ollama, sarvam, or opencode; using MockPlanner",
    )
    return None


def _opencode_provider(max_tokens: int | None = None) -> OpencodeProvider:
    return OpencodeProvider(
        base_url=settings.OPENCODE_BASE_URL,
        model=settings.OPENCODE_MODEL,
        agent=settings.OPENCODE_AGENT,
        password=settings.OPENCODE_PASSWORD or None,
        username=settings.OPENCODE_USERNAME or "opencode",
        timeout=settings.OPENCODE_TIMEOUT,
        temperature=settings.LLM_TEMPERATURE,
        max_tokens=max_tokens or settings.OPENCODE_MAX_TOKENS,
    )


def _with_usage_hook(provider: LLMProvider | None) -> LLMProvider | None:
    """Attach the usage recorder hook so every LLM call is logged to usage_events."""
    if provider is None:
        return None
    provider._usage_hook = get_usage_recorder()  # type: ignore[attr-defined]
    return provider


@lru_cache
def _build_classify_llm() -> LLMProvider | None:
    provider = _resolve_provider()
    if provider is None:
        return None
    if isinstance(provider, OllamaProvider):
        return _with_usage_hook(OllamaProvider(
            base_url=settings.OLLAMA_BASE_URL,
            model=settings.OLLAMA_MODEL,
            timeout=settings.OLLAMA_TIMEOUT,
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_CLASSIFY_MAX_TOKENS,
        ))
    if isinstance(provider, OpencodeProvider):
        return _with_usage_hook(_opencode_provider(max_tokens=settings.LLM_CLASSIFY_MAX_TOKENS))
    return _with_usage_hook(SarvamProvider(
        api_key=settings.SARVAM_API_KEY,
        base_url=settings.SARVAM_BASE_URL,
        model=settings.SARVAM_MODEL,
        timeout=settings.SARVAM_TIMEOUT,
        temperature=settings.LLM_TEMPERATURE,
        max_tokens=settings.LLM_CLASSIFY_MAX_TOKENS,
    ))


@lru_cache
def get_planner() -> BasePlanner:
    llm = _build_classify_llm()
    if llm is not None:
        model = (
            settings.SARVAM_MODEL
            if settings.LLM_PROVIDER.lower() in ("sarvam", "")
            else settings.OLLAMA_MODEL
        )
        logger.info("planner_initialized", provider=type(llm).__name__, model=model)
        return LLMPlanner(
            llm_provider=llm,
            fallback_intent=settings.LLM_FALLBACK_INTENT,
            pending_repo=get_pending_booking_repository(),
        )
    logger.info("planner_initialized", provider="MockPlanner")
    return MockPlanner()


@lru_cache
def get_context_builder() -> ContextBuilder:
    return MemoryContextBuilder(memory_client=get_memory_client())


@lru_cache
def get_order_service():
    return OrderService()


def get_calendar_service() -> CalendarService:
    return CalendarService()


@lru_cache
def get_pending_booking_repository() -> PendingBookingRepository:
    return PendingBookingRepository()


@lru_cache
def get_pricing_service():
    return PricingService()


@lru_cache
def get_email_service():
    return EmailService()


@lru_cache
def get_weather_service():
    return WeatherService()


@lru_cache
def get_escalation_service():
    return EscalationService()


@lru_cache
def get_task_service() -> TaskService:
    return TaskService()


@lru_cache
def get_gmail_client() -> GmailClient:
    return GmailClient()


@lru_cache
def _build_generate_llm() -> LLMProvider | None:
    provider = _resolve_provider()
    if provider is None:
        return None
    if isinstance(provider, OllamaProvider):
        return _with_usage_hook(OllamaProvider(
            base_url=settings.OLLAMA_BASE_URL,
            model=settings.OLLAMA_MODEL,
            timeout=settings.OLLAMA_TIMEOUT,
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS,
        ))
    return _with_usage_hook(provider)


@lru_cache
def get_usage_recorder() -> UsageRecorder:
    """Recorder used to persist LLM usage rows into the shared usage_events table."""
    if settings.USAGE_PRICING:
        os.environ["USAGE_PRICING"] = settings.USAGE_PRICING
    from orchestrator.database.session import async_session

    return UsageRecorder(
        session_factory=async_session,
        service="orchestrator",
        enabled=settings.USAGE_ENABLED,
    )


@lru_cache
def get_redis_client() -> aioredis.Redis:
    return aioredis.from_url(
        settings.REDIS_URL,
        decode_responses=True,
        socket_keepalive=True,
        socket_connect_timeout=5.0,
        socket_timeout=30.0,
        retry_on_timeout=True,
    )


@lru_cache
def get_memory_client() -> MemoryClient:
    return MemoryClient(base_url=settings.MEMORY_URL, timeout=5.0)


@lru_cache
def get_speech_client() -> SpeechClient:
    return SpeechClient(base_url=settings.SPEECH_URL, timeout=30.0)


@lru_cache
def get_samvaad_session_manager() -> SamvaadSessionManager:
    return SamvaadSessionManager(
        api_key=settings.SAMVAAD_API_KEY,
        agent_id=settings.SAMVAAD_AGENT_ID,
        org_id=settings.SAMVAAD_ORG_ID,
        workspace_id=settings.SAMVAAD_WORKSPACE_ID,
        base_url=settings.SAMVAAD_APP_RUNTIME_URL,
        sample_rate=settings.SAMVAAD_SAMPLE_RATE,
        default_language=settings.SAMVAAD_DEFAULT_LANGUAGE,
        version=settings.SAMVAAD_AGENT_VERSION or None,
        connect_timeout=settings.SAMVAAD_CONNECT_TIMEOUT,
        enabled=settings.SAMVAAD_ENABLED,
        max_turns=settings.SAMVAAD_MAX_TURNS,
        max_duration_s=settings.SAMVAAD_MAX_DURATION_S,
    )


@lru_cache
def get_voice_service() -> VoiceService:
    return VoiceService(
        agent_service=get_agent_service(),
        speech_client=get_speech_client(),
        memory_client=get_memory_client(),
    )


@lru_cache
def get_fact_extractor() -> FactExtractor:
    return FactExtractor(llm_provider=_build_generate_llm())


def get_memory_writer_worker() -> MemoryWriterWorker:
    return MemoryWriterWorker(
        redis_client=get_redis_client(),
        fact_extractor=get_fact_extractor(),
        memory_client=get_memory_client(),
        queue_key=settings.MEMORY_WRITER_QUEUE_KEY,
        enabled=settings.MEMORY_WRITER_ENABLED,
    )


@lru_cache
def get_approval_service() -> ApprovalService:
    return ApprovalService()


@lru_cache
def get_agent_service() -> AgentService:
    return AgentService(
        tool_registry=get_tool_registry(),
        planner=get_planner(),
        context_builder=get_context_builder(),
        llm_provider=_build_generate_llm(),
        memory_writer=get_memory_writer_worker(),
        memory_client=get_memory_client(),
        approval_service=get_approval_service(),
    )
