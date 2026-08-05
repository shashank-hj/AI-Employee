"""Integration tests for the M5 Memory Writer pipeline."""

import asyncio
import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
import redis.asyncio as aioredis

from orchestrator.services.fact_extractor import FactExtractor, ExtractedFacts
from orchestrator.services.memory_client import MemoryClient
from orchestrator.workers.memory_writer import MemoryWriterWorker
from orchestrator.services.agent_service import AgentService
from orchestrator.schemas.agent import AgentRequest


class TestFactExtractor:
    """Test the LLM-based fact extraction component."""

    async def test_extract_with_mock_llm(self):
        mock_llm = AsyncMock()
        mock_llm.generate.return_value = MagicMock(
            content=json.dumps({
                "display_name": "Priya Sharma",
                "preferences": {"language": "hindi", "channel": "whatsapp"},
                "facts": ["Interested in Enterprise plan", "Works at Acme Corp"],
                "sentiment": "positive",
                "summary": "User asked about pricing and showed interest in Enterprise tier.",
                "topics": ["pricing", "enterprise"],
            }),
            model="llama3.2",
            output_tokens=120,
            duration_ms=500.0,
        )
        extractor = FactExtractor(llm_provider=mock_llm)

        facts = await extractor.extract({
            "user_input": "What is your Enterprise pricing?",
            "final_response": "Our Enterprise plan is $999/month...",
            "tool_results": [
                {"tool_name": "search_pricing", "success": True, "data": {"tier": "Enterprise", "price": "$999/mo"}}
            ],
        })

        assert facts.display_name == "Priya Sharma"
        assert facts.preferences["language"] == "hindi"
        assert len(facts.facts) == 2
        assert facts.sentiment == "positive"
        assert "Enterprise" in facts.summary
        assert "pricing" in facts.topics

    async def test_extract_no_llm_returns_empty(self):
        extractor = FactExtractor(llm_provider=None)
        facts = await extractor.extract({"user_input": "hi"})
        assert facts == ExtractedFacts()

    async def test_extract_json_parse_failure_graceful(self):
        mock_llm = AsyncMock()
        mock_llm.generate.return_value = MagicMock(
            content="This is not JSON at all",
            model="llama3.2",
            output_tokens=5,
            duration_ms=100.0,
        )
        extractor = FactExtractor(llm_provider=mock_llm)
        facts = await extractor.extract({"user_input": "hi"})
        assert facts == ExtractedFacts()

    async def test_extract_llm_exception_graceful(self):
        mock_llm = AsyncMock()
        mock_llm.generate.side_effect = RuntimeError("LLM down")
        extractor = FactExtractor(llm_provider=mock_llm)
        facts = await extractor.extract({"user_input": "hi"})
        assert facts == ExtractedFacts()


class TestMemoryWriterWorker:
    """Test the background worker that consumes Redis queue jobs."""

    @pytest.fixture
    async def redis_client(self):
        client = aioredis.from_url("redis://localhost:6379/0", decode_responses=True)
        yield client
        await client.flushdb()
        await client.aclose()

    @pytest.fixture
    def mock_fact_extractor(self):
        extractor = AsyncMock(spec=FactExtractor)
        extractor.extract.return_value = ExtractedFacts(
            display_name="Test User",
            preferences={"language": "english"},
            facts=["Likes fast responses"],
            sentiment="positive",
            summary="User asked about pricing.",
            topics=["pricing"],
        )
        return extractor

    @pytest.fixture
    def mock_memory_client(self):
        return AsyncMock(spec=MemoryClient)

    @pytest.fixture
    def worker(self, redis_client, mock_fact_extractor, mock_memory_client):
        return MemoryWriterWorker(
            redis_client=redis_client,
            fact_extractor=mock_fact_extractor,
            memory_client=mock_memory_client,
            queue_key="test_memory_writer_queue",
            retry_delay=0.1,
            enabled=True,
        )

    async def test_enqueue_and_process(self, worker, redis_client, mock_memory_client):
        worker.start()
        await asyncio.sleep(0.1)  # let worker start

        job = {
            "request_id": "req-123",
            "user_id": "user-456",
            "session_id": "sess-789",
            "user_input": "What is your pricing?",
            "final_response": "Our pricing starts at $99/mo...",
            "tool_results": [],
            "execution_log": [],
        }
        enqueued = await worker.enqueue(job)
        assert enqueued is True

        # Wait for worker to process
        await asyncio.sleep(1.5)
        await worker.stop()

        # Verify memory client calls
        mock_memory_client.update_profile.assert_awaited_once()
        profile_call = mock_memory_client.update_profile.await_args
        assert profile_call.kwargs["user_id"] == "user-456"
        assert profile_call.kwargs["display_name"] == "Test User"

        # Should store summary + 1 fact = 2 long_term calls
        assert mock_memory_client.store_long_term.await_count == 2

    async def test_enqueue_skips_without_user_id(self, worker, redis_client, mock_memory_client):
        worker.start()
        await asyncio.sleep(0.1)

        job = {
            "request_id": "req-123",
            "user_id": None,
            "session_id": "sess-789",
            "user_input": "hi",
            "final_response": "Hello!",
        }
        await worker.enqueue(job)
        await asyncio.sleep(1.5)
        await worker.stop()

        mock_memory_client.update_profile.assert_not_awaited()
        mock_memory_client.store_long_term.assert_not_awaited()

    async def test_disabled_worker_skips_enqueue(self, redis_client, mock_fact_extractor, mock_memory_client):
        disabled_worker = MemoryWriterWorker(
            redis_client=redis_client,
            fact_extractor=mock_fact_extractor,
            memory_client=mock_memory_client,
            queue_key="test_disabled_queue",
            retry_delay=0.1,
            enabled=False,
        )
        enqueued = await disabled_worker.enqueue({"request_id": "x", "user_id": "u"})
        assert enqueued is False


class TestAgentServiceMemoryWriterIntegration:
    """Test that AgentService correctly enqueues completed conversations."""

    async def test_run_enqueues_conversation(self):
        from orchestrator.planner.mock_planner import MockPlanner
        from orchestrator.context.builder import MockContextBuilder
        from orchestrator.tools.registry import ToolRegistry
        from orchestrator.tools.mock_tools import register_mock_tools
        from orchestrator.tools.rag_client import MockRAGClient
        from orchestrator.services.mock_services import MockOrderService, MockCalendarService, MockPricingService

        registry = ToolRegistry()
        register_mock_tools(
            registry,
            rag_client=MockRAGClient(),
            order_service=MockOrderService(),
            calendar_service=MockCalendarService(),
            pricing_service=MockPricingService(),
        )

        mock_writer = AsyncMock(spec=MemoryWriterWorker)
        mock_writer.enqueue.return_value = True

        service = AgentService(
            tool_registry=registry,
            planner=MockPlanner(),
            context_builder=MockContextBuilder(),
            llm_provider=None,
            memory_writer=mock_writer,
        )

        request = AgentRequest(
            user_input="What is 5 + 5?",
            user_id="test-user-42",
            session_id="test-session-99",
        )
        response = await service.run(request)

        assert response.request_id is not None
        mock_writer.enqueue.assert_awaited_once()
        job = mock_writer.enqueue.await_args.args[0]
        assert job["user_id"] == "test-user-42"
        assert job["session_id"] == "test-session-99"
        assert job["user_input"] == "What is 5 + 5?"
        assert "final_response" in job
        assert "tool_results" in job

    async def test_run_without_memory_writer_does_not_crash(self):
        from orchestrator.planner.mock_planner import MockPlanner
        from orchestrator.context.builder import MockContextBuilder
        from orchestrator.tools.registry import ToolRegistry
        from orchestrator.tools.mock_tools import register_mock_tools
        from orchestrator.tools.rag_client import MockRAGClient
        from orchestrator.services.mock_services import MockOrderService, MockCalendarService, MockPricingService

        registry = ToolRegistry()
        register_mock_tools(
            registry,
            rag_client=MockRAGClient(),
            order_service=MockOrderService(),
            calendar_service=MockCalendarService(),
            pricing_service=MockPricingService(),
        )

        service = AgentService(
            tool_registry=registry,
            planner=MockPlanner(),
            context_builder=MockContextBuilder(),
            llm_provider=None,
            memory_writer=None,
        )

        request = AgentRequest(user_input="What is 5 + 5?")
        response = await service.run(request)
        assert response.final_response is not None


class TestFactExtractorParseResponse:
    """Verify fixes for JSON parsing edge cases."""

    def test_parse_json_with_trailing_text(self):
        content = (
            'Based on the conversation, here are the facts:\n'
            '{"display_name": "Priya", "preferences": {"lang": "hi"}, '
            '"facts": ["uses WhatsApp"], "sentiment": "positive", '
            '"summary": "User asked about pricing.", "topics": ["pricing"]}\n'
            'Let me know if you need anything else!'
        )
        result = FactExtractor._parse_response(content)
        assert result.display_name == "Priya"
        assert result.preferences["lang"] == "hi"
        assert result.facts == ["uses WhatsApp"]
        assert result.sentiment == "positive"
        assert "pricing" in result.topics

    def test_parse_multiple_json_objects_in_content(self):
        content = 'extra text {"display_name": "B", "preferences": {"x": 1}, "facts": [], "sentiment": "neutral", "summary": "ok", "topics": []} trailing'
        result = FactExtractor._parse_response(content)
        assert result.display_name == "B"
        assert result.preferences["x"] == 1

    def test_parse_code_fenced_json(self):
        content = (
            "Here is the analysis:\n\n"
            "```json\n"
            '{"display_name": "Amit", "preferences": {"language": "english"}, '
            '"facts": ["works at XYZ"], "sentiment": "positive", '
            '"summary": "User asked a question.", "topics": ["support"]}\n'
            "```\n\n"
            "End of analysis."
        )
        result = FactExtractor._parse_response(content)
        assert result.display_name == "Amit"
        assert result.preferences["language"] == "english"
        assert result.facts == ["works at XYZ"]


class TestMemoryWriterEmptyFactsGuard:
    """Verify the empty-facts guard correctly handles empty ExtractedFacts."""

    @pytest.fixture
    async def redis_client(self):
        client = aioredis.from_url("redis://localhost:6379/0", decode_responses=True)
        yield client
        await client.flushdb()
        await client.aclose()

    async def test_empty_extraction_skips_write(self, redis_client):
        mock_extractor = AsyncMock(spec=FactExtractor)
        mock_extractor.extract.return_value = ExtractedFacts()

        mock_memory = AsyncMock(spec=MemoryClient)

        worker = MemoryWriterWorker(
            redis_client=redis_client,
            fact_extractor=mock_extractor,
            memory_client=mock_memory,
            queue_key="test_empty_guard",
            retry_delay=0.1,
            enabled=True,
        )
        worker.start()

        job = {
            "request_id": "req-empty",
            "user_id": "user-empty",
            "session_id": "sess-empty",
            "user_input": "hi",
            "final_response": "Hello",
            "tool_results": [],
            "execution_log": [],
        }
        await worker.enqueue(job)
        await asyncio.sleep(1.0)
        await worker.stop()

        mock_memory.update_profile.assert_not_awaited()
        mock_memory.store_long_term.assert_not_awaited()

    async def test_extraction_with_only_empty_prefs_skips_write(self, redis_client):
        mock_extractor = AsyncMock(spec=FactExtractor)
        mock_extractor.extract.return_value = ExtractedFacts(preferences={})

        mock_memory = AsyncMock(spec=MemoryClient)

        worker = MemoryWriterWorker(
            redis_client=redis_client,
            fact_extractor=mock_extractor,
            memory_client=mock_memory,
            queue_key="test_empty_prefs",
            retry_delay=0.1,
            enabled=True,
        )
        worker.start()

        job = {
            "request_id": "req-123",
            "user_id": "user-456",
            "session_id": "sess-789",
            "user_input": "hi",
            "final_response": "Hello",
        }
        await worker.enqueue(job)
        await asyncio.sleep(1.0)
        await worker.stop()

        mock_memory.update_profile.assert_not_awaited()
        mock_memory.store_long_term.assert_not_awaited()


class TestMemoryWriterFIFOOrder:
    """Verify jobs are processed in FIFO order (enqueue→rpush, consume→lpop)."""

    @pytest.fixture
    async def redis_client(self):
        client = aioredis.from_url("redis://localhost:6379/0", decode_responses=True)
        yield client
        await client.flushdb()
        await client.aclose()

    async def test_jobs_processed_in_fifo_order(self, redis_client):
        processed_order = []

        mock_extractor = AsyncMock(spec=FactExtractor)
        mock_extractor.extract.return_value = ExtractedFacts(
            display_name="User",
            preferences={"lang": "en"},
            facts=["test fact"],
            sentiment="neutral",
            summary="Test summary",
            topics=["test"],
        )

        mock_memory = AsyncMock(spec=MemoryClient)

        worker = MemoryWriterWorker(
            redis_client=redis_client,
            fact_extractor=mock_extractor,
            memory_client=mock_memory,
            queue_key="test_fifo_queue",
            retry_delay=0.1,
            enabled=True,
        )
        worker.start()

        for i in range(3):
            await worker.enqueue({
                "request_id": f"req-{i}",
                "user_id": f"user-{i}",
                "session_id": f"sess-{i}",
                "user_input": f"msg-{i}",
                "final_response": f"resp-{i}",
            })

        await asyncio.sleep(2.0)
        await worker.stop()

        profile_calls = mock_memory.update_profile.await_args_list
        user_ids = [c.kwargs["user_id"] for c in profile_calls]
        assert user_ids == ["user-0", "user-1", "user-2"]
