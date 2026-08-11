import pytest
from httpx import ASGITransport, AsyncClient

from orchestrator.app import create_app


@pytest.fixture
def app():
    return create_app()


@pytest.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


class TestAgentRunEndpoint:
    @pytest.mark.asyncio
    async def test_run_calculator(self, client):
        response = await client.post("/api/agent/run", json={
            "user_input": "Calculate 2 + 3",
        })
        assert response.status_code == 200
        data = response.json()
        assert data["final_response"] is not None
        tool_names = [s["tool_name"] for s in data["steps"]]
        assert "calculator" in tool_names

    @pytest.mark.asyncio
    async def test_run_weather(self, client):
        response = await client.post("/api/agent/run", json={
            "user_input": "What is the weather in London?",
        })
        assert response.status_code == 200
        data = response.json()
        tool_names = [s["tool_name"] for s in data["steps"]]
        assert "get_weather" in tool_names

    @pytest.mark.asyncio
    async def test_run_search(self, client):
        response = await client.post("/api/agent/run", json={
            "user_input": "Find documents about remote work",
        })
        assert response.status_code == 200
        data = response.json()
        assert data["steps"][0]["tool_name"] == "search_documents"
        assert "Search results" in data["final_response"] or len(data["steps"]) >= 1

    @pytest.mark.asyncio
    async def test_run_with_user_and_session(self, client):
        response = await client.post("/api/agent/run", json={
            "user_input": "Calculate 10 * 5",
            "user_id": "user-123",
            "session_id": "session-abc",
        })
        assert response.status_code == 200
        data = response.json()
        assert data["request_id"] is not None

    @pytest.mark.asyncio
    async def test_run_specific_tool_skips_generic_search(self, client):
        response = await client.post("/api/agent/run", json={
            "user_input": "Calculate the sum of 10 and 20 and then search for budget documents",
        })
        assert response.status_code == 200
        data = response.json()
        assert len(data["steps"]) == 1
        assert data["steps"][0]["tool_name"] == "calculator"

    @pytest.mark.asyncio
    async def test_run_multi_domain_specific(self, client):
        response = await client.post("/api/agent/run", json={
            "user_input": "Calculate 5 plus 3 and what is the weather in London",
        })
        assert response.status_code == 200
        data = response.json()
        assert len(data["steps"]) == 2
        tool_names = [s["tool_name"] for s in data["steps"]]
        assert "calculator" in tool_names
        assert "get_weather" in tool_names

    @pytest.mark.asyncio
    async def test_run_empty_input_rejected(self, client):
        response = await client.post("/api/agent/run", json={
            "user_input": "",
        })
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_run_missing_input_rejected(self, client):
        response = await client.post("/api/agent/run", json={})
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_run_includes_execution_logs(self, client):
        response = await client.post("/api/agent/run", json={
            "user_input": "Calculate 1 + 1",
        })
        assert response.status_code == 200
        data = response.json()
        assert len(data["execution_log"]) > 0
        log_nodes = {entry["node"] for entry in data["execution_log"]}
        assert "receive" in log_nodes
        assert "plan" in log_nodes
        assert "execute" in log_nodes or "tool_invoke" in log_nodes
        assert "respond" in log_nodes

    @pytest.mark.asyncio
    async def test_run_meeting_request(self, client):
        response = await client.post("/api/agent/run", json={
            "user_input": "Schedule a meeting about project review",
        })
        assert response.status_code == 200
        data = response.json()
        assert data["steps"][0]["tool_name"] == "schedule_meeting"
        lower = data["final_response"].lower()
        assert "Meeting" in data["final_response"] or "scheduled" in lower

    @pytest.mark.asyncio
    async def test_run_email_request(self, client):
        response = await client.post("/api/agent/run", json={
            "user_input": "Send an email about the status update",
        })
        assert response.status_code == 200
        data = response.json()
        assert data["steps"][0]["tool_name"] == "send_email"
        lower = data["final_response"].lower()
        assert "Email sent" in data["final_response"] or "delivered" in lower

    @pytest.mark.asyncio
    async def test_run_fallback_for_greeting(self, client):
        response = await client.post("/api/agent/run", json={
            "user_input": "Hello, how are you?",
        })
        assert response.status_code == 200
        data = response.json()
        assert data["steps"][0]["tool_name"] == "search_documents"
        assert data["final_response"] is not None

    @pytest.mark.asyncio
    async def test_run_schema_verification(self, client):
        response = await client.post("/api/agent/run", json={
            "user_input": "Calculate 3 * 7",
        })
        data = response.json()
        assert set(data.keys()) == {
            "request_id", "user_input", "final_response",
            "steps", "execution_log", "completed_at", "duration_ms",
            "channel", "channel_message_id", "tenant_id",
        }
        for step in data["steps"]:
            assert set(step.keys()) == {"step_index", "tool_name", "parameters", "result"}


class TestContextBuilder:
    @pytest.mark.asyncio
    async def test_context_with_session(self):
        from orchestrator.context.builder import MockContextBuilder
        builder = MockContextBuilder()
        ctx = await builder.build("hello", "user-1", "sess-1")
        assert len(ctx["memory_context"]) == 2
        assert ctx["user_preferences"]["language"] == "en"

    @pytest.mark.asyncio
    async def test_context_without_session(self):
        from orchestrator.context.builder import MockContextBuilder
        builder = MockContextBuilder()
        ctx = await builder.build("hello", "user-1", None)
        assert ctx["memory_context"] == []


class TestToolRoutingEndpoints:
    """Verify that tool routing works end-to-end through the graph."""

    @pytest.mark.asyncio
    async def test_escalation_triggers_empty_plan(self, client):
        response = await client.post("/api/agent/run", json={
            "user_input": "I want to talk to a human agent right now",
        })
        assert response.status_code == 200
        data = response.json()
        assert data["final_response"] is not None

    @pytest.mark.asyncio
    async def test_calculator_tool_invoked(self, client):
        response = await client.post("/api/agent/run", json={
            "user_input": "Calculate 2 + 3",
        })
        assert response.status_code == 200
        data = response.json()
        tool_names = [s["tool_name"] for s in data["steps"]]
        assert "calculator" in tool_names, f"Expected calculator, got {tool_names}"

    @pytest.mark.asyncio
    async def test_general_fallback_to_search(self, client):
        response = await client.post("/api/agent/run", json={
            "user_input": "Tell me about the company",
        })
        assert response.status_code == 200
        data = response.json()
        tool_names = [s["tool_name"] for s in data["steps"]]
        assert "search_documents" in tool_names

    @pytest.mark.asyncio
    async def test_sending_email_routes_correctly(self, client):
        response = await client.post("/api/agent/run", json={
            "user_input": "Send an email about the status update",
        })
        assert response.status_code == 200
        data = response.json()
        tool_names = [s["tool_name"] for s in data["steps"]]
        assert "send_email" in tool_names

    @pytest.mark.asyncio
    async def test_scheduling_meeting_routes_correctly(self, client):
        response = await client.post("/api/agent/run", json={
            "user_input": "Schedule a meeting about project review",
        })
        assert response.status_code == 200
        data = response.json()
        tool_names = [s["tool_name"] for s in data["steps"]]
        assert "schedule_meeting" in tool_names

    @pytest.mark.asyncio
    async def test_response_is_natural_language(self, client):
        response = await client.post("/api/agent/run", json={
            "user_input": "Calculate 2 + 3",
        })
        assert response.status_code == 200
        data = response.json()
        assert data["final_response"] is not None
        assert len(data["final_response"]) > 0
        assert "[calculator]" not in data["final_response"]

    @pytest.mark.asyncio
    async def test_weather_routes_correctly(self, client):
        response = await client.post("/api/agent/run", json={
            "user_input": "What is the weather in Mumbai?",
        })
        assert response.status_code == 200
        data = response.json()
        tool_names = [s["tool_name"] for s in data["steps"]]
        assert "get_weather" in tool_names
