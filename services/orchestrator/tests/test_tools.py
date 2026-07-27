import pytest

from orchestrator.tools.registry import ToolRegistry
from orchestrator.tools.mock_tools import (
    CalculatorTool,
    SearchDocumentsTool,
    GetWeatherTool,
    SendEmailTool,
    ScheduleMeetingTool,
    register_mock_tools,
)


class TestCalculatorTool:
    @pytest.mark.asyncio
    async def test_simple_addition(self):
        tool = CalculatorTool()
        result = await tool.invoke({"expression": "2 + 3"})
        assert result["success"] is True
        assert result["data"]["result"] == 5

    @pytest.mark.asyncio
    async def test_sqrt(self):
        tool = CalculatorTool()
        result = await tool.invoke({"expression": "sqrt(16)"})
        assert result["success"] is True
        assert result["data"]["result"] == 4.0

    @pytest.mark.asyncio
    async def test_complex_expression(self):
        tool = CalculatorTool()
        result = await tool.invoke({"expression": "2 ** 3 + 4 * 5"})
        assert result["success"] is True
        assert result["data"]["result"] == 28

    @pytest.mark.asyncio
    async def test_invalid_expression(self):
        tool = CalculatorTool()
        result = await tool.invoke({"expression": "1 / 0"})
        assert result["success"] is False
        assert "error" in result["data"]

    @pytest.mark.asyncio
    async def test_empty_expression(self):
        tool = CalculatorTool()
        result = await tool.invoke({"expression": ""})
        assert result["success"] is False

    def test_tool_metadata(self):
        tool = CalculatorTool()
        assert tool.name == "calculator"
        assert "expression" in tool.parameters_schema["required"]


class TestSearchDocumentsTool:
    @pytest.mark.asyncio
    async def test_search_returns_results(self):
        tool = SearchDocumentsTool()
        result = await tool.invoke({"query": "remote work"})
        assert result["success"] is True
        assert len(result["data"]["results"]) > 0

    @pytest.mark.asyncio
    async def test_search_respects_top_k(self):
        tool = SearchDocumentsTool()
        result = await tool.invoke({"query": "policy", "top_k": 2})
        assert result["success"] is True
        assert len(result["data"]["results"]) == 2

    def test_tool_metadata(self):
        tool = SearchDocumentsTool()
        assert tool.name == "search_documents"
        assert "query" in tool.parameters_schema["required"]


class TestGetWeatherTool:
    @pytest.mark.asyncio
    async def test_weather_default_unit(self):
        tool = GetWeatherTool()
        result = await tool.invoke({"location": "London"})
        assert result["success"] is True
        assert result["data"]["location"] == "London"
        assert "temperature" in result["data"]
        assert "conditions" in result["data"]

    @pytest.mark.asyncio
    async def test_weather_fahrenheit(self):
        tool = GetWeatherTool()
        result = await tool.invoke({"location": "Tokyo", "unit": "fahrenheit"})
        assert result["success"] is True
        assert result["data"]["unit"] == "fahrenheit"

    def test_tool_metadata(self):
        tool = GetWeatherTool()
        assert tool.name == "get_weather"
        assert "location" in tool.parameters_schema["required"]


class TestSendEmailTool:
    @pytest.mark.asyncio
    async def test_send_email_success(self):
        tool = SendEmailTool()
        result = await tool.invoke({
            "to": "test@example.com",
            "subject": "Hello",
            "body": "Test message",
        })
        assert result["success"] is True
        assert result["data"]["to"] == "test@example.com"
        assert result["data"]["status"] == "delivered"
        assert "message_id" in result["data"]

    def test_tool_metadata(self):
        tool = SendEmailTool()
        assert tool.name == "send_email"
        assert len(tool.parameters_schema["required"]) == 3


class TestScheduleMeetingTool:
    @pytest.mark.asyncio
    async def test_schedule_meeting(self):
        tool = ScheduleMeetingTool()
        result = await tool.invoke({
            "title": "Sprint Review",
            "attendees": ["dev@example.com", "pm@example.com"],
            "date": "2026-08-01",
            "time": "14:00",
            "duration_minutes": 45,
        })
        assert result["success"] is True
        assert result["data"]["title"] == "Sprint Review"
        assert len(result["data"]["attendees"]) == 2
        assert result["data"]["status"] == "scheduled"

    def test_tool_metadata(self):
        tool = ScheduleMeetingTool()
        assert tool.name == "schedule_meeting"
        assert "attendees" in tool.parameters_schema["required"]


class TestToolRegistry:
    def test_register_and_get(self):
        registry = ToolRegistry()
        tool = CalculatorTool()
        registry.register(tool)
        assert registry.get("calculator") is tool

    def test_register_duplicate_raises(self):
        registry = ToolRegistry()
        registry.register(CalculatorTool())
        with pytest.raises(ValueError, match="already registered"):
            registry.register(CalculatorTool())

    def test_get_missing_returns_none(self):
        registry = ToolRegistry()
        assert registry.get("nonexistent") is None

    def test_list_tools(self):
        registry = ToolRegistry()
        register_mock_tools(registry)
        tools = registry.list_tools()
        assert len(tools) == 10
        names = {t.name for t in tools}
        assert "calculator" in names
        assert "search_documents" in names
        assert "lookup_order" in names
        assert "search_pricing" in names
        assert "transfer_to_human" in names

    def test_get_tool_schemas(self):
        registry = ToolRegistry()
        register_mock_tools(registry)
        schemas = registry.get_tool_schemas()
        assert len(schemas) == 10
        assert all("name" in s and "description" in s and "parameters" in s for s in schemas)


class TestServiceBackedTools:
    @pytest.mark.asyncio
    async def test_lookup_order_extracts_order_id(self):
        from orchestrator.services.mock_services import MockOrderService
        from orchestrator.tools.mock_tools import LookupOrderTool

        tool = LookupOrderTool(order_service=MockOrderService())
        result = await tool.invoke({"order_id": "ORD-7891"})
        assert result["success"] is True
        assert result["data"]["order_id"] == "ORD-7891"
        assert "status" in result["data"]

    @pytest.mark.asyncio
    async def test_search_pricing(self):
        from orchestrator.services.mock_services import MockPricingService
        from orchestrator.tools.mock_tools import SearchPricingTool

        tool = SearchPricingTool(pricing_service=MockPricingService())
        result = await tool.invoke({"query": "enterprise"})
        assert result["success"] is True
        assert len(result["data"]["results"]) >= 1
        tiers = [r["tier"] for r in result["data"]["results"]]
        assert "Enterprise" in tiers

    @pytest.mark.asyncio
    async def test_calendar_tool(self):
        from orchestrator.services.mock_services import MockCalendarService
        from orchestrator.tools.mock_tools import CalendarTool

        tool = CalendarTool(calendar_service=MockCalendarService())
        result = await tool.invoke({"query": "demo"})
        assert result["success"] is True
        slots = result["data"]["available_slots"]
        assert len(slots) >= 1
        assert "date" in slots[0]
        assert "available_slots" in slots[0]

    @pytest.mark.asyncio
    async def test_schedule_demo(self):
        from orchestrator.services.mock_services import MockCalendarService
        from orchestrator.tools.mock_tools import ScheduleDemoTool

        tool = ScheduleDemoTool(calendar_service=MockCalendarService())
        result = await tool.invoke({
            "title": "Q3 Review",
            "attendees": ["user@test.com"],
            "date": "2026-08-01",
            "time": "14:00",
        })
        assert result["success"] is True
        assert result["data"]["status"] == "scheduled"
        assert "DEMO-" in result["data"]["meeting_id"]

    @pytest.mark.asyncio
    async def test_transfer_to_human(self):
        from orchestrator.tools.mock_tools import TransferToHumanTool

        tool = TransferToHumanTool()
        result = await tool.invoke({"reason": "User requested escalation"})
        assert result["success"] is True
        assert result["data"]["status"] == "transferred"
        assert "ESC-" in result["data"]["escalation_id"]

    @pytest.mark.asyncio
    async def test_lookup_order_without_service_fails(self):
        from orchestrator.tools.mock_tools import LookupOrderTool

        tool = LookupOrderTool()
        result = await tool.invoke({"order_id": "ORD-123"})
        assert result["success"] is False
