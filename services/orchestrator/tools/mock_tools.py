import math
import random
from typing import Any

from orchestrator.tools.base import BaseTool
from orchestrator.tools.rag_client import RAGClient, MockRAGClient


class CalculatorTool(BaseTool):
    name = "calculator"
    description = "Evaluate mathematical expressions. Supports +, -, *, /, **, sqrt, sin, cos, abs."
    parameters_schema = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "The mathematical expression to evaluate, e.g. '2 + 3 * 4'",
            }
        },
        "required": ["expression"],
    }

    _SAFE_BUILTINS = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
        "pow": pow,
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "pi": math.pi,
        "e": math.e,
    }

    async def invoke(self, parameters: dict[str, Any]) -> dict[str, Any]:
        expression = parameters.get("expression", "")
        try:
            result = eval(expression, {"__builtins__": {}}, {**self._SAFE_BUILTINS, **math.__dict__})
            return {"success": True, "data": {"expression": expression, "result": result}}
        except Exception as exc:
            return {"success": False, "data": {"expression": expression, "error": str(exc)}}


class SearchDocumentsTool(BaseTool):
    name = "search_documents"
    description = "Search the knowledge base for documents matching a query."
    parameters_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query"},
            "top_k": {"type": "integer", "description": "Number of results", "default": 5},
        },
        "required": ["query"],
    }

    def __init__(self, rag_client: RAGClient | None = None) -> None:
        self._rag_client = rag_client or MockRAGClient()

    async def invoke(self, parameters: dict[str, Any]) -> dict[str, Any]:
        query = parameters.get("query", "")
        top_k = parameters.get("top_k", 5)
        results = await self._rag_client.search(query, top_k)
        return {"success": True, "data": {"query": query, "results": results}}


class GetWeatherTool(BaseTool):
    name = "get_weather"
    description = "Get current weather conditions for a location."
    parameters_schema = {
        "type": "object",
        "properties": {
            "location": {"type": "string", "description": "City name, e.g. 'Bengaluru'"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "default": "celsius"},
        },
        "required": ["location"],
    }

    def __init__(self, weather_service=None):
        self._weather = weather_service

    async def invoke(self, parameters: dict[str, Any]) -> dict[str, Any]:
        location = parameters.get("location", "Unknown")
        unit = parameters.get("unit", "celsius")
        if self._weather:
            result = await self._weather.get_weather(location, unit)
            return {"success": True, "data": result}
        temp = random.randint(5, 35) if unit == "celsius" else random.randint(40, 95)
        return {
            "success": True,
            "data": {
                "location": location,
                "temperature": temp,
                "unit": unit,
                "conditions": random.choice(["Sunny", "Partly Cloudy", "Cloudy", "Rainy", "Clear"]),
                "humidity": random.randint(30, 90),
            },
        }


class SendEmailTool(BaseTool):
    name = "send_email"
    description = "Send an email to a recipient. Returns delivery confirmation."
    parameters_schema = {
        "type": "object",
        "properties": {
            "to": {"type": "string", "description": "Recipient email address"},
            "subject": {"type": "string", "description": "Email subject line"},
            "body": {"type": "string", "description": "Email body content"},
        },
        "required": ["to", "subject", "body"],
    }

    def __init__(self, email_service=None):
        self._email = email_service

    async def invoke(self, parameters: dict[str, Any]) -> dict[str, Any]:
        to = parameters.get("to", "")
        subject = parameters.get("subject", "")
        body = parameters.get("body", "")
        if self._email:
            result = await self._email.send_email(to, subject, body)
            return {"success": True, "data": result}
        return {
            "success": True,
            "data": {
                "message_id": f"mock-{random.randint(10000, 99999)}",
                "to": to,
                "subject": subject,
                "status": "delivered",
            },
        }


class ScheduleMeetingTool(BaseTool):
    name = "schedule_meeting"
    description = "Schedule a meeting with specified attendees, date, and time."
    parameters_schema = {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Meeting title"},
            "attendees": {"type": "array", "items": {"type": "string"}, "description": "List of attendee emails"},
            "date": {"type": "string", "description": "Date in YYYY-MM-DD format"},
            "time": {"type": "string", "description": "Time in HH:MM format"},
            "duration_minutes": {"type": "integer", "description": "Meeting duration in minutes", "default": 30},
        },
        "required": ["title", "attendees", "date", "time"],
    }

    def __init__(self, calendar_service=None):
        self._calendar = calendar_service

    async def invoke(self, parameters: dict[str, Any]) -> dict[str, Any]:
        title = parameters.get("title", "Untitled")
        attendees = parameters.get("attendees", [])
        date = parameters.get("date", "")
        time = parameters.get("time", "")
        duration = parameters.get("duration_minutes", 30)
        if self._calendar:
            result = await self._calendar.schedule_meeting(title, date, time, attendees)
            return {"success": True, "data": dict(result, duration_minutes=duration)}
        return {
            "success": True,
            "data": {
                "meeting_id": f"mtg-{random.randint(1000, 9999)}",
                "title": title,
                "attendees": attendees,
                "datetime": f"{date}T{time}:00",
                "duration_minutes": duration,
                "status": "scheduled",
            },
        }


class SearchPricingTool(BaseTool):
    name = "search_pricing"
    description = "Search available pricing plans and tiers."

    parameters_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query for pricing plans"},
            "top_k": {"type": "integer", "description": "Number of results", "default": 5},
        },
        "required": ["query"],
    }

    def __init__(self, pricing_service: Any = None) -> None:
        self._pricing = pricing_service

    async def invoke(self, parameters: dict[str, Any]) -> dict[str, Any]:
        query = parameters.get("query", "")
        top_k = parameters.get("top_k", 5)
        if self._pricing is not None:
            results = await self._pricing.search_pricing(query, top_k)
            return {"success": True, "data": {"query": query, "results": results}}
        return {"success": False, "error": "Pricing service not available"}


class LookupOrderTool(BaseTool):
    name = "lookup_order"
    description = "Look up an order by its ID (e.g. ORD-7891). Returns order status, tracking, and delivery details."

    parameters_schema = {
        "type": "object",
        "properties": {
            "order_id": {"type": "string", "description": "The order ID to look up"},
        },
        "required": ["order_id"],
    }

    def __init__(self, order_service: Any = None) -> None:
        self._orders = order_service

    async def invoke(self, parameters: dict[str, Any]) -> dict[str, Any]:
        order_id = parameters.get("order_id", "")
        if self._orders is not None:
            return await self._orders.lookup_order(order_id)
        return {"success": False, "error": "Order service not available"}


class CalendarTool(BaseTool):
    name = "calendar"
    description = "Check calendar availability for scheduling. Returns available time slots for upcoming business days."

    parameters_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Description of the meeting type (e.g. 'demo', 'meeting')"},
            "days_ahead": {"type": "integer", "description": "Number of days to check", "default": 5},
        },
        "required": ["query"],
    }

    def __init__(self, calendar_service: Any = None) -> None:
        self._calendar = calendar_service

    async def invoke(self, parameters: dict[str, Any]) -> dict[str, Any]:
        query = parameters.get("query", "")
        days_ahead = parameters.get("days_ahead", 5)
        if self._calendar is not None:
            slots = await self._calendar.get_availability(query, days_ahead)
            return {"success": True, "data": {"query": query, "available_slots": slots}}
        return {"success": False, "error": "Calendar service not available"}


class ScheduleDemoTool(BaseTool):
    name = "schedule_demo"
    description = "Schedule a demo or appointment with a date and time."

    parameters_schema = {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Meeting title"},
            "attendees": {"type": "array", "items": {"type": "string"}, "description": "List of attendee emails"},
            "date": {"type": "string", "description": "Date in YYYY-MM-DD format"},
            "time": {"type": "string", "description": "Time in HH:MM format"},
            "duration_minutes": {"type": "integer", "description": "Meeting duration in minutes", "default": 30},
        },
        "required": ["title", "attendees", "date", "time"],
    }

    def __init__(self, calendar_service: Any = None) -> None:
        self._calendar = calendar_service

    async def invoke(self, parameters: dict[str, Any]) -> dict[str, Any]:
        title = parameters.get("title", "Demo")
        attendees = parameters.get("attendees", [])
        date = parameters.get("date", "")
        time = parameters.get("time", "")
        duration = parameters.get("duration_minutes", 30)
        if self._calendar is not None:
            return await self._calendar.schedule_demo(title, attendees, date, time, duration)
        return {"success": False, "error": "Calendar service not available"}


class TransferToHumanTool(BaseTool):
    name = "transfer_to_human"
    description = "Transfer the conversation to a human agent. Used for escalations and complaints."

    parameters_schema = {
        "type": "object",
        "properties": {
            "reason": {"type": "string", "description": "Reason for transferring to human agent"},
        },
        "required": ["reason"],
    }

    def __init__(self, escalation_service=None):
        self._escalation = escalation_service

    async def invoke(self, parameters: dict[str, Any]) -> dict[str, Any]:
        reason = parameters.get("reason", "User requested human agent")
        if self._escalation:
            result = await self._escalation.transfer_to_human(
                reason=reason,
                user_input=parameters.get("user_input", reason),
                priority=parameters.get("priority", "NORMAL"),
            )
            return {"success": True, "data": result}
        return {
            "success": True,
            "data": {
                "escalation_id": f"ESC-{random.randint(10000, 99999)}",
                "reason": reason,
                "status": "transferred",
                "message": "Your request has been forwarded to a human agent. They will respond shortly.",
                "estimated_wait_minutes": random.randint(1, 15),
            },
        }


class GmailSendTool(BaseTool):
    name = "email_send"
    description = "Send an email via SMTP using the configured email account."

    parameters_schema = {
        "type": "object",
        "properties": {
            "to": {"type": "string", "description": "Recipient email address"},
            "subject": {"type": "string", "description": "Email subject line"},
            "body": {"type": "string", "description": "Email body content"},
            "cc": {"type": "string", "description": "CC recipient (optional)"},
        },
        "required": ["to", "subject", "body"],
    }

    def __init__(self, gmail_client=None):
        self._email = gmail_client

    async def invoke(self, parameters: dict[str, Any]) -> dict[str, Any]:
        to = parameters.get("to", "")
        subject = parameters.get("subject", "")
        body = parameters.get("body", "")
        cc = parameters.get("cc")
        if self._email and self._email.enabled:
            try:
                import asyncio
                result = await asyncio.to_thread(
                    self._email.send_message, to, subject, body, cc
                )
                return {"success": True, "data": result}
            except Exception as exc:
                return {"success": False, "error": str(exc)}
        return {
            "success": True,
            "data": {
                "message_id": f"mock-{random.randint(10000, 99999)}",
                "to": to,
                "subject": subject,
                "status": "delivered (demo mode)",
            },
        }


class GmailListTool(BaseTool):
    name = "email_list"
    description = "List recent emails from the configured inbox via IMAP."

    parameters_schema = {
        "type": "object",
        "properties": {
            "max_results": {"type": "integer", "description": "Number of emails to list", "default": 5},
            "query": {"type": "string", "description": "Optional IMAP search (e.g. 'FROM \"boss@company.com\"')"},
        },
        "required": [],
    }

    def __init__(self, gmail_client=None):
        self._email = gmail_client

    async def invoke(self, parameters: dict[str, Any]) -> dict[str, Any]:
        max_results = parameters.get("max_results", 5)
        query = parameters.get("query", "")
        if self._email and self._email.enabled:
            try:
                import asyncio
                messages = await asyncio.to_thread(self._email.list_messages, max_results, query)
                return {"success": True, "data": {"messages": messages}}
            except Exception as exc:
                return {"success": False, "error": str(exc)}
        return {
            "success": True,
            "data": {
                "messages": [
                    {
                        "id": "mock-001",
                        "from": "Demo User <demo@example.com>",
                        "subject": "Welcome to AI Employee",
                        "date": "2026-01-01",
                        "snippet": "This is a demo inbox. Configure Gmail SMTP/IMAP to see real emails.",
                    }
                ]
            },
        }


def register_mock_tools(
    registry: "ToolRegistry",
    rag_client: RAGClient | None = None,
    order_service: Any = None,
    calendar_service: Any = None,
    pricing_service: Any = None,
    email_service: Any = None,
    escalation_service: Any = None,
    weather_service: Any = None,
    gmail_client: Any = None,
) -> None:
    registry.register(CalculatorTool())
    registry.register(SearchDocumentsTool(rag_client=rag_client))
    registry.register(GetWeatherTool(weather_service=weather_service))
    registry.register(SendEmailTool(email_service=email_service))
    registry.register(ScheduleMeetingTool(calendar_service=calendar_service))
    registry.register(SearchPricingTool(pricing_service=pricing_service))
    registry.register(LookupOrderTool(order_service=order_service))
    registry.register(CalendarTool(calendar_service=calendar_service))
    registry.register(ScheduleDemoTool(calendar_service=calendar_service))
    registry.register(TransferToHumanTool(escalation_service=escalation_service))
    registry.register(GmailSendTool(gmail_client=gmail_client))
    registry.register(GmailListTool(gmail_client=gmail_client))
