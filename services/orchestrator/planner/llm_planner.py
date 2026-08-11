import re
import time
from typing import Callable

import structlog

from shared.llm.base import IntentClassification, LLMProvider

from orchestrator.graph.state import PlanStep, AgentState
from orchestrator.planner.base import BasePlanner
from orchestrator.planner.mock_planner import (
    _extract_expression,
    _clean_query,
    _extract_location,
    _extract_subject,
    _extract_meeting_params,
)

logger = structlog.get_logger(__name__)

INTENT_TOOL_MAP: dict[str, list[str]] = {
    "sales": ["search_pricing", "search_documents"],
    "support": ["lookup_order", "search_documents"],
    "booking": ["calendar", "schedule_demo"],
    "general": ["search_documents"],
    "math": ["calculator"],
    "complaint": ["transfer_to_human"],
    "escalate": ["transfer_to_human"],
}

PARAM_EXTRACTORS: dict[str, Callable[[str], dict]] = {
    "calculator": lambda text: {"expression": _extract_expression(text)},
    "search_documents": lambda text: {"query": _clean_query(text), "top_k": 5},
    "search_pricing": lambda text: {"query": _clean_query(text), "top_k": 5},
    "lookup_order": lambda text: {"order_id": _extract_order_id(text)},
    "calendar": lambda text: {"query": text.strip()},
    "schedule_demo": lambda text: _extract_meeting_params(text),
    "schedule_meeting": lambda text: _extract_meeting_params(text),
    "get_weather": lambda text: {"location": _extract_location(text)},
    "send_email": lambda text: {
        "to": "user@example.com",
        "subject": _extract_subject(text),
        "body": text.strip(),
    },
    "transfer_to_human": lambda text: {"reason": text.strip()},
}

_ORDER_ID_PATTERN = re.compile(r"\b(ORD|ord)[-\s]?(\d{3,6})\b")


def _extract_order_id(text: str) -> str:
    match = _ORDER_ID_PATTERN.search(text)
    if match:
        return f"ORD-{match.group(2)}"
    return text.strip()[:30]

HUMAN_ESCALATION_RESPONSE = (
    "I understand your concern and want to make sure it gets the right attention. "
    "I'm escalating this to a human agent who will be able to help you better. "
    "They will review your conversation history and respond shortly."
)

_ESCALATION_PATTERN = re.compile(
    r"\b("
    r"talk\s+to\s+(a\s+|an\s+|the\s+)?(human|real\s+person|agent|someone|representative|support\s+team)"
    r"|(let\s+me\s+speak|connect\s+me|transfer\s+me|put\s+me\s+through)\s+to\s+(a\s+|an\s+|the\s+)?(human|agent|person|representative|support\s+team)"
    r"|(i\s+(want|need)\s+to\s+talk\s+to\s+(a\s+|an\s+|the\s+)?(human|real\s+person|agent))"
    r"|((get|connect)\s+me\s+(a\s+|an\s+)?(human|agent))"
    r")\b",
    re.IGNORECASE,
)

_MATH_PATTERN = re.compile(
    r"^\s*(what\s+is|calculate|compute|solve|eval(?:uate)?)\s+"
    r".*[\d]+[\s\d\+\-\*\/\(\)\.\^x×÷]+[?]?\s*$",
    re.IGNORECASE,
)

_SUPPORT_PATTERN = re.compile(
    r"\b(ord(er)?[-\s]?\d+|track(ing)?(\s+(id|number))?|"
    r"delivery|ship(ping|ment)|return(s)?|refund|"
    r"(where|when)\s+(is|will)\s+my\s+(order|package|delivery|shipment)|"
    r"not\s+arrived|hasn'?t\s+arrived|still\s+waiting)\b",
    re.IGNORECASE,
)

_SALES_PATTERN = re.compile(
    r"\b(pric(e|ing)(\s+(plan|tier))?|purchas(e|ing)|buy(ing)?\s*$|"
    r"subscription|enterprise\s+(plan|pricing|tier)|cost(\s+of|s)?|"
    r"how\s+much|quote|trial|upgrade|downgrade|"
    r"free\s+(plan|tier)|pro\s+(plan|tier))\b",
    re.IGNORECASE,
)

_BOOKING_PATTERN = re.compile(
    r"\b(book\s+(a\s+|the\s+)?(demo|appointment|meeting|call|session)|"
    r"schedule\s+(a\s+|the\s+)?(demo|appointment|meeting|call)|"
    r"set\s+up\s+(a\s+|the\s+)?(demo|meeting|call)|"
    r"reschedule|"
    r"(next|this)\s+(monday|tuesday|wednesday|thursday|friday|week)|"
    r"check\s+(my\s+)?availability)\b",
    re.IGNORECASE,
)

_WEATHER_PATTERN = re.compile(
    r"\b(weather|temperature|forecast|rain(ing)?|sunny|snow(ing)?|climate|humidity)\b",
    re.IGNORECASE,
)

_GREETING_SET = frozenset({
    "hello", "hi", "hey", "yo", "howdy", "sup", "greetings",
    "hello there", "hey there", "hi there",
    "good morning", "good afternoon", "good evening",
    "what's up", "whats up",
})


def _pre_classify(text: str) -> IntentClassification | None:
    lower = text.lower().strip()

    if _ESCALATION_PATTERN.search(lower):
        return IntentClassification(
            intent="escalate",
            confidence=1.0,
            requires_human=True,
            reason="regex: human escalation request",
        )

    if _MATH_PATTERN.search(lower):
        return IntentClassification(
            intent="general",
            confidence=0.95,
            suggested_tools=["calculator"],
            reason="regex: math expression",
        )

    if _WEATHER_PATTERN.search(lower):
        return IntentClassification(
            intent="general",
            confidence=0.92,
            suggested_tools=["get_weather"],
            reason="regex: weather query",
        )

    if _SUPPORT_PATTERN.search(lower):
        return IntentClassification(
            intent="support",
            confidence=0.90,
            suggested_tools=["lookup_order", "search_documents"],
            reason="regex: support/order query",
        )

    if _SALES_PATTERN.search(lower):
        return IntentClassification(
            intent="sales",
            confidence=0.90,
            suggested_tools=["search_pricing", "search_documents"],
            reason="regex: sales/pricing query",
        )

    if _BOOKING_PATTERN.search(lower):
        return IntentClassification(
            intent="booking",
            confidence=0.90,
            suggested_tools=["calendar", "schedule_demo"],
            reason="regex: booking/scheduling query",
        )

    clean = lower.strip(".,!?;:")
    if clean in _GREETING_SET:
        return IntentClassification(
            intent="general",
            confidence=0.95,
            reason="regex: simple greeting",
        )

    return None


class LLMPlanner(BasePlanner):
    def __init__(
        self,
        llm_provider: LLMProvider,
        fallback_intent: str = "general",
    ) -> None:
        self._llm = llm_provider
        self._fallback_intent = fallback_intent

    async def create_plan(self, state: AgentState) -> list[PlanStep]:
        user_input = state["user_input"]
        start = time.perf_counter()

        intent = await self._classify(user_input, _build_context_string(state))
        duration_ms = (time.perf_counter() - start) * 1000

        logger.info(
            "intent_classified",
            intent=intent.intent,
            confidence=intent.confidence,
            requires_human=intent.requires_human,
            entities_count=len(intent.entities),
            suggested_tools=intent.suggested_tools,
            duration_ms=round(duration_ms, 2),
        )

        if intent.intent in ("complaint", "escalate"):
            state["final_response"] = HUMAN_ESCALATION_RESPONSE
            logger.info(
                "plan_escalation_short_circuit",
                intent=intent.intent,
            )
            return []

        tool_names = self._resolve_tools(intent)
        logger.info(
            "tools_resolved",
            intent=intent.intent,
            llm_suggested=intent.suggested_tools,
            final_tools=tool_names,
        )

        steps: list[PlanStep] = []
        for tool_name in tool_names:
            extractor = PARAM_EXTRACTORS.get(tool_name, lambda t: {"input": t})
            steps.append(PlanStep(
                tool_name=tool_name,
                parameters=extractor(user_input),
                reason=f"LLM classified intent as '{intent.intent}' (confidence: {intent.confidence:.2f})",
            ))

        return steps

    def _resolve_tools(self, intent: IntentClassification) -> list[str]:
        if intent.suggested_tools:
            valid = [t for t in intent.suggested_tools if t in PARAM_EXTRACTORS]
            if valid:
                return self._apply_allowlist(valid, intent.intent)

        return self._apply_allowlist(
            INTENT_TOOL_MAP.get(intent.intent, ["search_documents"]),
            intent.intent,
        )

    def _apply_allowlist(self, tool_names: list[str], intent_name: str) -> list[str]:
        """C2: restrict planned tools to the resolved agent's allowlist."""
        from orchestrator.agents.roster import agent_roster

        profile = agent_roster.get(intent_name) or agent_roster.get(agent_roster.agent_fallback)
        allowed = profile.allowed_tools if profile else None
        if allowed is None:
            return tool_names
        filtered = [t for t in tool_names if t in allowed]
        if filtered:
            return filtered
        return [t for t in tool_names if t in PARAM_EXTRACTORS]

    async def _classify(self, user_input: str, context: str | None = None) -> IntentClassification:
        pre = _pre_classify(user_input)
        if pre is not None:
            logger.info(
                "pre_classify_hit",
                intent=pre.intent,
                confidence=pre.confidence,
                reason=pre.reason,
            )
            return pre

        try:
            return await self._llm.classify_intent(user_input, context=context)
        except Exception as exc:
            logger.error("intent_classification_error", error=str(exc))
            return IntentClassification(
                intent=self._fallback_intent,
                confidence=0.0,
                reason=f"Classification error, falling back to '{self._fallback_intent}': {str(exc)}",
            )


def _build_context_string(state: AgentState) -> str | None:
    """Serialize memory/document context for context-aware intent classification."""
    lines: list[str] = []

    memory_context = state.get("memory_context", [])
    for msg in memory_context:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if content:
            lines.append(f"{role}: {content}")

    document_context = state.get("document_context", [])
    for doc in document_context[:3]:
        title = doc.get("title", doc.get("filename", ""))
        snippet = doc.get("snippet") or doc.get("content", "")[:200]
        if title or snippet:
            lines.append(f"document: {title}: {snippet}")

    preferences = state.get("user_preferences", {})
    for key, value in list(preferences.items())[:5]:
        lines.append(f"preference: {key}={value}")

    if not lines:
        return None
    return "\n".join(lines)[:3000]
