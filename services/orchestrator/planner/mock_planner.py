import re
from typing import Callable

from orchestrator.graph.state import PlanStep, AgentState
from orchestrator.planner.base import BasePlanner


INTENT_PATTERNS: list[tuple[str, str, dict]] = [
    (
        r"\b(weather|temperature|forecast|rain|sunny|snow|climate|humidity)\b",
        "get_weather",
        {},
    ),
    (
        r"\b(calculat|comput|math|solv|sum|product|eval|add|subtract|multiply|divid|\d+[\+\-\*\/]\d+)",
        "calculator",
        {},
    ),
    (
        r"\b(send\s+(an?\s+)?email|mail|compos|draft)",
        "send_email",
        {},
    ),
    (
        r"\b(schedul|meeting|calendar|appointment|book|set\s+up)",
        "schedule_meeting",
        {},
    ),
    (
        r"\b(search|find|look\s*up|look\s*for|look\s*into|quer|tell\s+me\s+about|what\s+(is|are)|how\s+(do|to|does|can|should)|where\s+(is|are|can)|when\s+(is|are|did|does))",
        "search_documents",
        {},
    ),
]


def _extract_expression(text: str) -> str:
    match = re.search(
        r"((?:\d+\.?\d*|\w+)\s*[\+\-\*\/]\s*(?:\d+\.?\d*|\w+)(?:\s*[\+\-\*\/]\s*(?:\d+\.?\d*|\w+))*)",
        text,
    )
    return match.group(1) if match else text


def _clean_query(text: str) -> str:
    for phrase in [
        "search for", "find", "look up", "look into", "query",
        "tell me about", "what is", "what are", "how do", "how to",
        "how does", "where is", "where are", "when is", "when did",
        "calculate", "compute", "whats the weather", "schedule a meeting",
        "send an email",
    ]:
        text = re.sub(rf"\b{re.escape(phrase)}\b", "", text, flags=re.IGNORECASE)
    return text.strip().strip("?.") or text


def _extract_location(text: str) -> str:
    cities = [
        "San Francisco", "New York", "London", "Tokyo", "Paris",
        "Berlin", "Sydney", "Chicago", "Boston", "Toronto", "Mumbai",
        "Singapore", "Dubai", "Seattle", "Los Angeles", "Bengaluru", "Bangalore",
    ]
    for city in cities:
        if city.lower() in text.lower():
            return city
    match = re.search(r"\bin\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)\b", text)
    if match:
        return match.group(1)
    return "San Francisco"


def _extract_subject(text: str) -> str:
    for prefix in ["about", "regarding", "re:", "subject:", "titled", "called", "with subject", "saying"]:
        pattern = re.compile(rf"\b{prefix}\s+(.+?)(?:\s+(?:to|at|on|for|with|and|tomorrow|next|today|schedule)|\s*$)", re.IGNORECASE)
        match = pattern.search(text)
        if match:
            return match.group(1).strip().strip(".,;:")[:80]
    cleaned = re.sub(
        r"\b(send|email|mail|to|compos|draft|an|a|the)\b",
        "", text, flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b",
        "", cleaned,
    )
    cleaned = " ".join(cleaned.split()).strip(" .,;:")
    return cleaned[:80] if cleaned else "No Subject"


def _extract_meeting_params(text: str) -> dict:
    from orchestrator.planner.meeting_parser import parse_meeting_request

    parsed = parse_meeting_request(text)
    return {
        "title": parsed.title,
        "attendees": parsed.attendees or ["user@example.com"],
        "start_at": parsed.start_at.isoformat() if parsed.start_at else None,
        "end_at": parsed.end_at.isoformat() if parsed.end_at else None,
        "duration_minutes": parsed.duration_minutes,
    }


PARAM_EXTRACTORS: dict[str, Callable[[str], dict]] = {
    "calculator": lambda text: {"expression": _extract_expression(text)},
    "search_documents": lambda text: {"query": _clean_query(text), "top_k": 5},
    "get_weather": lambda text: {"location": _extract_location(text)},
    "send_email": lambda text: {
        "to": "user@example.com",
        "subject": _extract_subject(text),
        "body": text.strip(),
    },
    "schedule_meeting": lambda text: _extract_meeting_params(text),
}


class MockPlanner(BasePlanner):
    GENERIC_TOOLS = {"search_documents"}

    async def create_plan(self, state: AgentState) -> list[PlanStep]:
        user_input = state["user_input"]
        steps: list[PlanStep] = []
        seen_tools: set[str] = set()
        has_specific_tool = False

        for pattern, tool_name, _ in INTENT_PATTERNS:
            if re.search(pattern, user_input, re.IGNORECASE) and tool_name not in seen_tools:
                seen_tools.add(tool_name)

                if tool_name not in self.GENERIC_TOOLS:
                    has_specific_tool = True
                elif has_specific_tool:
                    continue

                extractor = PARAM_EXTRACTORS.get(tool_name, lambda t: {"input": t})
                steps.append(PlanStep(
                    tool_name=tool_name,
                    parameters=extractor(user_input),
                    reason=f"Matched intent pattern for {tool_name}",
                ))

        if not steps:
            steps.append(PlanStep(
                tool_name="search_documents",
                parameters={"query": user_input.strip(), "top_k": 3},
                reason="Default fallback: no specific intent matched, searching knowledge base",
            ))

        return steps