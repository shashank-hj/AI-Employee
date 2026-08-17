"""Light-touch agent roster (C2).

Each agent is defined by a system prompt and an allowlist of tools it may use.
The roster is intentionally simple: no subgraph handoffs — the orchestrator
selects the best-matching agent for a plan and enforces its tool allowlist.
"""

from dataclasses import dataclass, field
from typing import Any

from orchestrator.graph.nodes import RESPONSE_SYSTEM_PROMPT

_BOOKING_TOOLS = frozenset({
    "calendar", "schedule_demo", "schedule_meeting",
    "calendar_list", "calendar_update", "calendar_cancel",
})


@dataclass(frozen=True)
class AgentProfile:
    name: str
    description: str
    system_prompt: str
    allowed_tools: frozenset[str] = field(default_factory=frozenset)


AGENT_ROSTER: dict[str, AgentProfile] = {
    "sales": AgentProfile(
        name="sales",
        description="Product information, pricing, and purchase decisions.",
        system_prompt=(
            "You are a Sales AI Employee. You help prospects and customers with product information, pricing, "
            "and purchase decisions. Be enthusiastic, knowledgeable, and solution-oriented. "
            "When presenting pricing, highlight the value proposition of each tier. "
            "Offer to schedule a demo if appropriate. End with a helpful next step."
        ),
        allowed_tools=frozenset({"search_pricing", "search_documents", "calendar", "schedule_demo"}),
    ),
    "support": AgentProfile(
        name="support",
        description="Order tracking, returns, troubleshooting, and account issues.",
        system_prompt=(
            "You are a Support AI Employee. You help customers with order tracking, returns, troubleshooting, "
            "and account issues. Be empathetic, patient, and thorough. If you cannot resolve the issue, "
            "offer to escalate to a human agent. Always confirm that the customer's concern is addressed."
        ),
        allowed_tools=frozenset({"lookup_order", "search_documents"}),
    ),
    "booking": AgentProfile(
        name="booking",
        description="Demos, appointments, and meeting scheduling.",
        system_prompt=(
            "You are a Booking AI Employee. You help users schedule demos, appointments, and meetings. "
            "Be efficient and precise with dates, times, and availability. Confirm all booking details "
            "before finalizing. If the requested time is unavailable, state clearly that the slot is taken "
            "and offer the concrete alternative slots listed in available_slots. "
            "Never discuss pricing, product features, or demos unless the user asks. "
            "Never invent dates, times, or email addresses. "
            "When a meeting is scheduled, state the date/time and the calendar event link."
        ),
        allowed_tools=frozenset({
            "calendar", "schedule_demo", "schedule_meeting",
            "calendar_list", "calendar_update", "calendar_cancel",
        }),
    ),
    "general": AgentProfile(
        name="general",
        description="Greetings, FAQs, chitchat, math, knowledge questions, and email.",
        system_prompt=RESPONSE_SYSTEM_PROMPT,
        allowed_tools=frozenset({"search_documents", "calculator", "get_weather", "send_email", "email_send", "email_list"}),
    ),
    "email": AgentProfile(
        name="email",
        description="Send and read emails.",
        system_prompt=(
            "You are an Email AI Employee. You help users send and read emails. "
            "When sending emails, confirm the recipient, subject, and body before sending. "
            "When listing emails, summarize the most relevant ones. Be professional and concise."
        ),
        allowed_tools=frozenset({"email_send", "email_list"}),
    ),
    "complaint": AgentProfile(
        name="complaint",
        description="Complaints, refund demands, and aggressive dissatisfaction.",
        system_prompt=(
            "You are handling a customer complaint. Be deeply empathetic, acknowledge their frustration, "
            "and confirm that their concern has been escalated to a human agent who will address it personally. "
            "Do not make promises about refunds or compensation — the human agent will handle that."
        ),
        allowed_tools=frozenset({"transfer_to_human"}),
    ),
    "escalate": AgentProfile(
        name="escalate",
        description='Explicit "talk to a human" requests.',
        system_prompt=(
            "The user has requested to speak with a human agent. Confirm that their request "
            "has been received and that a human agent will review their conversation history "
            "and respond shortly. Be courteous and reassuring."
        ),
        allowed_tools=frozenset({"transfer_to_human"}),
    ),
}

AGENT_FALLBACK = "general"


class AgentRoster:
    agent_fallback = AGENT_FALLBACK

    def get(self, name: str) -> AgentProfile | None:
        return AGENT_ROSTER.get(name)

    def resolve_for_tools(self, tool_names: list[str]) -> AgentProfile:
        """Pick the agent whose allowlist covers the most planned tools."""
        if not tool_names:
            return AGENT_ROSTER[AGENT_FALLBACK]

        tool_set = set(tool_names)
        best: AgentProfile | None = None
        best_score = -1

        for profile in AGENT_ROSTER.values():
            overlap = len(tool_set & profile.allowed_tools)
            score = overlap
            # Pure scheduling plans should resolve to the booking agent even
            # when the sales agent shares tools like schedule_demo/calendar.
            if tool_set <= _BOOKING_TOOLS and profile.name == "booking":
                score += 0.5
            if score > best_score:
                best = profile
                best_score = score
            elif score == best_score and profile.name == AGENT_FALLBACK:
                best = profile

        return best or AGENT_ROSTER[AGENT_FALLBACK]

    def list(self) -> list[dict[str, Any]]:
        return [
            {
                "name": profile.name,
                "description": profile.description,
                "allowed_tools": sorted(profile.allowed_tools),
            }
            for profile in AGENT_ROSTER.values()
        ]


agent_roster = AgentRoster()
