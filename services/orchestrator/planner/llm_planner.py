import json
import re
import time
from typing import Any, Callable

import structlog

from shared.llm.base import IntentClassification, LLMProvider

from orchestrator.config import settings
from orchestrator.graph.state import PlanStep, AgentState
from orchestrator.planner.base import BasePlanner
from orchestrator.planner.meeting_parser import (
    extract_meeting_ref,
    parse_meeting_request,
)
from orchestrator.planner.mock_planner import (
    _extract_expression,
    _clean_query,
    _extract_location,
    _extract_subject,
)

logger = structlog.get_logger(__name__)


def _merge_attendees(parsed: list[str], structured: list[str]) -> list[str]:
    """Merge parsed attendees with structured ones, de-duplicated, preserving order."""
    seen: set[str] = set()
    merged: list[str] = []
    for email in list(parsed) + list(structured):
        normalized = email.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        merged.append(normalized)
    return merged


INTENT_TOOL_MAP: dict[str, list[str]] = {
    "sales": ["search_pricing", "search_documents"],
    "support": ["lookup_order", "search_documents"],
    "booking": ["calendar"],
    "calendar_list": ["calendar_list"],
    "calendar_update": ["calendar_update"],
    "calendar_cancel": ["calendar_cancel"],
    "general": ["search_documents"],
    "math": ["calculator"],
    "complaint": ["transfer_to_human"],
    "escalate": ["transfer_to_human"],
    "email": ["email_send"],
}

_CALENDAR_TOOLS = {
    "calendar",
    "schedule_demo",
    "schedule_meeting",
    "calendar_update",
    "calendar_cancel",
    "calendar_list",
}

# Map management intents to the booking agent for allowlist enforcement.
_CALENDAR_INTENT_AGENT = {
    "calendar_list": "booking",
    "calendar_update": "booking",
    "calendar_cancel": "booking",
}

PARAM_EXTRACTORS: dict[str, Callable[[str], dict]] = {
    "calculator": lambda text: {"expression": _extract_expression(text)},
    "search_documents": lambda text: {"query": _clean_query(text), "top_k": 5},
    "search_pricing": lambda text: {"query": _clean_query(text), "top_k": 5},
    "lookup_order": lambda text: {"order_id": _extract_order_id(text)},
    "calendar": lambda text: {"query": text.strip()},
    "schedule_demo": lambda text: _meeting_params_from_text(text),
    "schedule_meeting": lambda text: _meeting_params_from_text(text),
    "calendar_update": lambda text: _meeting_params_from_text(text),
    "calendar_cancel": lambda text: _meeting_params_from_text(text),
    "calendar_list": lambda text: {},
    "get_weather": lambda text: {"location": _extract_location(text)},
    "send_email": lambda text: _extract_email_params(text),
    "email_send": lambda text: _extract_email_params(text),
    "email_list": lambda text: {"max_results": 5, "query": ""},
    "transfer_to_human": lambda text: {"reason": text.strip()},
}

_ORDER_ID_PATTERN = re.compile(r"\b(ORD|ord)[-\s]?(\d{3,6})\b")

_EMAIL_ADDR_PATTERN = re.compile(
    r"\b([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b"
)


def _extract_order_id(text: str) -> str:
    match = _ORDER_ID_PATTERN.search(text)
    if match:
        return f"ORD-{match.group(2)}"
    return text.strip()[:30]


EMAIL_DRAFT_SYSTEM_PROMPT = (
    "You draft professional emails. Given a user's request to send an email, return ONLY a "
    "single JSON object with exactly these keys: "
    "\"to\" (the recipient email address as a string, or empty string if none), "
    "\"subject\" (a concise, professional subject line), and "
    "\"body\" (the complete email message: a greeting, the explanation/request in detail, "
    "and a sign-off such as 'Best regards'). "
    "Write the email in the same language as the user's request. "
    "Return only the JSON object, with no surrounding commentary, no markdown, no extra text."
)


def _extract_json_object(content: str) -> dict:
    """Best-effort extraction of a JSON object from an LLM response."""
    if not content:
        return {}
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.DOTALL | re.IGNORECASE)
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
    return {}


def _extract_email_params(text: str) -> dict[str, str]:
    to = ""
    match = _EMAIL_ADDR_PATTERN.search(text)
    if match:
        to = match.group(1)

    request = text
    if match:
        request = request.replace(match.group(0), " ")
    request = " ".join(re.sub(r"\s+", " ", request).split())

    if not to:
        return {"to": to, "subject": request or "(no subject)", "body": request}

    subject = _build_email_subject(request)
    body = _build_email_body(request, subject)

    return {"to": to, "subject": subject, "body": body}


_EMAIL_INTRO_WORDS = re.compile(
    r"\b(send|email|mail|compose|draft|write|an?|the)\b", re.IGNORECASE
)

# A standalone 'to' that is not part of 'due to' / 'for to' (e.g. 'email to <addr>').
_STANDALONE_TO = re.compile(
    r"(?<!\b(?:due|for))\bto\b(?!\s+\S+\.\S+)", re.IGNORECASE
)


def _strip_fillers(text: str) -> str:
    cleaned = _EMAIL_INTRO_WORDS.sub(" ", text)
    cleaned = _STANDALONE_TO.sub(" ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .,;:")
    return cleaned


def _build_email_subject(request: str) -> str:
    cleaned = _strip_fillers(request)
    lowered = cleaned.lower()
    if "leave" in lowered:
        return "Request for Leave"
    match = re.search(
        r"\b(?:about|regarding|re:|subject:?|titled?|called)\s+(.+?)(?:\s+(?:due\s+to|for|on|tomorrow|next|at)\b|\.?$)",
        cleaned,
        re.IGNORECASE,
    )
    if match:
        topic = match.group(1).strip(" .,;:")
        if topic:
            return topic[:80]
    return cleaned[:80] or "(no subject)"

def _build_email_body(request: str, subject: str) -> str:
    raw = request

    reason = "personal reasons"
    m = re.search(
        r"\b(?:due\s+to|because\s+of|owing\s+to)\s+(.+?)(?:\s+(?:tomorrow|next\b|on\b|for\b)|\.?$)",
        raw,
        re.IGNORECASE,
    )
    if m and m.group(1).strip():
        reason = re.sub(r"\bsend\b|\ban?\b|\bthe\b", " ", m.group(1), flags=re.IGNORECASE)
        reason = " ".join(reason.split()).strip(" .,;:") or reason

    day = "tomorrow"
    mday = re.search(
        r"\bnext\s+(monday|tuesday|wednesday|thursday|friday|week)\b|(?:\bon\b\s+)?(monday|tuesday|wednesday|thursday|friday)\b|\btomorrow\b",
        raw,
        re.IGNORECASE,
    )
    if mday:
        day = mday.group(0).strip(" .,;:")
        if day.lower().startswith("on "):
            day = day[3:]
        if day.lower() == "tomorrow":
            day = "tomorrow"
        else:
            day = day.title()

    lines = [
        f"Subject: {subject}",
        "",
        "Dear Team,",
        "",
        f"I would like to request leave for {day} due to {reason}.",
        "",
        "I will make sure my pending work is completed before the leave and "
        "remain reachable for urgent matters.",
        "",
        "Please let me know if any further information is required.",
        "",
        "Thank you,",
        "Best regards,",
        "[Your Name]",
    ]
    return "\n".join(lines)


_ABOVE_REFERENCE_PATTERN = re.compile(
    r"\b(above|the\s+above|this|previous|prior|last|recent|that\s+information|"
    r"the\s+information|my\s+question|as\s+mentioned|cop(y|ies))\b",
    re.IGNORECASE,
)


def _enrich_fallback_with_context(
    fallback: dict[str, str],
    user_input: str,
    context: str | None,
) -> dict[str, str]:
    """If the user refers to prior conversation content (\"send the above info\"),
    embed the most recent assistant answer into the email body so the email actually
    carries what they meant to send. Falls back to the heuristic draft otherwise."""
    if not context:
        return fallback
    if not _ABOVE_REFERENCE_PATTERN.search(user_input):
        return fallback

    recent = ""
    for line in reversed(context.splitlines()):
        stripped = line.strip()
        if stripped.lower().startswith("assistant:"):
            recent = stripped.split(":", 1)[1].strip()
            break
        if stripped.lower().startswith("user:"):
            # Most recent segment ended; keep whatever assistant text came before.
            break

    if not recent:
        return fallback

    subject = fallback.get("subject") or "(no subject)"
    if (
        subject in ("(no subject)", "above information", "above info")
        or not re.search(r"\w", re.sub(r"\babove\b|\binfo(rmation)?\b|\bid\b", "", subject, flags=re.IGNORECASE))
    ):
        subject = "Information You Requested"

    body = (
        f"Subject: {subject}\n\n"
        f"Dear Team,\n\n"
        f"Please find the information you requested below:\n\n"
        f"{recent}\n\n"
        f"Best regards,\n[Your Name]"
    )
    return {"to": fallback.get("to", ""), "subject": subject, "body": body}


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
    r"(next|this)\s+(monday|tuesday|wednesday|thursday|friday|week)|"
    r"check\s+(my\s+)?availability)\b",
    re.IGNORECASE,
)

_CALENDAR_LIST_PATTERN = re.compile(
    r"\b(list|show|view|what).{0,20}(meetings|events|appointments|demos|schedule|calendar)|"
    r"\bupcoming\s+(meetings|events|appointments)|"
    r"\bmy\s+(meetings|schedule|calendar|appointments)\b",
    re.IGNORECASE,
)

_CALENDAR_CANCEL_PATTERN = re.compile(
    r"\b(cancel|delete|remove|call\s+off)\b"
    r"(?:\s+(?:my|the|a|an|this|that))?"
    r"(?:\s+\w+){0,8}\s*(meeting|appointment|demo|booking|event|call|schedule)\b",
    re.IGNORECASE,
)

_CALENDAR_UPDATE_PATTERN = re.compile(
    r"\b(reschedule|move|postpone|change|shift|update)\b"
    r"(?:\s+(?:my|the|a|an|this|that))?"
    r"(?:\s+\w+){0,8}\s*(meeting|appointment|demo|booking|event|call|schedule)\b",
    re.IGNORECASE,
)

_WEATHER_PATTERN = re.compile(
    r"\b(weather|temperature|forecast|rain(ing)?|sunny|snow(ing)?|climate|humidity)\b",
    re.IGNORECASE,
)

_EMAIL_PATTERN = re.compile(
    r"\b(send\s+(an?\s+)?email|email\s+(to\s+)?|"
    r"compose\s+(an?\s+)?email|write\s+(an?\s+)?email|"
    r"draft\s+(an?\s+)?email|mail\s+(to\s+)?|send\s+(a\s+)?mail|"
    r"check\s+(my\s+)?inbox|list\s+(my\s+)?emails|show\s+(my\s+)?emails)\b",
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

    if _EMAIL_PATTERN.search(lower):
        has_addr = bool(_EMAIL_ADDR_PATTERN.search(lower))
        return IntentClassification(
            intent="email",
            confidence=0.90,
            suggested_tools=["email_send"] if has_addr else ["email_send", "email_list"],
            reason="regex: email request",
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

    if _CALENDAR_UPDATE_PATTERN.search(lower):
        return IntentClassification(
            intent="calendar_update",
            confidence=0.90,
            suggested_tools=["calendar_update"],
            reason="regex: reschedule meeting query",
        )

    if _CALENDAR_CANCEL_PATTERN.search(lower):
        return IntentClassification(
            intent="calendar_cancel",
            confidence=0.90,
            suggested_tools=["calendar_cancel"],
            reason="regex: cancel meeting query",
        )

    if _CALENDAR_LIST_PATTERN.search(lower):
        return IntentClassification(
            intent="calendar_list",
            confidence=0.90,
            suggested_tools=["calendar_list"],
            reason="regex: list meetings query",
        )

    if _BOOKING_PATTERN.search(lower):
        return IntentClassification(
            intent="booking",
            confidence=0.90,
            suggested_tools=["calendar"],
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


def _meeting_params_from_text(text: str) -> dict[str, Any]:
    """Extract meeting params from free text (regex-only path, no session state)."""
    parsed = parse_meeting_request(text, timezone=settings.CALENDAR_TIMEZONE)
    return {
        "title": parsed.title,
        "start_at": parsed.start_at.isoformat() if parsed.start_at else None,
        "end_at": parsed.end_at.isoformat() if parsed.end_at else None,
        "duration_minutes": parsed.duration_minutes,
        "timezone": parsed.timezone,
        "attendees": parsed.attendees,
    }


_CONFIRM_YES = re.compile(
    r"^(yes|yeah|yep|y|ok|okay|sure|fine|go ahead|do it|confirm|please do|"
    r"looks? (good|great)|sounds (good|great|perfect)|that's? fine|that works|"
    r"that works for me|sounds? (good|great|perfect) to me|book (it|the meeting))$",
    re.IGNORECASE,
)
_CONFIRM_NO = re.compile(
    r"^(no|nope|n|nah|cancel|decline|don'?t|not now|never mind|no thanks|"
    r"that's? not (right|good)|wrong (time|day|date)|change it|different time)$",
    re.IGNORECASE,
)
_YES_LEADER = re.compile(
    r"^(yes|yeah|yep|ok|okay|sure|fine|go ahead|do it|confirm)$", re.IGNORECASE
)
_NO_LEADER = re.compile(r"^(no|nope|nah|cancel|decline|don'?t|not now)$", re.IGNORECASE)
_YES_TRAIL = re.compile(
    r"^(please|that's? (fine|good|great|perfect|ok)|that works|sounds? (good|great|perfect)|"
    r"looks? (good|great)|for sure|sure|go ahead|do it|book (it|the meeting)|please go ahead)$",
    re.IGNORECASE,
)
_NO_TRAIL = re.compile(
    r"^(thanks|thank you|not for now|cancel it|forget it|never mind|no thanks)$",
    re.IGNORECASE,
)
_AMBIGUOUS_WORDS = re.compile(
    r"\b(but|which|when|where|why|how|what|maybe|perhaps|actually)\b", re.IGNORECASE
)


def _classify_confirmation(user_input: str) -> bool | None:
    """Classify a reply to a pending booking proposal.

    Returns ``True`` (confirm), ``False`` (decline) or ``None`` (unrelated
    input, e.g. a free-form message that should be handled normally).
    """
    text = user_input.strip().lower().rstrip(".!?")
    if not text or len(text.split()) > 25:
        return None
    # Match the leading decision token; "yes, go ahead" / "no, cancel" both work.
    first = text.split(",")[0].strip()
    if _CONFIRM_YES.match(first):
        return True
    if _CONFIRM_NO.match(first):
        return False
    if _AMBIGUOUS_WORDS.search(first):
        return None
    # Tolerate a leading decision word followed by a short confirmation phrase,
    # e.g. "yes that works", "yes please", "no thanks", "yes sure".
    words = first.split()
    if not words:
        return None
    if _YES_LEADER.match(words[0]) and len(words) <= 6:
        rest = " ".join(words[1:])
        if not rest or _YES_TRAIL.match(rest):
            return True
        return None
    if _NO_LEADER.match(words[0]) and len(words) <= 6:
        rest = " ".join(words[1:])
        if not rest or _NO_TRAIL.match(rest):
            return False
        return None
    return None


class LLMPlanner(BasePlanner):
    def __init__(
        self,
        llm_provider: LLMProvider,
        fallback_intent: str = "general",
        pending_repo: Any = None,
        timezone: str | None = None,
    ) -> None:
        self._llm = llm_provider
        self._fallback_intent = fallback_intent
        self._pending_repo = pending_repo
        self._timezone = timezone or settings.CALENDAR_TIMEZONE

    async def create_plan(self, state: AgentState) -> list[PlanStep]:
        user_input = state["user_input"]
        start = time.perf_counter()

        pending_plan = await self._handle_pending_booking(state)
        if pending_plan is not None:
            return pending_plan

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

        context_str = _build_context_string(state)

        steps: list[PlanStep] = []
        for tool_name in tool_names:
            if tool_name in ("email_send", "send_email"):
                parameters = await self._draft_email_params(user_input, context=context_str)
            elif tool_name in _CALENDAR_TOOLS:
                parameters = self._calendar_params(tool_name, user_input, state)
            else:
                extractor = PARAM_EXTRACTORS.get(tool_name, lambda t: {"input": t})
                parameters = extractor(user_input)
            steps.append(PlanStep(
                tool_name=tool_name,
                parameters=parameters,
                reason=f"LLM classified intent as '{intent.intent}' (confidence: {intent.confidence:.2f})",
            ))

        return steps

    def _calendar_params(
        self, tool_name: str, user_input: str, state: AgentState
    ) -> dict[str, Any]:
        """Build calendar tool parameters, injecting session/user context.

        Attendees parsed from the message text are merged with structured emails
        carried through ``request_metadata["attendees"]`` (e.g. re-surfaced by the
        gateway after PII redaction of the free text).
        """
        structured_attendees = (
            (state.get("request_metadata") or {}).get("attendees") or []
        )
        if isinstance(structured_attendees, list):
            structured_attendees = [a for a in structured_attendees if isinstance(a, str)]
        else:
            structured_attendees = []

        if tool_name == "calendar":
            parsed = parse_meeting_request(user_input, timezone=self._timezone)
            params: dict[str, Any] = {
                "title": parsed.title,
                "start_at": parsed.start_at.isoformat() if parsed.start_at else None,
                "end_at": parsed.end_at.isoformat() if parsed.end_at else None,
                "duration_minutes": parsed.duration_minutes,
                "timezone": parsed.timezone,
                "attendees": _merge_attendees(parsed.attendees, structured_attendees),
                "needs_datetime": parsed.needs_datetime,
            }
        elif tool_name in ("schedule_demo", "schedule_meeting"):
            parsed = parse_meeting_request(user_input, timezone=self._timezone)
            params = {
                "title": parsed.title,
                "start_at": parsed.start_at.isoformat() if parsed.start_at else None,
                "end_at": parsed.end_at.isoformat() if parsed.end_at else None,
                "duration_minutes": parsed.duration_minutes,
                "timezone": parsed.timezone,
                "attendees": _merge_attendees(parsed.attendees, structured_attendees),
            }
        elif tool_name == "calendar_update":
            parsed = parse_meeting_request(
                user_input, timezone=self._timezone, prefer_last_time=True
            )
            ref = extract_meeting_ref(user_input, timezone=self._timezone)
            params = {
                **ref,
                "new_start_at": parsed.start_at.isoformat() if parsed.start_at else None,
                "new_end_at": parsed.end_at.isoformat() if parsed.end_at else None,
                "title": parsed.title,
                "attendees": _merge_attendees(parsed.attendees, structured_attendees),
                "timezone": parsed.timezone,
            }
        elif tool_name == "calendar_cancel":
            params = extract_meeting_ref(user_input, timezone=self._timezone)
        else:  # calendar_list
            params = {}

        params.setdefault("session_id", state.get("session_id"))
        params["user_id"] = state.get("user_id")
        return params

    async def _handle_pending_booking(self, state: AgentState) -> list[PlanStep] | None:
        """Resolve a pending booking proposal on the confirmation turn.

        Returns a plan when the user confirms or declines a pending proposal,
        or ``None`` when there is nothing pending / input is unrelated.
        """
        if self._pending_repo is None:
            return None
        session_id = state.get("session_id")
        if not session_id:
            return None

        proposal = await self._pending_repo.get_active(session_id)
        if proposal is None:
            return None

        decision = _classify_confirmation(state["user_input"])
        if decision is True:
            logger.info("pending_booking_confirmed", session_id=session_id, proposal_id=proposal.id)
            return [
                PlanStep(
                    tool_name="schedule_meeting",
                    parameters={
                        "session_id": session_id,
                        "user_id": state.get("user_id"),
                        "pending_id": proposal.id,
                    },
                    reason="User confirmed the pending booking proposal",
                )
            ]
        if decision is False:
            await self._pending_repo.clear(session_id)
            state["final_response"] = (
                "Okay, I won't book the meeting. Let me know if you'd like to try a different time."
            )
            logger.info("pending_booking_declined", session_id=session_id)
            return []

        return None

    async def _draft_email_params(self, user_input: str, context: str | None = None) -> dict[str, str]:
        """Draft email subject/body with the LLM for any scenario, falling back to
        heuristic extraction if the LLM is unavailable or returns nothing usable.

        ``context`` carries prior conversation (e.g. an assistant answer the user
        asks to send "above"/previously). When present and the user references it,
        the referenced content is used as the email body source.
        """
        fallback = _extract_email_params(user_input)

        # If the user is asking to send previously-shown content (\"send the above
        # info\", \"send this to ...\"), construct the email deterministically from
        # the referenced assistant message. This is far more reliable than asking
        # the LLM to guess what \"above\" points to.
        if context and _ABOVE_REFERENCE_PATTERN.search(user_input):
            enriched = _enrich_fallback_with_context(fallback, user_input, context)
            if enriched.get("body") and "requested below" in enriched["body"]:
                return enriched

        if self._llm is None:
            return _enrich_fallback_with_context(fallback, user_input, context)

        user_message = user_input
        if context:
            user_message = (
                f"{user_input}\n\nRecent conversation context (use it if the user "
                f"refers to 'above', 'this', 'the above information', 'previous', or "
                f"similar referring to it):\n{context}"
            )
        try:
            response = await self._llm.generate(
                system_prompt=EMAIL_DRAFT_SYSTEM_PROMPT,
                user_message=user_message,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("email_draft_llm_failed", error=str(exc))
            return _enrich_fallback_with_context(fallback, user_input, context)

        data = _extract_json_object(response.content)
        to = (data.get("to") or "").strip()
        subject = (data.get("subject") or "").strip()
        body = (data.get("body") or "").strip()
        valid = (not data) or (to and subject and body)
        if not valid:
            logger.warning(
                "email_draft_invalid",
                has_to=bool(to),
                has_subject=bool(subject),
                has_body=bool(body),
            )
            return fallback

        if to and subject and body and to != fallback.get("to"):
            return {"to": to, "subject": subject, "body": body}

        # If the user references prior conversation ("send the above info"), and the
        # LLM didn't produce a usable body, fall back to embedding the referenced
        # content directly so the email still carries what they meant to send.
        merged = _enrich_fallback_with_context(fallback, user_input, context)
        return {"to": to or merged.get("to", ""), "subject": subject or merged["subject"], "body": body or merged["body"]}

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

        agent_name = _CALENDAR_INTENT_AGENT.get(intent_name, intent_name)
        profile = agent_roster.get(agent_name) or agent_roster.get(agent_roster.agent_fallback)
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
