import json
import re
import uuid
from datetime import UTC, datetime
from typing import Any
from zoneinfo import ZoneInfo

import structlog

from shared.llm.base import LLMProvider

from orchestrator.graph.state import AgentState, PlanStep, _RESET_TURN
from orchestrator.planner.base import BasePlanner
from orchestrator.tools.registry import ToolRegistry
from orchestrator.context.builder import ContextBuilder

logger = structlog.get_logger(__name__)

_CALENDAR_TOOL_NAMES = frozenset({
    "calendar",
    "schedule_demo",
    "schedule_meeting",
    "calendar_list",
    "calendar_update",
    "calendar_cancel",
})


def create_receive_node() -> Any:
    async def receive(state: AgentState) -> dict[str, Any]:
        request_id = state.get("request_id") or str(uuid.uuid4())
        return {
            "request_id": request_id,
            "current_step_index": 0,
            "tool_results": [_RESET_TURN],
            "execution_log": [
                _RESET_TURN,
                {
                    "node": "receive",
                    "event": "request_received",
                    "user_input": state["user_input"],
                    "request_id": request_id,
                },
            ],
            "final_response": None,
            "error": None,
        }
    return receive


def create_context_node(context_builder: ContextBuilder) -> Any:
    async def build_context(state: AgentState) -> dict[str, Any]:
        ctx = await context_builder.build(
            user_input=state["user_input"],
            user_id=state.get("user_id"),
            session_id=state.get("session_id"),
        )
        return {
            "memory_context": ctx.get("memory_context", []),
            "document_context": ctx.get("document_context", []),
            "user_preferences": ctx.get("user_preferences", {}),
            "execution_log": [
                {
                    "node": "build_context",
                    "event": "context_loaded",
                    "memory_items": len(ctx.get("memory_context", [])),
                    "document_items": len(ctx.get("document_context", [])),
                }
            ],
        }
    return build_context


def create_plan_node(planner: BasePlanner) -> Any:
    async def plan(state: AgentState) -> dict[str, Any]:
        plan_steps = await planner.create_plan(state)
        result: dict[str, Any] = {
            "plan": plan_steps,
            "current_step_index": 0,
            "execution_log": [
                {
                    "node": "plan",
                    "event": "plan_created",
                    "num_steps": len(plan_steps),
                    "steps": [s["tool_name"] for s in plan_steps],
                }
            ],
        }
        if state.get("final_response") is not None:
            result["final_response"] = state["final_response"]
        return result
    return plan


def create_execute_node(
    tool_registry: ToolRegistry,
    approval_service=None,
) -> Any:
    async def execute(state: AgentState) -> dict[str, Any]:
        plan = state["plan"]
        index = state["current_step_index"]

        if index >= len(plan):
            return {
                "execution_log": [{"node": "execute", "event": "plan_complete", "total_steps": len(plan)}],
            }

        current_step = plan[index]
        tool = tool_registry.get(current_step["tool_name"])
        step_log = {
            "node": "execute",
            "event": "step_started",
            "step_index": index,
            "tool_name": current_step["tool_name"],
            "parameters": current_step["parameters"],
        }

        if tool is None:
            return {
                "execution_log": [
                    {**step_log, "event": "step_failed", "error": f"Tool '{{current_step['tool_name']}}' not found in registry"},
                ],
                "current_step_index": index + 1,
                "tool_results": [{"tool_name": current_step["tool_name"], "success": False, "error": "Tool not found"}],
            }

        # ── C4: HITL approval gate ──
        if approval_service is not None and approval_service.requires_approval(current_step["tool_name"]):
            try:
                decision = await approval_service.check_or_request(
                    session_id=state.get("session_id"),
                    user_id=state.get("user_id"),
                    user_input=state["user_input"],
                    tool_name=current_step["tool_name"],
                    parameters=current_step["parameters"],
                )
            except Exception as exc:
                logger.error("approval_gate_error", tool_name=current_step["tool_name"], error=str(exc))
                decision = None

            if decision is None or not decision.approved:
                message = (
                    decision.message
                    if decision is not None
                    else "This action requires approval before it can be executed."
                )
                logger.info(
                    "approval_pending",
                    tool_name=current_step["tool_name"],
                    task_id=getattr(decision, "task_id", None),
                )
                return {
                    "awaiting_approval": True,
                    "approval_task_id": getattr(decision, "task_id", None),
                    "final_response": message,
                    "execution_log": [
                        {
                            **step_log,
                            "event": "step_awaiting_approval",
                            "approval_task_id": getattr(decision, "task_id", None),
                        }
                    ],
                }

        params = dict(current_step["parameters"])
        if current_step["tool_name"] in _CALENDAR_TOOL_NAMES:
            params.setdefault("session_id", state.get("session_id"))
            params["user_id"] = state.get("user_id")
        result = await tool.invoke(params)
        return {
            "current_step_index": index + 1,
            "awaiting_approval": False,
            "approval_task_id": None,
            "tool_results": [
                {
                    "tool_name": current_step["tool_name"],
                    "step_index": index,
                    **result,
                }
            ],
            "execution_log": [
                {
                    **step_log,
                    "event": "step_completed",
                    "success": result.get("success", False),
                }
            ],
        }

    return execute


def create_tool_invoke_node(tool_registry: ToolRegistry) -> Any:
    async def tool_invoke(state: AgentState) -> dict[str, Any]:
        plan = state["plan"]
        index = state["current_step_index"]

        if index >= len(plan):
            return {
                "execution_log": [{"node": "tool_invoke", "event": "no_steps_remaining"}],
            }

        current_step = plan[index]
        tool = tool_registry.get(current_step["tool_name"])

        if tool is None:
            return {
                "tool_results": [
                    {"tool_name": current_step["tool_name"], "success": False, "error": "Tool not found in registry"}
                ],
                "execution_log": [
                    {"node": "tool_invoke", "event": "tool_not_found", "tool_name": current_step["tool_name"]}
                ],
            }

        result = await tool.invoke(current_step["parameters"])
        return {
            "tool_results": [
                {
                    "tool_name": current_step["tool_name"],
                    "step_index": index,
                    **result,
                }
            ],
            "execution_log": [
                {
                    "node": "tool_invoke",
                    "event": "tool_executed",
                    "tool_name": current_step["tool_name"],
                    "success": result.get("success", False),
                }
            ],
        }

    return tool_invoke


RESPONSE_SYSTEM_PROMPT = (
    "You are a helpful multilingual AI employee assistant for an enterprise platform. "
    "You have access to tool results that provide factual information about the company. "
    "Rules: "
    "1. Use the provided tool results as your primary source to answer the user's question. "
    "2. If tool results are empty, irrelevant, or insufficient, use your own general knowledge to answer directly. "
    "You know basic facts like today's date, geography, history, math, definitions, and common knowledge. "
    "3. If a user asks about past conversation details, reference the conversation history provided in the prompt. "
    "4. Keep responses friendly, professional, and under 3 paragraphs when possible. "
    "5. If the user's request requires a human, confirm the escalation has been initiated. "
    "6. IMPORTANT: Detect the language the user is writing in and ALWAYS respond in the SAME language. "
    "If the user writes in Hindi, respond in Hindi. If in Kannada, respond in Kannada. If in Tamil, Tamil. "
    "If in English, English. Match the user's language exactly. "
    "Do NOT translate — respond in the native language and script of the user's message."
)

PERSONA_PROMPTS: dict[str, str] = {
    "sales": (
        "You are a Sales AI Employee. You help prospects and customers with product information, pricing, "
        "and purchase decisions. Be enthusiastic, knowledgeable, and solution-oriented. "
        "When presenting pricing, highlight the value proposition of each tier. "
        "Offer to schedule a demo if appropriate. End with a helpful next step. "
        "IMPORTANT: Respond in the same language as the user's message."
    ),
    "support": (
        "You are a Support AI Employee. You help customers with order tracking, returns, troubleshooting, "
        "and account issues. Be empathetic, patient, and thorough. If you cannot resolve the issue, "
        "offer to escalate to a human agent. Always confirm that the customer's concern is addressed. "
        "IMPORTANT: Respond in the same language as the user's message."
    ),
    "booking": (
        "You are a Booking AI Employee. You help users schedule demos, appointments, and meetings. "
        "Be efficient and precise with dates, times, and availability. Confirm all booking details "
        "before finalizing. "
        "If the requested time is unavailable, state clearly that the slot is taken and offer the "
        "concrete alternative slots listed in available_slots (e.g. 'the next free slot is at 5:00 PM'). "
        "Never discuss pricing, product features, demos, or unrelated topics unless the user asks. "
        "Never invent dates, times, or email addresses. "
        "IMPORTANT: Respond in the same language as the user's message."
    ),
    "general": RESPONSE_SYSTEM_PROMPT,
    "complaint": (
        "You are handling a customer complaint. Be deeply empathetic, acknowledge their frustration, "
        "and confirm that their concern has been escalated to a human agent who will address it personally. "
        "Do not make promises about refunds or compensation — the human agent will handle that. "
        "IMPORTANT: Respond in the same language as the user's message."
    ),
    "escalate": (
        "The user has requested to speak with a human agent. Confirm that their request "
        "has been received and that a human agent will review their conversation history "
        "and respond shortly. Be courteous and reassuring. "
        "IMPORTANT: Respond in the same language as the user's message."
    ),
    "email": (
        "You are an Email AI Employee. You help users send and read emails. "
        "When sending emails, confirm the recipient, subject, and body before sending. "
        "When listing emails, summarize the most relevant ones. Be professional and concise. "
        "IMPORTANT: Respond in the same language as the user's message."
    ),
}


def create_respond_node(llm_provider: LLMProvider | None = None, memory_client: Any = None) -> Any:
    async def respond(state: AgentState) -> dict[str, Any]:
        # Helper to store messages to memory service for next-turn context
        async def _persist_turn(sid: str, uid: str, user_msg: str, assistant_msg: str):
            if memory_client is None or not sid:
                return
            try:
                await memory_client.add_message(sid, "user", user_msg, user_id=uid)
                await memory_client.add_message(sid, "assistant", assistant_msg, user_id=uid)
            except Exception:
                pass
        if state.get("final_response") is not None:
            logger.info(
                "respond_escalation_short_circuit",
                request_id=state.get("request_id"),
                response_length=len(state["final_response"]),
            )
            await _persist_turn(state.get("session_id"), state.get("user_id"), state.get("user_input", ""), state["final_response"])
            return {
                "execution_log": [
                    {
                        "node": "respond",
                        "event": "escalation_response_used",
                        "response_length": len(state["final_response"]),
                    }
                ],
            }

        plan = state.get("plan", [])
        tool_results = state.get("tool_results", [])

        if not tool_results and not plan:
            msg = "I received your request but I'm not sure how to help with that. Could you provide more details?"
            logger.info("respond_no_results", request_id=state.get("request_id"))
            await _persist_turn(state.get("session_id"), state.get("user_id"), state.get("user_input", ""), msg)
            return {
                "final_response": msg,
                "execution_log": [
                    {"node": "respond", "event": "no_tool_results", "response_length": len(msg)}
                ],
            }

        tool_context = _build_tool_context(tool_results)
        user_message = _build_respond_prompt(
            state["user_input"],
            tool_context,
            state.get("memory_context", []),
        )

        logger.info(
            "respond_assembling_context",
            request_id=state.get("request_id"),
            tool_count=len(tool_results),
            prompt_length=len(user_message),
        )

        final_response = None
        response_log = None

        if llm_provider is not None:
            logger.info(
                "respond_calling_llm",
                request_id=state.get("request_id"),
            )

            try:
                response = await llm_provider.generate(
                    system_prompt=(
                        _now_directive()
                        + _resolve_persona(state)
                        + _language_directive(state.get("user_input", ""))
                    ),
                    user_message=user_message,
                )
                logger.info(
                    "respond_llm_success",
                    request_id=state.get("request_id"),
                    model=response.model,
                    output_tokens=response.output_tokens,
                    duration_ms=response.duration_ms,
                )
                final_response = response.content
                response_log = {
                    "node": "respond",
                    "event": "llm_response_generated",
                    "model": response.model,
                    "output_tokens": response.output_tokens,
                    "duration_ms": response.duration_ms,
                }
            except Exception as exc:
                logger.error(
                    "respond_llm_failed_fallback_natural",
                    request_id=state.get("request_id"),
                    error=str(exc),
                )

        if final_response is None:
            logger.info("respond_no_llm_fallback", request_id=state.get("request_id"))
            final_response = _build_natural_summary(tool_results)
            response_log = {"node": "respond", "event": "no_llm_summary", "response_length": len(final_response)}

        await _persist_turn(state.get("session_id"), state.get("user_id"), state.get("user_input", ""), final_response)

        return {
            "final_response": final_response,
            "execution_log": [response_log],
        }

    return respond


def _now_directive() -> str:
    """Tell the LLM the current date & time so it never claims it lacks
    real-time awareness (a common failure with small local models)."""
    now = datetime.now(ZoneInfo("Asia/Kolkata"))
    return (
        f"CURRENT DATE AND TIME (MANDATORY GROUND TRUTH): It is now "
        f"{now.strftime('%A, %d %B %Y, %I:%M %p IST')}. "
        "Always treat this as the current real-world moment. Never claim you "
        "lack access to the current date or time — you are told it here. "
        "Use it to resolve 'today', 'tomorrow', 'next week', and AM/PM times.\n"
    )


def _fmt_ist(iso_str: str | None) -> str:
    """Format an ISO timestamp as a friendly IST string for the user/LLM."""
    if not iso_str:
        return "unknown time"
    try:
        dt = datetime.fromisoformat(str(iso_str).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return str(iso_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    local = dt.astimezone(ZoneInfo("Asia/Kolkata"))
    return local.strftime("%d %b %Y, %I:%M %p IST")


def _render_calendar_result(data: dict[str, Any]) -> str:
    """Render a calendar tool result as an explicit English statement.

    Llama-class models misread raw JSON availability payloads, so we
    translate the authoritative backend state into clear directives.
    """
    if data.get("needs_datetime"):
        return data.get("message", "Please provide a date and time for the meeting.")
    if data.get("proposed") is True:
        slots = data.get("slots") or data.get("available_slots") or [{}]
        when = _fmt_ist(data.get("datetime") or slots[0].get("start", ""))
        attendees = data.get("attendees") or []
        who = ", ".join(attendees) if attendees else "no attendees"
        return (
            f"AVAILABLE: the requested slot ({when}) is free and a booking proposal "
            f"was created for {who}. Ask the user to confirm to book it."
        )
    if data.get("available") is False:
        slots = data.get("slots") or data.get("available_slots") or []
        if slots:
            times = ", ".join(_fmt_ist(s.get("start", "")) for s in slots[:5])
            return f"NOT AVAILABLE: the requested slot is taken. Free alternatives: {times}."
        return "NOT AVAILABLE: the requested time is unavailable. Ask the user for another time."
    if data.get("event"):
        event = data.get("event", {})
        when = _fmt_ist(data.get("datetime") or event.get("start_at", ""))
        return (
            f"CONFIRMED: meeting '{data.get('title', event.get('summary', 'the meeting'))}' "
            f"scheduled at {when} with attendees "
            f"{data.get('attendees') or event.get('attendees') or []}. "
            f"Calendar link: {event.get('link') or 'not provided'}."
        )
    return json.dumps(data, default=str)


def _build_tool_context(tool_results: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for result in tool_results:
        tool_name = result.get("tool_name", "unknown")
        if result.get("success"):
            data = result.get("data", {})
            try:
                if tool_name in ("calendar", "schedule_demo", "schedule_meeting"):
                    parts.append(f"[{tool_name}]: {_render_calendar_result(data)}")
                else:
                    parts.append(f"[{tool_name}]: {json.dumps(data, default=str)}")
            except (TypeError, ValueError):
                parts.append(f"[{tool_name}]: {str(data)}")
        else:
            error = result.get("error", result.get("data", {}).get("error", "Unknown error"))
            parts.append(f"[{tool_name} - FAILED]: {error}")
    return "\n".join(parts)


def _build_respond_prompt(
    user_input: str,
    tool_context: str,
    memory_context: list[dict[str, Any]],
    max_len: int = 3000,
) -> str:
    """Assemble the respond-node prompt.

    The current question and the authoritative tool results must never be
    truncated. History is auxiliary: trim it to the remaining budget so a long
    conversation can't crowd out the tool results (which caused the LLM to
    lose the booking state and hallucinate/ramble).

    Stale PII (e.g. email addresses from an earlier turn) is redacted out of the
    history so the LLM doesn't resurrect old contact details that are unrelated
    to the current request.
    """
    history_lines: list[str] = []
    for msg in (memory_context or []):
        role = msg.get("role", "user")
        content = msg.get("content", "")
        # Redact stale emails from all history (past user, assistant, or memory
        # echoes) unless the user re-mentions them in the current turn.
        content = _redact_history_pii(content, user_input)
        history_lines.append(f"[{role}]: {content}")
    history = "\n".join(history_lines)
    core = f"User's question: {user_input}\n\nTool results:\n{tool_context}"
    if history:
        history_budget = max(0, max_len - len(core))
        history = history[-history_budget:] if history_budget > 0 else ""
        return f"{history}\n\n{core}"
    return core


_EMAIL_ADDR_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


def _redact_history_pii(content: str, current_user_input: str) -> str:
    """Redact email addresses from past user messages unless the user re-mentions
    the same email in the current turn. Prevents stale contacts from leaking into
    the LLM context."""
    current_emails = set(_EMAIL_ADDR_RE.findall(current_user_input))
    out = content
    for email in _EMAIL_ADDR_RE.findall(content):
        if email not in current_emails:
            out = out.replace(email, "[EMAIL]")
    return out


def _build_natural_summary(tool_results: list[dict[str, Any]]) -> str:
    """Fallback response when LLM is unavailable. Summarizes tool results in plain English."""
    parts: list[str] = []
    for result in tool_results:
        tool_name = result.get("tool_name", "")
        if not result.get("success"):
            if tool_name in ("send_email", "email_send"):
                error = result.get("error") or (result.get("data") or {}).get("error", "unknown error")
                parts.append(f"The email could not be sent: {error}")
            else:
                parts.append(f"Sorry, the {tool_name} operation failed.")
            continue
        data = result.get("data", {})

        if tool_name == "calculator":
            parts.append(f"{data.get('expression', '')} = {data.get('result', '')}")
        elif tool_name in ("search_documents", "search_pricing"):
            items = data.get("results", [])
            if items:
                names = [r.get("title", r.get("tier", "")) for r in items[:3]]
                parts.append(f"Found: {', '.join(names)}")
        elif tool_name == "lookup_order":
            order = data
            parts.append(
                f"Order {order.get('order_id', '')}: {order.get('status', 'unknown')}"
                f", delivery by {order.get('estimated_delivery', 'unknown')}"
            )
        elif tool_name == "get_weather":
            parts.append(
                f"Weather in {data.get('location', '')}: "
                f"{data.get('temperature', '?')}{chr(176)}{data.get('unit', 'celsius')[0].upper()}, {data.get('conditions', '')}"
            )
        elif tool_name == "calendar":
            if data.get("needs_datetime"):
                parts.append(data.get("message", "Please provide a date and time for the meeting."))
                continue
            slots = data.get("available_slots", [])
            if not data.get("available") and slots:
                times = [_fmt_ist(s.get("start", "")) for s in slots[:5]]
                parts.append(
                    "That slot is unavailable. Free slots nearby: "
                    + ", ".join(times)
                )
            elif slots:
                dates = [_fmt_ist(s.get("start", "")) for s in slots[:3]]
                parts.append(f"Available on: {', '.join(dates)}")
            else:
                parts.append("That time is unavailable. Please pick another time.")
        elif tool_name in ("schedule_demo", "schedule_meeting"):
            if data.get("needs_datetime"):
                parts.append(data.get("message", "Please provide a date and time for the meeting."))
                continue
            event = data.get("event", {})
            parts.append(
                f"Scheduled: {event.get('summary', data.get('title', ''))} "
                f"on {_fmt_ist(data.get('datetime', event.get('start_at', '')))}"
            )
        elif tool_name == "calendar_list":
            events = data.get("events", [])
            if events:
                parts.append(
                    f"Found {len(events)} upcoming event(s): "
                    + ", ".join(
                        f"{e.get('summary', e.get('title', ''))} at "
                        f"{_fmt_ist(e.get('start_at', e.get('date', '')))}"
                        for e in events[:5]
                    )
                )
            else:
                parts.append("No upcoming events found on your calendar.")
        elif tool_name == "calendar_update":
            parts.append(
                f"Rescheduled '{data.get('title', 'the meeting')}' to "
                f"{_fmt_ist(data.get('start_at', ''))}."
            )
        elif tool_name == "calendar_cancel":
            parts.append(
                f"Cancelled '{data.get('title', 'the meeting')}'."
            )
        elif tool_name == "transfer_to_human":
            parts.append(data.get("message", "Escalated to human agent."))
        elif tool_name in ("send_email", "email_send"):
            parts.append(
                f"Email sent to {data.get('to', 'recipient')} "
                f"with subject '{data.get('subject', '')}'. Status: {data.get('status', 'sent')}"
            )
        elif tool_name == "email_list":
            messages = data.get("messages", [])
            if messages:
                parts.append(f"Found {len(messages)} email(s) in the inbox.")
            else:
                parts.append("No emails found in the inbox.")

    return "\n".join(parts) if parts else "I received your request but found no results to share."


def _resolve_persona(state: AgentState) -> str:
    from orchestrator.agents.roster import agent_roster

    plan = state.get("plan", [])
    if not plan:
        if state.get("final_response"):
            return agent_roster.get("escalate").system_prompt
        return agent_roster.get("general").system_prompt

    tool_names = [s["tool_name"] for s in plan]
    return agent_roster.resolve_for_tools(tool_names).system_prompt


_SCRIPT_HINTS = [
    ("hi-MM", "\u0900-\u097F", "Hindi"),
    ("ta-MM", "\u0B80-\u0BFF", "Tamil"),
    ("te-MM", "\u0C00-\u0C7F", "Telugu"),
    ("kn-MM", "\u0C80-\u0CFF", "Kannada"),
    ("ml-MM", "\u0D00-\u0D7F", "Malayalam"),
    ("bn-MM", "\u0980-\u09FF", "Bengali"),
    ("mr-MM", "\u0900-\u097F", "Marathi"),
    ("gu-MM", "\u0A80-\u0AFF", "Gujarati"),
    ("pa-MM", "\u0A00-\u0A7F", "Punjabi"),
    ("ur-MM", "\u0600-\u06FF", "Urdu"),
    ("ar-MM", "\u0600-\u06FF", "Arabic"),
    ("zh-MM", "\u4E00-\u9FFF", "Chinese"),
    ("ja-MM", "\u3040-\u30FF", "Japanese"),
    ("ko-MM", "\uAC00-\uD7AF", "Korean"),
]


def _language_directive(user_input: str) -> str:
    """Return a firm, explicit directive forcing the LLM to reply in the same
    language/script the user wrote in. Reliable because it overrides the model's
    ambiguous auto-detection with a concrete instruction."""
    if not user_input:
        return ""
    ascii_chars = sum(1 for c in user_input if ord(c) < 128)
    total = len(user_input)
    # If predominantly ASCII/Latin, force English.
    if total and ascii_chars / total > 0.8:
        return (
            " LANGUAGE RULE (MANDATORY): The user wrote in English. You MUST write your "
            "entire reply in English. Do NOT reply in Hindi, Devanagari, or any other "
            "language/script. Only English."
        )
    for code, span, name in _SCRIPT_HINTS:
        if any([c in span for c in user_input]):
            return (
                f" LANGUAGE RULE (MANDATORY): The user wrote in {name}. You MUST write "
                f"your entire reply in {name}. Do NOT switch to English or any other language."
            )
    return ""