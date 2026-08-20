"""Real-data webhook endpoints the Samvaad agent's Tools / On-Start / On-End
hooks call.

These thin endpoints reuse the platform's existing production services so the
hosted Samvaad agent operates on real data instead of mock data. The agent side
is authored in the Sarvam dashboard (Tools -> Webhook); the base URL points at
this orchestrator (or through the gateway proxy path
``/api/orchestrator/samvaad/tools/...``).
"""

from datetime import UTC, datetime
from typing import Any

import structlog
from fastapi import APIRouter, Depends, Header, HTTPException, Query

from orchestrator.config import settings
from orchestrator.container import (
    _build_rag_client,
    get_calendar_service,
    get_escalation_service,
    get_memory_client,
    get_order_service,
    get_pricing_service,
    get_task_service,
)
from orchestrator.services.calendar.base import CalendarEventDraft

router = APIRouter(prefix="/api/samvaad/tools", tags=["Samvaad Tools"])
logger = structlog.get_logger(__name__)

# Safe, read-only tools always available. The real-action tools (email/send,
# calendar/schedule, human/transfer, calendar/update, tasks/manage) are blocked
# by default and only run after they are explicitly listed in
# SAMVAAD_TOOLS_ALLOWLIST — the confirmation gate.
DEFAULT_ALLOWED_TOOLS = frozenset(
    {
        "on-start/context",
        "on-end/record",
        "calendar/availability",
        "email/search",
        "search/documents",
        "orders/lookup",
        "pricing/search",
    }
)
ACTION_TOOLS = frozenset(
    {
        "email/send",
        "calendar/schedule",
        "human/transfer",
        "calendar/update",
        "tasks/manage",
    }
)


def _allowed_tools() -> frozenset[str]:
    raw = settings.SAMVAAD_TOOLS_ALLOWLIST.strip()
    parsed: list[str] = []
    if raw:
        try:
            parsed = [str(t).strip() for t in __import__("json").loads(raw)]
        except Exception:
            raise HTTPException(
                status_code=500,
                detail="SAMVAAD_TOOLS_ALLOWLIST is not a valid JSON array",
            ) from None
    # Read-only defaults always stay available; the allowlist only adds the
    # explicit action tools on top of them.
    return DEFAULT_ALLOWED_TOOLS | frozenset(t for t in parsed if t)


def _require_tool_auth(
    x_api_key: str | None = Header(default=None),
    x_samvaad_secret: str | None = Header(default=None),
    token: str | None = Query(default=None),
) -> None:
    """Webhook auth: header takes precedence, then the ?token= query fallback."""
    if settings.SAMVAAD_TOOL_SECRET:
        if x_samvaad_secret == settings.SAMVAAD_TOOL_SECRET:
            return
        if token and token == settings.SAMVAAD_TOOL_SECRET:
            return
        raise HTTPException(status_code=401, detail="Invalid X-Samvaad-Secret / token")
    if not x_api_key:
        raise HTTPException(status_code=401, detail="Missing X-API-Key header")


def _require_tool_enabled(tool: str) -> None:
    if tool in _allowed_tools():
        return
    raise HTTPException(
        status_code=403,
        detail=(
            f"Tool '{tool}' is disabled. Add it to SAMVAAD_TOOLS_ALLOWLIST in "
            "your orchestrator .env to enable it."
        ),
    )


def tools_gate_status() -> dict[str, Any]:
    """Allowed/blocked webhook tools, for the /api/samvaad/status report."""
    allowed = _allowed_tools()
    all_tools = sorted(DEFAULT_ALLOWED_TOOLS | ACTION_TOOLS)
    return {
        "allowed": sorted(t for t in all_tools if t in allowed),
        "blocked": sorted(t for t in all_tools if t not in allowed),
    }


def _iso_to_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value))
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid datetime: {value}") from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


@router.post("/on-start/context", dependencies=[Depends(_require_tool_auth)])
async def tool_on_start_context(payload: dict[str, Any]) -> dict[str, Any]:
    _require_tool_enabled("on-start/context")
    """Seed agent variables for the Samvaad On-Start hook.

    Body: ``{"user_identifier": str, "session_id": str?}``
    Returns an ``agent_variables`` object (name, language, recent meetings).
    """
    user_id = payload.get("user_identifier") or payload.get("user_id")
    session_id = payload.get("session_id")
    variables: dict[str, Any] = {"user_identifier": user_id or ""}

    memory = get_memory_client()
    if user_id:
        profile = await memory.get_profile(user_id)
        if profile:
            variables["user_name"] = profile.get("display_name") or ""
            variables["preferences"] = profile.get("preferences") or {}
    if session_id:
        lang = await memory.get_session_language(session_id)
        if lang and lang != "unknown":
            variables["language"] = lang

    try:
        calendar = get_calendar_service()
        if calendar.enabled:
            start = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
            meetings = await calendar.list_meetings(start=start, status="scheduled", limit=10)
            # Trim to a small, fixed size so the context the harness re-sends on
            # every turn stays small (large payloads inflate per-turn LLM cost).
            trimmed: list[dict[str, Any]] = []
            for m in (meetings.get("meetings") or [])[:5]:
                trimmed.append(
                    {
                        "title": str(m.get("title") or "")[:80],
                        "start_at": str(m.get("start_at") or "")[:30],
                        "end_at": str(m.get("end_at") or "")[:30],
                        "status": str(m.get("status") or ""),
                    }
                )
            variables["today_meetings"] = trimmed
    except Exception as exc:
        logger.warning("samvaad_tools_on_start_calendar_failed", error=str(exc))

    logger.info("samvaad_tool_on_start_context", user_id=user_id)
    return {"agent_variables": variables}


@router.post("/calendar/availability", dependencies=[Depends(_require_tool_auth)])
async def tool_calendar_availability(payload: dict[str, Any]) -> dict[str, Any]:
    _require_tool_enabled("calendar/availability")
    """Check real calendar availability for a window.

    Body: ``{"start_at": str, "end_at": str?, "duration_minutes": int?,
    "timezone": str?}``
    """
    start = _iso_to_dt(payload.get("start_at"))
    if start is None:
        return {
            "success": False,
            "error": "start_at is required (ISO-8601)",
            "message": "I couldn't check availability — please provide a start time.",
        }
    end = _iso_to_dt(payload.get("end_at")) or start
    try:
        result = await get_calendar_service().check_availability(
            start_at=start,
            end_at=end,
            timezone=payload.get("timezone"),
            duration_minutes=int(payload.get("duration_minutes") or 30),
        )
    except Exception as exc:
        logger.warning("samvaad_tool_availability_error", error=str(exc))
        return {
            "success": False,
            "error": f"calendar unavailable: {exc}",
            "message": "I couldn't reach the calendar right now.",
        }
    return {"success": True, **result}


@router.post("/calendar/schedule", dependencies=[Depends(_require_tool_auth)])
async def tool_calendar_schedule(payload: dict[str, Any]) -> dict[str, Any]:
    _require_tool_enabled("calendar/schedule")
    """Create a real calendar event.

    Body: ``{"session_id": str, "user_id": str?, "title": str,
    "start_at": str, "end_at": str, "timezone": str?, "attendees": [str]?,
    "description": str?}``
    """
    start = _iso_to_dt(payload.get("start_at"))
    end = _iso_to_dt(payload.get("end_at"))
    if start is None or end is None:
        return {
            "success": False,
            "error": "start_at and end_at are required",
            "message": "I need a start and end time to schedule that.",
        }
    draft = CalendarEventDraft(
        title=payload.get("title") or "AI Employee meeting",
        start_at=start,
        end_at=end,
        timezone=payload.get("timezone") or settings.CALENDAR_TIMEZONE,
        attendees=list(payload.get("attendees") or []),
        description=payload.get("description"),
    )
    try:
        result = await get_calendar_service().create_meeting(
            draft=draft,
            session_id=payload.get("session_id"),
            user_id=payload.get("user_id"),
        )
    except Exception as exc:
        logger.warning("samvaad_tool_schedule_error", error=str(exc))
        return {
            "success": False,
            "error": f"calendar schedule failed: {exc}",
            "message": "I couldn't schedule that on the calendar right now.",
        }
    return {"success": bool(result.get("success")), **result}


@router.post("/calendar/update", dependencies=[Depends(_require_tool_auth)])
async def tool_calendar_update(payload: dict[str, Any]) -> dict[str, Any]:
    _require_tool_enabled("calendar/update")
    """Reschedule, edit, or cancel an existing meeting.

    Body: ``{"action": "reschedule"|"cancel"|"edit", "meeting_id": str?,
    "session_id": str?, "start_at": str? (ISO reference for matching),
    "new_start_at": str?, "new_end_at": str?, "title": str?,
    "attendees": [str]?, "timezone": str?}``
    """
    action = str(payload.get("action") or "reschedule").lower()
    if action not in ("reschedule", "cancel", "edit"):
        return {
            "success": False,
            "error": "action must be one of: reschedule, cancel, edit",
            "message": "I couldn't update that meeting — the action wasn't clear.",
        }

    meeting_id = payload.get("meeting_id")
    existing_title = None
    if not meeting_id:
        ref_date = _iso_to_dt(payload.get("start_at"))
        try:
            matched = await get_calendar_service().match_meetings(
                session_id=payload.get("session_id"), ref_date=ref_date
            )
        except Exception as exc:
            logger.warning("samvaad_tool_update_match_error", error=str(exc))
            return {
                "success": False,
                "error": f"calendar lookup failed: {exc}",
                "message": "I couldn't find that meeting right now.",
            }
        matches = matched.get("matches", [])
        if not matches:
            return {
                "success": True,
                "data": {"updated": False, "message": "No matching meeting found."},
            }
        if len(matches) > 1:
            return {
                "success": True,
                "data": {
                    "updated": False,
                    "needs_disambiguation": True,
                    "meetings": matches,
                    "message": "Which meeting did you want to update?",
                },
            }
        meeting_id = matches[0]["id"]
        existing_title = matches[0].get("title")
    else:
        matched = await get_calendar_service().match_meetings(meeting_id=meeting_id)
        if matched.get("matches"):
            existing_title = matched["matches"][0].get("title")

    if action == "cancel":
        try:
            result = await get_calendar_service().cancel_meeting(meeting_id)
        except Exception as exc:
            logger.warning("samvaad_tool_cancel_error", error=str(exc))
            return {
                "success": False,
                "error": f"calendar cancel failed: {exc}",
                "message": "I couldn't cancel that meeting right now.",
            }
        return {
            "success": bool(result.get("success")),
            "data": {**result.get("meeting", {}), "cancelled": True},
        }

    new_start = _iso_to_dt(payload.get("new_start_at"))
    new_end = _iso_to_dt(payload.get("new_end_at"))
    if new_start is None or new_end is None:
        return {
            "success": False,
            "error": "new_start_at and new_end_at are required (ISO-8601)",
            "message": "I need the new start and end times to reschedule.",
        }

    title = payload.get("title") or existing_title or "AI Employee meeting"
    draft = CalendarEventDraft(
        title=title,
        start_at=new_start,
        end_at=new_end,
        timezone=payload.get("timezone") or settings.CALENDAR_TIMEZONE,
        attendees=list(payload.get("attendees") or []),
        description=payload.get("description"),
    )
    try:
        result = await get_calendar_service().update_meeting(meeting_id, draft)
    except Exception as exc:
        logger.warning("samvaad_tool_update_error", error=str(exc))
        return {
            "success": False,
            "error": f"calendar update failed: {exc}",
            "message": "I couldn't update that meeting right now.",
        }
    return {"success": bool(result.get("success")), **result}


@router.post("/email/send", dependencies=[Depends(_require_tool_auth)])
async def tool_email_send(payload: dict[str, Any]) -> dict[str, Any]:
    _require_tool_enabled("email/send")
    """Send an email via the configured SMTP/Gmail stack.

    Body: ``{"to": str, "subject": str, "body": str}``
    """
    to = payload.get("to")
    subject = payload.get("subject")
    body = payload.get("body")
    if not to or not subject or not body:
        return {
            "success": False,
            "error": "to, subject and body are required",
            "message": "I need a recipient, subject and message to send that email.",
        }

    from orchestrator.services.gmail_client import EmailClient

    client = EmailClient()
    if not client.enabled:
        return {
            "success": False,
            "error": "Email not configured (EMAIL_ENABLED, EMAIL_ADDRESS, EMAIL_PASSWORD)",
            "message": "I'm not set up to send email right now.",
        }
    import asyncio

    try:
        result = await asyncio.to_thread(client.send_message, to, subject, body)
        return {"success": True, "result": result}
    except Exception as exc:
        logger.error("samvaad_tool_email_failed", error=str(exc))
        return {
            "success": False,
            "error": str(exc),
            "message": "I couldn't send that email just now.",
        }


@router.post("/email/search", dependencies=[Depends(_require_tool_auth)])
async def tool_email_search(payload: dict[str, Any]) -> dict[str, Any]:
    _require_tool_enabled("email/search")
    """Search the configured inbox via IMAP.

    Body: ``{"query": str?, "max_results": int?, "with_body": bool?}``
    ``query`` is an IMAP search string (e.g. ``FROM "boss@example.com"``);
    ``with_body`` optionally pulls full message bodies for each hit.
    """
    from orchestrator.services.gmail_client import EmailClient

    client = EmailClient()
    if not client.enabled:
        return {
            "success": False,
            "error": "Email not configured (EMAIL_ENABLED, EMAIL_ADDRESS, EMAIL_PASSWORD)",
            "message": "I couldn't search email — it's not configured.",
        }

    max_results = int(payload.get("max_results") or 10)
    query = payload.get("query") or ""
    import asyncio

    try:
        messages = await asyncio.to_thread(client.list_messages, max_results, query)
        if payload.get("with_body") and messages:
            for msg in messages:
                try:
                    full = await asyncio.to_thread(client.get_message, msg["id"])
                    msg["snippet"] = full.get("snippet") or msg.get("snippet", "")
                    msg["body"] = full.get("body")
                except Exception as exc:
                    logger.warning(
                        "samvaad_email_body_failed",
                        message_id=msg["id"],
                        error=str(exc),
                    )
        return {"success": True, "messages": messages, "count": len(messages)}
    except Exception as exc:
        logger.error("samvaad_tool_email_search_failed", error=str(exc))
        return {
            "success": False,
            "error": str(exc),
            "message": "I couldn't search email right now.",
        }


@router.post("/search/documents", dependencies=[Depends(_require_tool_auth)])
async def tool_search_documents(payload: dict[str, Any]) -> dict[str, Any]:
    _require_tool_enabled("search/documents")
    """Search the platform knowledge base (RAG)."""
    query = payload.get("query")
    if not query:
        return {
            "success": False,
            "error": "query is required",
            "message": "I need a search query to look that up.",
        }
    try:
        results = await _build_rag_client().search(
            query=query,
            top_k=int(payload.get("top_k") or 5),
        )
    except Exception as exc:
        logger.warning("samvaad_tool_rag_error", error=str(exc))
        return {
            "success": False,
            "error": f"document search failed: {exc}",
            "message": "I couldn't search the documents right now.",
        }
    return {"success": True, "results": results}


@router.post("/orders/lookup", dependencies=[Depends(_require_tool_auth)])
async def tool_orders_lookup(payload: dict[str, Any]) -> dict[str, Any]:
    _require_tool_enabled("orders/lookup")
    """Look up an order status."""
    order_id = payload.get("order_id")
    if not order_id:
        return {
            "success": False,
            "error": "order_id is required",
            "message": "I need an order id to look that up.",
        }
    try:
        result = await get_order_service().lookup_order(order_id)
    except Exception as exc:
        logger.warning("samvaad_tool_order_error", error=str(exc))
        return {
            "success": False,
            "error": f"order lookup failed: {exc}",
            "message": "I couldn't find that order right now.",
        }
    return {"success": True, "result": result}


@router.post("/pricing/search", dependencies=[Depends(_require_tool_auth)])
async def tool_pricing_search(payload: dict[str, Any]) -> dict[str, Any]:
    _require_tool_enabled("pricing/search")
    """Search pricing tiers."""
    query = payload.get("query")
    if not query:
        return {
            "success": False,
            "error": "query is required",
            "message": "I need a query to search pricing.",
        }
    try:
        results = await get_pricing_service().search_pricing(
            query=query,
            top_k=int(payload.get("top_k") or 5),
        )
    except Exception as exc:
        logger.warning("samvaad_tool_pricing_error", error=str(exc))
        return {
            "success": False,
            "error": f"pricing search failed: {exc}",
            "message": "I couldn't look that up right now.",
        }
    return {"success": True, "results": results}


@router.post("/human/transfer", dependencies=[Depends(_require_tool_auth)])
async def tool_human_transfer(payload: dict[str, Any]) -> dict[str, Any]:
    _require_tool_enabled("human/transfer")
    """Escalate to a human agent."""
    reason = payload.get("reason") or "Transferred by Samvaad agent"
    try:
        result = await get_escalation_service().transfer_to_human(
            user_input=payload.get("user_input") or reason,
            reason=reason,
            priority=str(payload.get("priority") or "NORMAL"),
        )
    except Exception as exc:
        logger.warning("samvaad_tool_transfer_error", error=str(exc))
        return {
            "success": False,
            "error": f"transfer failed: {exc}",
            "message": "I couldn't reach a human agent right now.",
        }
    return {"success": True, "result": result}


@router.post("/tasks/manage", dependencies=[Depends(_require_tool_auth)])
async def tool_tasks_manage(payload: dict[str, Any]) -> dict[str, Any]:
    _require_tool_enabled("tasks/manage")
    """Create, list, complete, update, or delete a user task.

    Body: ``{"action": "create"|"list"|"complete"|"update"|"delete",
    "task_id": str?, "session_id": str?, "user_id": str?, "title": str?,
    "description": str?, "priority": int?, "due_at": str? (ISO-8601),
    "status": str? (pending|in_progress|completed)}``
    """
    action = str(payload.get("action") or "").lower()
    task_service = get_task_service()

    if action == "list":
        try:
            tasks = await task_service.list(
                session_id=payload.get("session_id"),
                user_id=payload.get("user_id"),
                status=payload.get("status"),
                limit=int(payload.get("limit") or 50),
            )
        except Exception as exc:
            logger.warning("samvaad_tool_tasks_list_error", error=str(exc))
            return {
                "success": False,
                "error": f"task list failed: {exc}",
                "message": "I couldn't list your tasks right now.",
            }
        return {"success": True, "tasks": tasks, "count": len(tasks)}

    if action == "create":
        title = payload.get("title")
        if not title:
            return {
                "success": False,
                "error": "title is required for create",
                "message": "I need a title to create that task.",
            }
        try:
            task = await task_service.create(
                title=title,
                session_id=payload.get("session_id"),
                user_id=payload.get("user_id"),
                description=payload.get("description"),
                priority=int(payload.get("priority") or 0),
                due_at=_iso_to_dt(payload.get("due_at")),
            )
        except Exception as exc:
            logger.warning("samvaad_tool_tasks_create_error", error=str(exc))
            return {
                "success": False,
                "error": f"task create failed: {exc}",
                "message": "I couldn't create that task right now.",
            }
        return {"success": True, "task": task}

    task_id = payload.get("task_id")
    if not task_id:
        return {
            "success": False,
            "error": "task_id is required for complete/update/delete",
            "message": "I need a task id to do that.",
        }

    try:
        if action == "complete":
            completed_task = await task_service.complete(task_id)
            if completed_task is None:
                return {
                    "success": False,
                    "error": f"Task not found: {task_id}",
                    "message": "I couldn't find that task.",
                }
            return {"success": True, "task": completed_task}

        if action == "update":
            updated_task = await task_service.update(
                task_id,
                title=payload.get("title"),
                description=payload.get("description"),
                priority=(
                    int(payload["priority"])
                    if payload.get("priority") is not None
                    else None
                ),
                due_at=_iso_to_dt(payload.get("due_at")),
                status=payload.get("status"),
            )
            if updated_task is None:
                return {
                    "success": False,
                    "error": f"Task not found: {task_id}",
                    "message": "I couldn't find that task.",
                }
            return {"success": True, "task": updated_task}

        if action == "delete":
            deleted = await task_service.delete(task_id)
            if not deleted:
                return {
                    "success": False,
                    "error": f"Task not found: {task_id}",
                    "message": "I couldn't find that task.",
                }
            return {"success": True, "deleted": True, "task_id": task_id}
    except Exception as exc:
        logger.warning("samvaad_tool_tasks_error", error=str(exc))
        return {
            "success": False,
            "error": f"task operation failed: {exc}",
            "message": "I couldn't update that task right now.",
        }

    return {
        "success": False,
        "error": "action must be one of: create, list, complete, update, delete",
        "message": "I couldn't understand which task action you wanted.",
    }


@router.post("/on-end/record", dependencies=[Depends(_require_tool_auth)])
async def tool_on_end_record(payload: dict[str, Any]) -> dict[str, Any]:
    _require_tool_enabled("on-end/record")
    """Persist the conversation transcript + a summary fact to the memory service.

    Body: ``{"session_id": str, "user_id": str?, "transcript":
    [{"role": "user"|"bot", "text": str}], "duration_ms": int?}``
    """
    session_id = str(payload.get("session_id") or "")
    user_id = payload.get("user_id")
    transcript: list[dict[str, Any]] = payload.get("transcript") or []
    memory = get_memory_client()

    if session_id:
        for turn in transcript[:200]:
            role = (
                "user"
                if str(turn.get("role", "")).lower() in ("user", "customer")
                else "assistant"
            )
            text = (turn.get("text") or turn.get("content") or "").strip()
            if not text:
                continue
            await memory.add_message(
                session_id=session_id,
                role=role,
                content=text[:4000],
                user_id=user_id,
            )

    if transcript and user_id:
        summary = " ".join(
            (t.get("text") or t.get("content") or "") for t in transcript
        )[:4000]
        if summary.strip():
            await memory.store_long_term(
                user_id=user_id,
                content=f"Samvaad conversation summary: {summary}",
                memory_type="conversation_summary",
                source="samvaad",
                metadata={"session_id": session_id},
            )

    logger.info(
        "samvaad_tool_on_end_record",
        session_id=session_id,
        user_id=user_id,
        turns=len(transcript),
    )
    return {"success": True, "stored_messages": min(len(transcript), 200)}
