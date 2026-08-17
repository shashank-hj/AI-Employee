"""Production calendar tools consumed by the orchestrator tool registry.

Replacements for the mock calendar tools in the production path.
"""

from datetime import datetime
import re
from typing import Any

from orchestrator.services.calendar.base import CalendarEventDraft
from orchestrator.services.calendar_service import CalendarService
from orchestrator.tools.base import BaseTool


def _to_dt(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


_AUTO_TITLE_PATTERN = re.compile(
    r"^(update|reschedule|move|postpone|change|shift|re-schedule|meeting|demo|appointment)\b",
    re.IGNORECASE,
)


def _is_auto_title(title: Any) -> bool:
    """True when the title looks like a parser byproduct rather than a user-supplied name."""
    if not title or not isinstance(title, str):
        return True
    stripped = title.strip()
    if not stripped:
        return True
    return bool(_AUTO_TITLE_PATTERN.match(stripped))


def _wrap_meeting_result(result: dict[str, Any], cancelled: bool = False) -> dict[str, Any]:
    """Normalize a service lifecycle result into the ``{success, data}`` shape
    the graph's natural-language builder and response schema expect."""
    if not result.get("success"):
        return {"success": False, "error": result.get("error", "operation failed")}
    meeting = result.get("meeting") or {}
    data = dict(meeting)
    if cancelled:
        data["cancelled"] = True
    return {"success": True, "data": data}


def _wrap_schedule_result(result: dict[str, Any]) -> dict[str, Any]:
    """Normalize a create/confirm result into ``{success, data: {event, title, datetime}}``."""
    if not result.get("success"):
        return {"success": False, "error": result.get("error", "scheduling failed")}
    meeting = result.get("meeting") or {}
    return {
        "success": True,
        "data": {
            "event": dict(meeting),
            "title": meeting.get("title", ""),
            "datetime": meeting.get("start_at", ""),
        },
    }


class CalendarAvailabilityTool(BaseTool):
    name = "calendar"
    description = (
        "Check availability for a meeting slot and prepare a booking proposal "
        "that the user must confirm before it is scheduled."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "start_at": {"type": "string", "description": "Requested start (ISO 8601)"},
            "end_at": {"type": "string", "description": "Requested end (ISO 8601)"},
            "duration_minutes": {"type": "integer", "default": 30},
            "timezone": {"type": "string", "default": "Asia/Kolkata"},
            "attendees": {"type": "array", "items": {"type": "string"}},
            "session_id": {"type": "string"},
            "user_id": {"type": "string"},
        },
        "required": ["session_id", "start_at", "end_at"],
    }

    def __init__(self, calendar_service: CalendarService) -> None:
        self._svc = calendar_service

    async def invoke(self, parameters: dict[str, Any]) -> dict[str, Any]:
        session_id = parameters.get("session_id")
        start_at = _to_dt(parameters.get("start_at"))
        end_at = _to_dt(parameters.get("end_at"))
        if not session_id or start_at is None or end_at is None:
            return {
                "success": True,
                "data": {
                    "needs_datetime": True,
                    "message": "I need a date and time for the meeting. Could you tell me when works for you?",
                },
            }
        draft = CalendarEventDraft(
            title=parameters.get("title") or "Meeting",
            start_at=start_at,
            end_at=end_at,
            timezone=parameters.get("timezone") or self._svc._timezone,
            description=parameters.get("description"),
            attendees=parameters.get("attendees") or [],
        )
        result = await self._svc.propose_booking(
            session_id=session_id,
            user_id=parameters.get("user_id"),
            draft=draft,
            duration_minutes=parameters.get("duration_minutes", 30),
        )
        data = dict(result)
        data["available_slots"] = result.get("slots", [])
        return {"success": True, "data": data}


class ScheduleMeetingTool(BaseTool):
    name = "schedule_meeting"
    description = "Create a meeting after the user confirms a pending booking proposal."
    parameters_schema = {
        "type": "object",
        "properties": {
            "session_id": {"type": "string"},
            "user_id": {"type": "string"},
            "pending_id": {"type": "string"},
            "title": {"type": "string"},
            "start_at": {"type": "string"},
            "end_at": {"type": "string"},
            "duration_minutes": {"type": "integer", "default": 30},
            "timezone": {"type": "string", "default": "Asia/Kolkata"},
            "attendees": {"type": "array", "items": {"type": "string"}},
            "description": {"type": "string"},
        },
        "required": ["session_id"],
    }

    def __init__(self, calendar_service: CalendarService) -> None:
        self._svc = calendar_service

    async def invoke(self, parameters: dict[str, Any]) -> dict[str, Any]:
        session_id = parameters.get("session_id")
        if not session_id:
            return {"success": False, "error": "session_id required"}

        pending_id = parameters.get("pending_id")
        has_pending = await self._svc.has_pending_booking(session_id)
        if pending_id or has_pending:
            result = await self._svc.confirm_booking(
                session_id=session_id, user_id=parameters.get("user_id")
            )
            return _wrap_schedule_result(result)

        start_at = _to_dt(parameters.get("start_at"))
        end_at = _to_dt(parameters.get("end_at"))
        if start_at is None or end_at is None:
            return {
                "success": True,
                "data": {
                    "needs_datetime": True,
                    "message": (
                        "I need a date and time for the meeting. "
                        "Could you tell me when works for you?"
                    ),
                },
            }
        draft = CalendarEventDraft(
            title=parameters.get("title") or "Meeting",
            start_at=start_at,
            end_at=end_at,
            timezone=parameters.get("timezone") or self._svc._timezone,
            description=parameters.get("description"),
            attendees=parameters.get("attendees") or [],
        )
        result = await self._svc.create_meeting(
            draft, session_id=session_id, user_id=parameters.get("user_id")
        )
        return _wrap_schedule_result(result)


class ScheduleDemoTool(ScheduleMeetingTool):
    name = "schedule_demo"
    description = "Confirm and create a demo/meeting after the user confirms a pending booking proposal."


class ListMeetingsTool(BaseTool):
    name = "calendar_list"
    description = "List the user's scheduled meetings. Optionally filter by status or time window."
    parameters_schema = {
        "type": "object",
        "properties": {
            "session_id": {"type": "string"},
            "status": {"type": "string", "default": "scheduled"},
            "start": {"type": "string"},
            "end": {"type": "string"},
            "limit": {"type": "integer", "default": 50},
        },
        "required": ["session_id"],
    }

    def __init__(self, calendar_service: CalendarService) -> None:
        self._svc = calendar_service

    async def invoke(self, parameters: dict[str, Any]) -> dict[str, Any]:
        result = await self._svc.list_meetings(
            session_id=parameters.get("session_id"),
            start=_to_dt(parameters.get("start")),
            end=_to_dt(parameters.get("end")),
            status=parameters.get("status") or "scheduled",
            limit=parameters.get("limit", 50),
        )
        events = result.get("meetings", [])
        return {
            "success": bool(result.get("success")),
            "data": {"events": events, "count": len(events)},
            "provider": result.get("provider"),
        }


class CancelMeetingTool(BaseTool):
    name = "calendar_cancel"
    description = "Cancel a meeting by its ID or by matching the referenced date/time."
    parameters_schema = {
        "type": "object",
        "properties": {
            "meeting_id": {"type": "string"},
            "session_id": {"type": "string"},
            "start_at": {"type": "string", "description": "Reference date/time to match (ISO 8601)"},
            "ref_date": {"type": "string", "description": "Reference date (YYYY-MM-DD)"},
        },
        "required": [],
    }

    def __init__(self, calendar_service: CalendarService) -> None:
        self._svc = calendar_service

    async def invoke(self, parameters: dict[str, Any]) -> dict[str, Any]:
        meeting_id = parameters.get("meeting_id")
        if meeting_id:
            result = await self._svc.cancel_meeting(meeting_id)
            return _wrap_meeting_result(result, cancelled=True)

        ref_date = _to_dt(parameters.get("start_at"))
        if ref_date is None and parameters.get("ref_date"):
            try:
                from datetime import date

                ref_date = datetime.combine(date.fromisoformat(parameters["ref_date"]), datetime.min.time())
            except ValueError:
                ref_date = None

        matched = await self._svc.match_meetings(
            session_id=parameters.get("session_id"),
            meeting_id=None,
            ref_date=ref_date,
        )
        matches = matched.get("matches", [])
        if not matches:
            return {"success": True, "data": {"cancelled": False, "message": "No matching upcoming meeting found."}}
        if len(matches) == 1:
            result = await self._svc.cancel_meeting(matches[0]["id"])
            return _wrap_meeting_result(result, cancelled=True)
        return {
            "success": True,
            "data": {
                "cancelled": False,
                "needs_disambiguation": True,
                "meetings": matches,
                "message": "Which meeting did you want to cancel?",
            },
        }


class UpdateMeetingTool(BaseTool):
    name = "calendar_update"
    description = "Reschedule or edit an existing meeting by its ID or by matching the referenced date/time."
    parameters_schema = {
        "type": "object",
        "properties": {
            "meeting_id": {"type": "string"},
            "session_id": {"type": "string"},
            "start_at": {"type": "string", "description": "Existing meeting reference (ISO 8601)"},
            "new_start_at": {"type": "string", "description": "New start time (ISO 8601)"},
            "new_end_at": {"type": "string", "description": "New end time (ISO 8601)"},
            "title": {"type": "string"},
            "attendees": {"type": "array", "items": {"type": "string"}},
            "duration_minutes": {"type": "integer", "default": 30},
            "timezone": {"type": "string", "default": "Asia/Kolkata"},
        },
        "required": [],
    }

    def __init__(self, calendar_service: CalendarService) -> None:
        self._svc = calendar_service

    async def invoke(self, parameters: dict[str, Any]) -> dict[str, Any]:
        meeting_id = parameters.get("meeting_id")
        existing_title = None
        if not meeting_id:
            ref_date = _to_dt(parameters.get("start_at"))
            matched = await self._svc.match_meetings(
                session_id=parameters.get("session_id"), ref_date=ref_date
            )
            matches = matched.get("matches", [])
            if not matches:
                return {"success": True, "data": {"updated": False, "message": "No matching meeting found."}}
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
            matched = await self._svc.match_meetings(meeting_id=meeting_id)
            if matched.get("matches"):
                existing_title = matched["matches"][0].get("title")

        new_start = _to_dt(parameters.get("new_start_at"))
        new_end = _to_dt(parameters.get("new_end_at"))
        if new_start is None or new_end is None:
            return {"success": True, "data": {"updated": False, "message": "What is the new date/time?"}}

        new_title = parameters.get("title")
        if _is_auto_title(new_title):
            new_title = existing_title or "Meeting"
        draft = CalendarEventDraft(
            title=new_title,
            start_at=new_start,
            end_at=new_end,
            timezone=parameters.get("timezone") or self._svc._timezone,
            attendees=parameters.get("attendees") or [],
        )
        result = await self._svc.update_meeting(meeting_id, draft)
        meeting = result.get("meeting")
        if result.get("success") and meeting:
            return {"success": True, "data": meeting}
        return {"success": False, "error": result.get("error", "update failed")}


def register_calendar_tools(registry, calendar_service: CalendarService) -> None:
    """Register production calendar tools, replacing any mock calendar tools."""
    for tool in (
        CalendarAvailabilityTool(calendar_service),
        ScheduleMeetingTool(calendar_service),
        ScheduleDemoTool(calendar_service),
        ListMeetingsTool(calendar_service),
        CancelMeetingTool(calendar_service),
        UpdateMeetingTool(calendar_service),
    ):
        registry.register_or_replace(tool)
