"""Natural-language meeting parsing (dates, times, attendees, titles)."""

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import dateparser
from dateparser.search import search_dates

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_DURATION_RE = re.compile(r"\b(\d+)\s*[- ]?(?:min(?:ute)?s?)\b", re.IGNORECASE)
_BOOKING_VERBS = re.compile(
    r"\b(please\s+|can\s+you\s+|could\s+you\s+|i\s+want\s+to\s+|i\s+'?d\s+like\s+to\s+)?"
    r"(book|schedule|set\s+up|arrange|plan|organize|reserve|slots?\s+for)\s+"
    r"(a\s+|an\s+|the\s+)?"
    r"(demo|meeting|appointment|call|session|conference|interview)?\s*",
    re.IGNORECASE,
)
_DEMO_WORDS = re.compile(
    r"\b(demo|appointment|meeting|call|session|conference|interview)\b", re.IGNORECASE
)
_TIME12_RE = re.compile(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", re.IGNORECASE)
_TIME24_RE = re.compile(r"\b(\d{1,2}):(\d{2})\b")
_DAYPART_MAP = {"morning": (9, 0), "afternoon": (15, 0), "evening": (18, 0), "night": (21, 0)}
_DAYPART_RE = re.compile(r"\b(morning|afternoon|evening|night)\b", re.IGNORECASE)
_WEEKDAY_RE = re.compile(
    r"\b(next|this|coming)\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    re.IGNORECASE,
)
_GOOD_DATE_RE = re.compile(
    r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday|tomorrow|today|"
    r"january|february|march|april|may|june|july|august|september|october|november|december|"
    r"\d{1,2}[/-]\d{1,2})\b",
    re.IGNORECASE,
)
_PREP_RE = re.compile(r"\b(at|on|for|to|from|with|about)\b", re.IGNORECASE)
_REDACTION_TOKEN_RE = re.compile(r"\[[A-Z][A-Z0-9_]+\]")


@dataclass
class ParsedMeeting:
    found: bool
    start_at: datetime | None = None
    end_at: datetime | None = None
    title: str = "Meeting"
    attendees: list[str] = field(default_factory=list)
    timezone: str = "UTC"
    duration_minutes: int = 30
    matched_text: str | None = None
    needs_datetime: bool = False


def _tzinfo(timezone: str) -> ZoneInfo | None:
    try:
        return ZoneInfo(timezone)
    except Exception:
        return None


def _dateparser_settings(timezone: str) -> dict[str, Any]:
    tz = _tzinfo(timezone)
    return {
        "RETURN_AS_TIMEZONE_AWARE": True,
        "TIMEZONE": timezone,
        "TO_TIMEZONE": timezone,
        "PREFER_DATES_FROM": "future",
        "PREFER_DAY_OF_MONTH": "first",
        "RELATIVE_BASE": datetime.now(tz or UTC),
    }


def _extract_time_of_day(
    text: str, prefer_last: bool = False
) -> tuple[int, int] | None:
    match = _TIME12_RE.search(text)
    if match:
        hour = int(match.group(1))
        minute = int(match.group(2) or 0)
        suffix = match.group(3).lower()
        if suffix == "pm" and hour < 12:
            hour += 12
        elif suffix == "am" and hour == 12:
            hour = 0
        if not prefer_last:
            return hour, minute
        all_matches = list(_TIME12_RE.finditer(text))
        last = all_matches[-1]
        hour = int(last.group(1))
        minute = int(last.group(2) or 0)
        suffix = last.group(3).lower()
        if suffix == "pm" and hour < 12:
            hour += 12
        elif suffix == "am" and hour == 12:
            hour = 0
        return hour, minute
    if not prefer_last:
        match = _TIME24_RE.search(text)
        if match:
            return int(match.group(1)), int(match.group(2))
    else:
        match24 = list(_TIME24_RE.finditer(text))
        if match24:
            last = match24[-1]
            return int(last.group(1)), int(last.group(2))
    match = _DAYPART_RE.search(text)
    if match:
        return _DAYPART_MAP[match.group(1).lower()]
    return None


def _resolve_datetime(
    text: str, timezone: str, time_of_day: tuple[int, int] | None
) -> tuple[datetime | None, str | None]:
    settings = _dateparser_settings(timezone)

    weekday = _WEEKDAY_RE.search(text)
    if weekday:
        name = weekday.group(2).lower()
        base = dateparser.parse(name, settings=settings)
        if base:
            if weekday.group(1).lower() in ("next", "coming"):
                base += timedelta(days=7)
            return base, weekday.group(0)

    for candidate in (text, _strip_time_noise(text)):
        parsed = dateparser.parse(candidate, settings=settings)
        if parsed:
            return parsed, candidate.strip()

    try:
        hits = search_dates(text, settings=settings) or []
    except Exception:
        hits = []
    for matched, parsed in hits:
        if _GOOD_DATE_RE.search(matched):
            return parsed, matched
    return None, None


def _strip_time_noise(text: str) -> str:
    cleaned = _BOOKING_VERBS.sub(" ", text)
    cleaned = _TIME12_RE.sub(" ", cleaned)
    cleaned = _TIME24_RE.sub(" ", cleaned)
    cleaned = _DAYPART_RE.sub(" ", cleaned)
    cleaned = _PREP_RE.sub(" ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def parse_meeting_request(
    text: str,
    timezone: str = "Asia/Kolkata",
    default_duration_minutes: int = 30,
    prefer_last_time: bool = False,
) -> ParsedMeeting:
    attendees = list(dict.fromkeys(_EMAIL_RE.findall(text)))
    duration = _extract_duration(text, default_duration_minutes)

    time_of_day = _extract_time_of_day(text, prefer_last=prefer_last_time)
    start_at, matched = _resolve_datetime(text, timezone, time_of_day)

    if start_at is None:
        title = _extract_title(text, attendees, None)
        return ParsedMeeting(
            found=False,
            title=title,
            attendees=attendees,
            timezone=timezone,
            duration_minutes=duration,
            needs_datetime=True,
        )

    if start_at.tzinfo is None:
        start_at = start_at.replace(tzinfo=_tzinfo(timezone) or ZoneInfo("UTC"))
    if time_of_day is not None:
        start_at = start_at.replace(
            hour=time_of_day[0], minute=time_of_day[1], second=0, microsecond=0
        )
    end_at = start_at + timedelta(minutes=duration)
    title = _extract_title(text, attendees, matched)

    return ParsedMeeting(
        found=True,
        start_at=start_at,
        end_at=end_at,
        title=title,
        attendees=attendees,
        timezone=timezone,
        duration_minutes=duration,
        matched_text=matched,
    )


def _extract_duration(text: str, default: int) -> int:
    match = _DURATION_RE.search(text)
    if not match:
        return default
    value = int(match.group(1))
    return max(10, min(value, 480))


def _extract_title(text: str, attendees: list[str], matched: str | None) -> str:
    cleaned = _BOOKING_VERBS.sub(" ", text)
    for attendee in attendees:
        cleaned = cleaned.replace(attendee, " ")
    if matched:
        cleaned = cleaned.replace(matched, " ")
    cleaned = _REDACTION_TOKEN_RE.sub(" ", cleaned)
    cleaned = _TIME12_RE.sub(" ", cleaned)
    cleaned = _TIME24_RE.sub(" ", cleaned)
    cleaned = _DAYPART_RE.sub(" ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .,;:-")
    cleaned = re.sub(r"^\s*(for|on|at|with|about|regarding|re|to)\b", "", cleaned)
    cleaned = re.sub(r"\b(for|on|at|with|about|regarding|re|to)\s*$", "", cleaned)
    cleaned = re.sub(r"\b(next|this|coming|last|upcoming)\b", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .,;:-")
    cleaned = cleaned[:120]
    if cleaned:
        cleaned = cleaned[0].upper() + cleaned[1:]
    if not cleaned or len(cleaned) < 2:
        if _DEMO_WORDS.search(text.lower()):
            return "Demo"
        return "Meeting"
    return cleaned


def extract_meeting_ref(text: str, timezone: str = "Asia/Kolkata") -> dict[str, Any]:
    """Best-effort extraction for update/cancel/list: date ref + meeting id + emails."""
    parsed = parse_meeting_request(text, timezone=timezone)
    meeting_id = None
    id_match = re.search(
        r"\b([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\b",
        text,
        re.IGNORECASE,
    )
    if id_match:
        meeting_id = id_match.group(1)
    return {
        "meeting_id": meeting_id,
        "start_at": parsed.start_at.isoformat() if parsed.start_at else None,
        "end_at": parsed.end_at.isoformat() if parsed.end_at else None,
        "attendees": parsed.attendees,
        "timezone": parsed.timezone,
        "ref_date": parsed.start_at.date().isoformat() if parsed.start_at else None,
    }
