
from orchestrator.graph.nodes import (
    _build_respond_prompt,
    _build_tool_context,
    _fmt_ist,
    _now_directive,
    _redact_history_pii,
    _render_calendar_result,
)


class TestNowDirective:
    def test_contains_current_date_ist(self):
        directive = _now_directive()
        assert "IST" in directive
        assert "CURRENT DATE AND TIME" in directive
        assert "Never claim you lack access" in directive


class TestRedactHistoryPii:
    def test_redacts_stale_email_when_current_input_has_no_email(self):
        out = _redact_history_pii(
            "please email me at hjshashank77@gmail.com", "Schedule a demo tomorrow at 2 PM"
        )
        assert "hjshashank77@gmail.com" not in out
        assert "[EMAIL]" in out

    def test_keeps_email_when_user_rementions_it(self):
        out = _redact_history_pii(
            "use hjshashank77@gmail.com for the invite",
            "book with ash@example.com and hjshashank77@gmail.com",
        )
        assert "hjshashank77@gmail.com" in out

    def test_redacts_in_current_turn_with_different_email(self):
        out = _redact_history_pii(
            "old contact bob@example.com", "Book with ash@example.com"
        )
        assert "bob@example.com" not in out
        assert "[EMAIL]" in out


class TestBuildRespondPrompt:
    def test_tool_results_never_truncated_with_long_history(self):
        tool_context = "AVAILABLE: the requested slot (15 Aug 2026, 04:30 PM IST) is free."
        history = [
            {"role": "user", "content": "x" * 50},
            {"role": "assistant", "content": "y" * 50},
        ] * 30  # huge history
        prompt = _build_respond_prompt(
            "Book a demo tomorrow at 2 PM",
            tool_context,
            history,
            max_len=500,
        )
        assert tool_context in prompt
        assert "Book a demo tomorrow at 2 PM" in prompt
        assert len(prompt) <= 500 + len(tool_context)

    def test_no_history(self):
        prompt = _build_respond_prompt("hi", "tool out", [])
        assert prompt == "User's question: hi\n\nTool results:\ntool out"

    def test_stale_email_redacted_from_history(self):
        history = [
            {"role": "user", "content": "send email to old@example.com"},
            {"role": "assistant", "content": "Done."},
        ]
        prompt = _build_respond_prompt(
            "Schedule a demo tomorrow at 2 PM", "tool out", history
        )
        assert "old@example.com" not in prompt
        assert "[EMAIL]" in prompt


class TestFmtIst:
    def test_utc_converts_to_ist(self):
        # 11:00 UTC on 15 Aug = 16:30 IST
        assert _fmt_ist("2026-08-15T11:00:00+00:00") == "15 Aug 2026, 04:30 PM IST"

    def test_ist_passthrough(self):
        assert _fmt_ist("2026-08-15T16:30:00+05:30") == "15 Aug 2026, 04:30 PM IST"

    def test_invalid_returns_raw(self):
        assert _fmt_ist("not-a-date") == "not-a-date"

    def test_none(self):
        assert _fmt_ist(None) == "unknown time"


class TestRenderCalendarResult:
    def test_propose_available(self):
        data = {
            "proposed": True,
            "proposal_id": "p-1",
            "available": True,
            "slots": [
                {
                    "start": "2026-08-15T11:00:00+00:00",
                    "end": "2026-08-15T11:30:00+00:00",
                    "available": True,
                }
            ],
            "attendees": ["ash@example.com"],
        }
        text = _render_calendar_result(data)
        assert "AVAILABLE" in text
        assert "ash@example.com" in text
        assert "confirm" in text.lower()

    def test_propose_available_via_available_slots(self):
        data = {
            "proposed": True,
            "available": True,
            "available_slots": [
                {"start": "2026-08-15T11:00:00+00:00", "end": "2026-08-15T11:30:00+00:00"}
            ],
        }
        assert "AVAILABLE" in _render_calendar_result(data)

    def test_propose_busy_with_alternatives(self):
        data = {
            "proposed": False,
            "available": False,
            "available_slots": [
                {"start": "2026-08-15T12:00:00+00:00", "end": "2026-08-15T12:30:00+00:00"},
            ],
        }
        text = _render_calendar_result(data)
        assert "NOT AVAILABLE" in text
        assert "IST" in text

    def test_confirm_scheduled(self):
        data = {
            "event": {
                "title": "Demo",
                "start_at": "2026-08-15T11:00:00+00:00",
                "attendees": ["ash@example.com"],
                "link": "https://cal.example/evt-1",
            },
            "title": "Demo",
            "datetime": "2026-08-15T11:00:00+00:00",
        }
        text = _render_calendar_result(data)
        assert "CONFIRMED" in text
        assert "https://cal.example/evt-1" in text
        assert "ash@example.com" in text

    def test_needs_datetime(self):
        text = _render_calendar_result(
            {"needs_datetime": True, "message": "I need a date and time."}
        )
        assert "date and time" in text

    def test_build_tool_context_renders_calendar_as_english(self):
        results = [
            {
                "tool_name": "calendar",
                "success": True,
                "data": {"proposed": True, "available": True},
            },
            {
                "tool_name": "get_weather",
                "success": True,
                "data": {"location": "Pune", "temperature": 30},
            },
        ]
        context = _build_tool_context(results)
        assert "AVAILABLE" in context
        assert "proposal" in context.lower()
        assert "get_weather" in context

    def test_build_tool_context_failed_result(self):
        context = _build_tool_context(
            [{"tool_name": "calendar", "success": False, "error": "provider down"}]
        )
        assert "FAILED" in context
        assert "provider down" in context
