import json

import pytest

from shared.llm.base import IntentClassification, LLMProvider, LLMResponse

from orchestrator.graph.state import AgentState
from orchestrator.planner.mock_planner import MockPlanner
from orchestrator.planner.llm_planner import LLMPlanner


def _make_state(user_input: str) -> AgentState:
    return AgentState(
        request_id="test-1",
        user_input=user_input,
        user_id=None,
        session_id=None,
        memory_context=[],
        document_context=[],
        user_preferences={},
        plan=[],
        current_step_index=0,
        tool_results=[],
        execution_log=[],
        final_response=None,
        error=None,
    )


class _FakeLLMProvider(LLMProvider):
    def __init__(self, intent: str = "general", confidence: float = 0.9, suggested_tools: list[str] | None = None):
        self._intent = intent
        self._confidence = confidence
        self._suggested_tools = suggested_tools or []

    async def classify_intent(
        self,
        user_input: str,
        context: str | None = None,
    ) -> IntentClassification:
        return IntentClassification(
            intent=self._intent,
            confidence=self._confidence,
            reason=f"Fake classification for: {user_input[:50]}",
            suggested_tools=self._suggested_tools,
        )

    async def generate(self, system_prompt: str, user_message: str) -> LLMResponse:
        return LLMResponse(content="Fake response", model="fake-model")

    async def health_check(self) -> bool:
        return True


class _DraftLLMProvider(LLMProvider):
    """LLM whose generate() returns a canned email-draft JSON object."""

    def __init__(self, draft: dict):
        self._draft = draft

    async def classify_intent(
        self,
        user_input: str,
        context: str | None = None,
    ) -> IntentClassification:
        return IntentClassification(
            intent="email",
            confidence=0.9,
            reason=f"Fake classification for: {user_input[:50]}",
            suggested_tools=["email_send"],
        )

    async def generate(self, system_prompt: str, user_message: str) -> LLMResponse:
        return LLMResponse(content=json.dumps(self._draft), model="fake-model")

    async def health_check(self) -> bool:
        return True


class TestMockPlanner:
    @pytest.mark.asyncio
    async def test_calculator_intent(self):
        planner = MockPlanner()
        state = _make_state("Calculate 2 + 3 * 4")
        plan = await planner.create_plan(state)
        assert len(plan) >= 1
        assert plan[0]["tool_name"] == "calculator"

    @pytest.mark.asyncio
    async def test_weather_intent(self):
        planner = MockPlanner()
        state = _make_state("What is the weather in Tokyo?")
        plan = await planner.create_plan(state)
        assert len(plan) >= 1
        assert plan[0]["tool_name"] == "get_weather"
        assert plan[0]["parameters"]["location"] == "Tokyo"

    @pytest.mark.asyncio
    async def test_search_intent(self):
        planner = MockPlanner()
        state = _make_state("Find documents about remote work policy")
        plan = await planner.create_plan(state)
        assert len(plan) >= 1
        assert plan[0]["tool_name"] == "search_documents"
        assert "remote work policy" in plan[0]["parameters"]["query"].lower()

    @pytest.mark.asyncio
    async def test_email_intent(self):
        planner = MockPlanner()
        state = _make_state("Send an email to John about the project update")
        plan = await planner.create_plan(state)
        assert len(plan) >= 1
        assert plan[0]["tool_name"] == "send_email"

    @pytest.mark.asyncio
    async def test_meeting_intent(self):
        planner = MockPlanner()
        state = _make_state("Schedule a meeting about the Q3 roadmap with the engineering team")
        plan = await planner.create_plan(state)
        assert len(plan) >= 1
        assert plan[0]["tool_name"] == "schedule_meeting"

    @pytest.mark.asyncio
    async def test_multi_intent(self):
        planner = MockPlanner()
        state = _make_state("Calculate the budget and then search for expense policies")
        plan = await planner.create_plan(state)
        assert len(plan) == 1
        assert plan[0]["tool_name"] == "calculator"

    @pytest.mark.asyncio
    async def test_weather_without_document_search(self):
        planner = MockPlanner()
        state = _make_state("What is the weather in Mumbai?")
        plan = await planner.create_plan(state)
        assert len(plan) == 1
        assert plan[0]["tool_name"] == "get_weather"
        assert plan[0]["parameters"]["location"] == "Mumbai"

    @pytest.mark.asyncio
    async def test_bengaluru_location_extracted(self):
        planner = MockPlanner()
        state = _make_state("What is the weather in Bengaluru?")
        plan = await planner.create_plan(state)
        assert len(plan) == 1
        assert plan[0]["parameters"]["location"] == "Bengaluru"

    @pytest.mark.asyncio
    async def test_in_cityname_regex_fallback(self):
        planner = MockPlanner()
        state = _make_state("What is the weather in Amsterdam?")
        plan = await planner.create_plan(state)
        assert len(plan) == 1
        assert plan[0]["parameters"]["location"] == "Amsterdam"

    @pytest.mark.asyncio
    async def test_no_intent_falls_back_to_search(self):
        planner = MockPlanner()
        state = _make_state("Hello! How are you doing today?")
        plan = await planner.create_plan(state)
        assert len(plan) == 1
        assert plan[0]["tool_name"] == "search_documents"

    @pytest.mark.asyncio
    async def test_greeting_short_circuits_without_search(self):
        planner = MockPlanner()
        state = _make_state("hi")
        plan = await planner.create_plan(state)
        assert plan == []
        assert state.get("final_response") is not None

    @pytest.mark.asyncio
    async def test_plan_steps_have_required_fields(self):
        planner = MockPlanner()
        state = _make_state("Calculate 5 + 5")
        plan = await planner.create_plan(state)
        for step in plan:
            assert "tool_name" in step
            assert "parameters" in step
            assert "reason" in step


class TestLLMPlanner:
    @pytest.mark.asyncio
    async def test_sales_intent_maps_to_pricing_and_search(self):
        llm = _FakeLLMProvider(intent="sales", confidence=0.95)
        planner = LLMPlanner(llm_provider=llm)
        state = _make_state("What are your pricing plans for enterprise?")
        plan = await planner.create_plan(state)
        assert len(plan) == 2
        tool_names = [s["tool_name"] for s in plan]
        assert "search_pricing" in tool_names
        assert "search_documents" in tool_names

    @pytest.mark.asyncio
    async def test_support_intent_maps_to_order_and_search(self):
        llm = _FakeLLMProvider(intent="support", confidence=0.92)
        planner = LLMPlanner(llm_provider=llm)
        state = _make_state("My order hasn't arrived yet")
        plan = await planner.create_plan(state)
        assert len(plan) == 2
        tool_names = [s["tool_name"] for s in plan]
        assert "lookup_order" in tool_names
        assert "search_documents" in tool_names

    @pytest.mark.asyncio
    async def test_suggested_tools_from_llm_override_map(self):
        llm = _FakeLLMProvider(
            intent="sales",
            confidence=0.92,
            suggested_tools=["search_pricing", "search_documents"],
        )
        planner = LLMPlanner(llm_provider=llm)
        state = _make_state("Show me enterprise pricing")
        plan = await planner.create_plan(state)
        assert len(plan) == 2
        tool_names = [s["tool_name"] for s in plan]
        assert "search_pricing" in tool_names
        assert "search_documents" in tool_names

    @pytest.mark.asyncio
    async def test_suggested_tools_filtered_by_agent_allowlist(self):
        llm = _FakeLLMProvider(
            intent="support",
            confidence=0.92,
            suggested_tools=["send_email", "search_documents"],
        )
        planner = LLMPlanner(llm_provider=llm)
        state = _make_state("What can you help me with?")
        plan = await planner.create_plan(state)
        tool_names = [s["tool_name"] for s in plan]
        assert "send_email" not in tool_names
        assert "search_documents" in tool_names

    @pytest.mark.asyncio
    async def test_invalid_suggested_tools_filtered(self):
        llm = _FakeLLMProvider(
            intent="sales",
            confidence=0.85,
            suggested_tools=["nonexistent_tool", "search_documents"],
        )
        planner = LLMPlanner(llm_provider=llm)
        state = _make_state("Show me the product catalog")
        plan = await planner.create_plan(state)
        assert len(plan) == 1
        assert plan[0]["tool_name"] == "search_documents"

    @pytest.mark.asyncio
    async def test_booking_intent_maps_to_calendar(self):
        llm = _FakeLLMProvider(intent="booking", confidence=0.88)
        planner = LLMPlanner(llm_provider=llm)
        state = _make_state("Schedule a demo for next Tuesday")
        plan = await planner.create_plan(state)
        assert len(plan) == 1
        assert plan[0]["tool_name"] == "calendar"
        assert plan[0]["parameters"]["title"] == "Demo"
        assert "session_id" in plan[0]["parameters"]

    @pytest.mark.asyncio
    async def test_booking_merges_structured_attendees(self):
        llm = _FakeLLMProvider(intent="booking", confidence=0.88)
        planner = LLMPlanner(llm_provider=llm)
        state = _make_state("Book a meeting tomorrow at 4:30 PM with [EMAIL]")
        state["request_metadata"] = {"attendees": ["ash@example.com"]}
        plan = await planner.create_plan(state)
        assert plan[0]["tool_name"] == "calendar"
        assert plan[0]["parameters"]["attendees"] == ["ash@example.com"]
        assert plan[0]["parameters"]["title"] not in ("[EMAIL]", "Tomorrow at with [EMAIL]")

    @pytest.mark.asyncio
    async def test_booking_merges_structured_and_parsed_attendees_dedup(self):
        llm = _FakeLLMProvider(intent="booking", confidence=0.88)
        planner = LLMPlanner(llm_provider=llm)
        state = _make_state("Book a meeting tomorrow at 4:30 PM with bob@example.com")
        state["request_metadata"] = {"attendees": ["bob@example.com", "carol@example.com"]}
        plan = await planner.create_plan(state)
        attendees = plan[0]["parameters"]["attendees"]
        assert attendees == ["bob@example.com", "carol@example.com"]

    @pytest.mark.asyncio
    async def test_email_uses_structured_recipient_from_redacted_input(self):
        llm = _FakeLLMProvider(intent="email", confidence=0.95)
        planner = LLMPlanner(llm_provider=llm)
        state = _make_state("Can you send email to [EMAIL]")
        state["request_metadata"] = {"attendees": ["hjshashank77@gmail.com"]}
        state["contact"] = {"user_id": "u1", "email": "shashankhj1@gmail.com"}
        plan = await planner.create_plan(state)
        email_step = next(s for s in plan if s["tool_name"] in ("email_send", "send_email"))
        assert email_step["parameters"]["to"] == "hjshashank77@gmail.com"

    @pytest.mark.asyncio
    async def test_email_placeholder_substituted_from_structured_emails(self):
        llm = _DraftLLMProvider({"to": "[EMAIL]", "subject": "Demo", "body": "Hello"})
        planner = LLMPlanner(llm_provider=llm)
        state = _make_state("Can you send email to [EMAIL]")
        state["request_metadata"] = {"attendees": ["hjshashank77@gmail.com"]}
        plan = await planner.create_plan(state)
        email_step = next(s for s in plan if s["tool_name"] in ("email_send", "send_email"))
        assert email_step["parameters"]["to"] == "hjshashank77@gmail.com"

    @pytest.mark.asyncio
    async def test_email_uses_contact_email_when_no_attendees(self):
        llm = _DraftLLMProvider({"to": "[EMAIL]", "subject": "Demo", "body": "Hello"})
        planner = LLMPlanner(llm_provider=llm)
        state = _make_state("Can you send email to [EMAIL]")
        state["contact"] = {"user_id": "u1", "email": "hjshashank77@gmail.com"}
        plan = await planner.create_plan(state)
        email_step = next(s for s in plan if s["tool_name"] in ("email_send", "send_email"))
        assert email_step["parameters"]["to"] == "hjshashank77@gmail.com"

    @pytest.mark.asyncio
    async def test_schedule_meeting_merges_structured_attendees(self):
        llm = _FakeLLMProvider(intent="booking", confidence=0.88, suggested_tools=["schedule_meeting"])
        planner = LLMPlanner(llm_provider=llm)
        state = _make_state("yes please confirm the meeting")
        state["request_metadata"] = {"attendees": ["ash@example.com"]}
        plan = await planner.create_plan(state)
        assert plan[0]["tool_name"] == "schedule_meeting"
        assert "ash@example.com" in plan[0]["parameters"]["attendees"]

    @pytest.mark.asyncio
    async def test_complaint_intent_returns_empty_plan(self):
        llm = _FakeLLMProvider(intent="complaint", confidence=0.97)
        planner = LLMPlanner(llm_provider=llm)
        state = _make_state("I am extremely dissatisfied with the service quality and want to file a formal grievance")
        plan = await planner.create_plan(state)
        assert len(plan) == 0
        assert state.get("final_response") is not None

    @pytest.mark.asyncio
    async def test_escalate_intent_returns_empty_plan(self):
        llm = _FakeLLMProvider(intent="escalate", confidence=0.85)
        planner = LLMPlanner(llm_provider=llm)
        state = _make_state("Let me talk to a real person")
        plan = await planner.create_plan(state)
        assert len(plan) == 0
        assert state.get("final_response") is not None

    @pytest.mark.asyncio
    async def test_general_intent_falls_back_to_search(self):
        llm = _FakeLLMProvider(intent="general", confidence=0.75)
        planner = LLMPlanner(llm_provider=llm)
        state = _make_state("Hello! How are you?")
        plan = await planner.create_plan(state)
        assert len(plan) == 1
        assert plan[0]["tool_name"] == "search_documents"

    @pytest.mark.asyncio
    async def test_greeting_short_circuits_without_tools(self):
        llm = _FakeLLMProvider(intent="general", confidence=0.9)
        planner = LLMPlanner(llm_provider=llm)
        state = _make_state("hi")
        plan = await planner.create_plan(state)
        assert plan == []
        assert state.get("final_response") is not None

    @pytest.mark.asyncio
    async def test_fallback_on_classification_error(self):
        class _ErrorLLM(LLMProvider):
            async def classify_intent(
                self,
                user_input: str,
                context: str | None = None,
            ) -> IntentClassification:
                raise RuntimeError("Simulated LLM failure")
            async def generate(self, system_prompt: str, user_message: str) -> LLMResponse:
                return LLMResponse(content="", model="error")  # pragma: no cover
            async def health_check(self) -> bool:
                return False

        planner = LLMPlanner(llm_provider=_ErrorLLM(), fallback_intent="general")
        state = _make_state("Tell me about the company history")
        plan = await planner.create_plan(state)
        assert len(plan) == 1
        assert plan[0]["tool_name"] == "search_documents"

    @pytest.mark.asyncio
    async def test_plan_steps_have_required_fields(self):
        llm = _FakeLLMProvider(intent="sales", confidence=0.9)
        planner = LLMPlanner(llm_provider=llm)
        state = _make_state("Show me product prices")
        plan = await planner.create_plan(state)
        for step in plan:
            assert "tool_name" in step
            assert "parameters" in step
            assert "reason" in step


class _FakePendingRepo:
    def __init__(self, proposal=None):
        self.proposal = proposal
        self.cleared = False

    async def get_active(self, session_id):
        return self.proposal

    async def clear(self, session_id):
        self.cleared = True


class _FakeProposal:
    id = "pending-123"


class TestCalendarIntents:
    @pytest.mark.asyncio
    async def test_calendar_list_intent(self):
        llm = _FakeLLMProvider(intent="calendar_list", confidence=0.9)
        planner = LLMPlanner(llm_provider=llm)
        state = _make_state("Show my upcoming meetings")
        state["session_id"] = "sess-1"
        plan = await planner.create_plan(state)
        assert len(plan) == 1
        assert plan[0]["tool_name"] == "calendar_list"
        assert plan[0]["parameters"]["session_id"] == "sess-1"

    @pytest.mark.asyncio
    async def test_calendar_update_intent(self):
        llm = _FakeLLMProvider(intent="calendar_update", confidence=0.9)
        planner = LLMPlanner(llm_provider=llm)
        state = _make_state("Reschedule my meeting next Tuesday at 3pm")
        state["session_id"] = "sess-1"
        plan = await planner.create_plan(state)
        assert len(plan) == 1
        assert plan[0]["tool_name"] == "calendar_update"
        assert "new_start_at" in plan[0]["parameters"]
        assert plan[0]["parameters"]["session_id"] == "sess-1"

    @pytest.mark.asyncio
    async def test_calendar_cancel_intent(self):
        llm = _FakeLLMProvider(intent="calendar_cancel", confidence=0.9)
        planner = LLMPlanner(llm_provider=llm)
        state = _make_state("Cancel my meeting on Monday")
        state["session_id"] = "sess-1"
        plan = await planner.create_plan(state)
        assert len(plan) == 1
        assert plan[0]["tool_name"] == "calendar_cancel"
        assert plan[0]["parameters"]["session_id"] == "sess-1"

    @pytest.mark.asyncio
    async def test_booking_intent_injects_session_id(self):
        llm = _FakeLLMProvider(intent="booking", confidence=0.88)
        planner = LLMPlanner(llm_provider=llm)
        state = _make_state("Book a demo on Tuesday at 2pm")
        state["session_id"] = "sess-1"
        state["user_id"] = "user-1"
        plan = await planner.create_plan(state)
        assert plan[0]["tool_name"] == "calendar"
        assert plan[0]["parameters"]["session_id"] == "sess-1"
        assert plan[0]["parameters"]["user_id"] == "user-1"
        assert plan[0]["parameters"]["title"] == "Demo"

    @pytest.mark.asyncio
    async def test_pending_booking_confirmed(self):
        llm = _FakeLLMProvider(intent="general", confidence=0.9)
        repo = _FakePendingRepo(proposal=_FakeProposal())
        planner = LLMPlanner(llm_provider=llm, pending_repo=repo)
        state = _make_state("Yes, go ahead")
        state["session_id"] = "sess-1"
        plan = await planner.create_plan(state)
        assert len(plan) == 1
        assert plan[0]["tool_name"] == "schedule_meeting"
        assert plan[0]["parameters"]["pending_id"] == "pending-123"
        assert plan[0]["parameters"]["session_id"] == "sess-1"

    @pytest.mark.asyncio
    async def test_pending_booking_declined(self):
        llm = _FakeLLMProvider(intent="general", confidence=0.9)
        repo = _FakePendingRepo(proposal=_FakeProposal())
        planner = LLMPlanner(llm_provider=llm, pending_repo=repo)
        state = _make_state("No thanks")
        state["session_id"] = "sess-1"
        plan = await planner.create_plan(state)
        assert plan == []
        assert repo.cleared is True
        assert state.get("final_response") is not None

    @pytest.mark.asyncio
    async def test_pending_booking_ignored_for_unrelated_input(self):
        llm = _FakeLLMProvider(intent="general", confidence=0.9)
        repo = _FakePendingRepo(proposal=_FakeProposal())
        planner = LLMPlanner(llm_provider=llm, pending_repo=repo)
        state = _make_state("What's the weather in Mumbai?")
        state["session_id"] = "sess-1"
        plan = await planner.create_plan(state)
        assert len(plan) == 1
        assert plan[0]["tool_name"] == "get_weather"
        assert repo.cleared is False

    @pytest.mark.asyncio
    async def test_pending_booking_no_proposal_returns_none(self):
        llm = _FakeLLMProvider(intent="general", confidence=0.9)
        repo = _FakePendingRepo(proposal=None)
        planner = LLMPlanner(llm_provider=llm, pending_repo=repo)
        state = _make_state("Yes")
        state["session_id"] = "sess-1"
        plan = await planner.create_plan(state)
        assert len(plan) == 1
        assert plan[0]["tool_name"] == "search_documents"


class TestConfirmationClassification:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("yes", True),
            ("Yeah, sure", True),
            ("ok", True),
            ("go ahead", True),
            ("no", False),
            ("nope", False),
            ("cancel", False),
            ("never mind", False),
            ("which meeting", None),
            ("What's the weather in Mumbai?", None),
        ],
    )
    @pytest.mark.asyncio
    async def test_confirmation_classification(self, text, expected):
        from orchestrator.planner.llm_planner import _classify_confirmation

        assert _classify_confirmation(text) is expected
