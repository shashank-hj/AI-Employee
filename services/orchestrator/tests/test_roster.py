from orchestrator.agents.roster import AGENT_ROSTER, AgentRoster, agent_roster


class TestAgentRoster:
    def test_list_includes_all_agents(self):
        names = {p["name"] for p in agent_roster.list()}
        assert names == set(AGENT_ROSTER.keys())
        assert {"sales", "support", "booking", "general", "complaint", "escalate"} <= names

    def test_get_returns_profile(self):
        profile = agent_roster.get("sales")
        assert profile is not None
        assert profile.name == "sales"
        assert "pricing" in profile.description.lower()

    def test_get_missing_returns_none(self):
        assert agent_roster.get("nonexistent") is None

    def test_resolve_for_tools_empty_returns_general(self):
        profile = agent_roster.resolve_for_tools([])
        assert profile.name == "general"

    def test_resolve_matches_sales(self):
        profile = agent_roster.resolve_for_tools(["search_pricing", "schedule_demo"])
        assert profile.name == "sales"

    def test_resolve_matches_support(self):
        profile = agent_roster.resolve_for_tools(["lookup_order"])
        assert profile.name == "support"

    def test_resolve_fallback_for_unknown_tools(self):
        profile = agent_roster.resolve_for_tools(["totally_unknown_tool"])
        assert profile.name == "general"

    def test_resolve_allows_send_email_on_general(self):
        profile = agent_roster.resolve_for_tools(["send_email"])
        assert profile.name == "general"
        assert "send_email" in profile.allowed_tools

    def test_support_does_not_allow_send_email(self):
        support = agent_roster.get("support")
        assert support is not None
        assert "send_email" not in support.allowed_tools

    def test_agent_fallback_constant(self):
        assert AgentRoster.agent_fallback == "general"
