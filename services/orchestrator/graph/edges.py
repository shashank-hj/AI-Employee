from orchestrator.graph.state import AgentState


def after_plan_route(state: AgentState) -> str:
    if state.get("error"):
        return "respond"
    if state.get("final_response") is not None or not state.get("plan", []):
        return "respond"
    return "execute"


def should_continue(state: AgentState) -> str:
    plan = state["plan"]
    index = state["current_step_index"]

    if state.get("error"):
        return "respond"

    if state.get("awaiting_approval"):
        return "respond"

    if index >= len(plan):
        return "respond"

    # Loop back to the single execution node (which invokes the tool AND
    # advances the index). Keeping a second "tool_invoke" node here caused
    # every plan step except the first to run its tool twice.
    return "execute"