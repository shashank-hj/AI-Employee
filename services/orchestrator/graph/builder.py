from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import StateGraph, END, START

from shared.llm.base import LLMProvider

from orchestrator.graph.state import AgentState
from orchestrator.graph.nodes import (
    create_receive_node,
    create_context_node,
    create_plan_node,
    create_execute_node,
    create_tool_invoke_node,
    create_respond_node,
)
from orchestrator.graph.edges import should_continue, after_plan_route
from orchestrator.planner.base import BasePlanner
from orchestrator.tools.registry import ToolRegistry
from orchestrator.context.builder import ContextBuilder


def build_orchestrator_graph(
    tool_registry: ToolRegistry,
    planner: BasePlanner,
    context_builder: ContextBuilder,
    llm_provider: LLMProvider | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    approval_service=None,
) -> StateGraph:
    builder = StateGraph(AgentState)

    builder.add_node("receive", create_receive_node())
    builder.add_node("build_context", create_context_node(context_builder))
    builder.add_node("plan", create_plan_node(planner))
    builder.add_node("execute", create_execute_node(tool_registry, approval_service))
    builder.add_node("tool_invoke", create_tool_invoke_node(tool_registry))
    builder.add_node("respond", create_respond_node(llm_provider))

    builder.add_edge(START, "receive")
    builder.add_edge("receive", "build_context")
    builder.add_edge("build_context", "plan")

    builder.add_conditional_edges(
        "plan",
        after_plan_route,
        {
            "execute": "execute",
            "respond": "respond",
        },
    )

    builder.add_conditional_edges(
        "execute",
        should_continue,
        {
            "tool_invoke": "tool_invoke",
            "respond": "respond",
        },
    )
    builder.add_edge("tool_invoke", "execute")
    builder.add_edge("respond", END)

    kwargs: dict = {}
    if checkpointer is not None:
        kwargs["checkpointer"] = checkpointer
    return builder.compile(**kwargs)
