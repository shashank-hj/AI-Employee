import pytest
from langgraph.types import Command as LGCommand

from workflow.graph.builder import build_workflow_graph
from workflow.graph.checkpointer import CheckpointEngine
from workflow.graph.handlers import HANDLERS


@pytest.mark.asyncio
async def test_graph_runs_to_completion():
    engine = CheckpointEngine()
    await engine.setup()
    steps = [
        {
            "name": "start", "type": "task", "handler": "echo",
            "params": {"stage": "start"}, "next": "finish",
        },
        {"name": "finish", "type": "task", "handler": "echo", "params": {"stage": "finish"}},
    ]
    graph = build_workflow_graph(steps, HANDLERS, checkpointer=engine.saver)
    config = {"configurable": {"thread_id": "wf-test-1"}}
    result = await graph.ainvoke(
        {"workflow_id": "wf-test-1", "input_data": {"msg": "hi"}, "outputs": {}, "history": []},
        config=config,
    )
    assert result["outputs"]["start"]["echo"]["stage"] == "start"
    assert result["outputs"]["finish"]["echo"]["stage"] == "finish"
    assert result["current_step"] == "finish"


@pytest.mark.asyncio
async def test_interrupt_and_resume_at_approval_gate():
    engine = CheckpointEngine()
    await engine.setup()
    steps = [
        {
            "name": "before", "type": "task", "handler": "echo",
            "params": {"stage": "before"}, "next": "gate",
        },
        {
            "name": "gate",
            "type": "task",
            "handler": "echo",
            "params": {"stage": "gate"},
            "requires_approval": True,
            "next": "after",
        },
        {"name": "after", "type": "task", "handler": "echo", "params": {"stage": "after"}},
    ]
    graph = build_workflow_graph(steps, HANDLERS, checkpointer=engine.saver)
    config = {"configurable": {"thread_id": "wf-test-2"}}
    initial = {"workflow_id": "wf-test-2", "input_data": {}, "outputs": {}, "history": []}

    result = await graph.ainvoke(initial, config=config)
    assert "__interrupt__" in result
    assert result["outputs"]["before"]["echo"]["stage"] == "before"

    resumed = await graph.ainvoke(LGCommand(resume={"approved": True}), config=config)
    assert "__interrupt__" not in resumed
    assert resumed["outputs"]["gate"]["echo"]["stage"] == "gate"
    assert resumed["outputs"]["after"]["echo"]["stage"] == "after"


@pytest.mark.asyncio
async def test_branch_steps_route_by_field():
    engine = CheckpointEngine()
    await engine.setup()
    steps = [
        {
            "name": "decide", "type": "branch", "field": "$input.sentiment",
            "branches": {"ok": "approve"}, "default": "hold",
        },
        {"name": "approve", "type": "task", "handler": "echo", "params": {"stage": "approved"}},
        {"name": "hold", "type": "task", "handler": "echo", "params": {"stage": "held"}},
    ]
    graph = build_workflow_graph(steps, HANDLERS, checkpointer=engine.saver)

    ok_config = {"configurable": {"thread_id": "wf-branch-ok"}}
    ok_result = await graph.ainvoke(
        {
            "workflow_id": "wf-branch-ok",
            "input_data": {"sentiment": "ok"},
            "outputs": {},
            "history": [],
        },
        config=ok_config,
    )
    assert ok_result["outputs"]["approve"]["echo"]["stage"] == "approved"

    hold_config = {"configurable": {"thread_id": "wf-branch-hold"}}
    hold_result = await graph.ainvoke(
        {
            "workflow_id": "wf-branch-hold",
            "input_data": {"sentiment": "no"},
            "outputs": {},
            "history": [],
        },
        config=hold_config,
    )
    assert hold_result["outputs"]["hold"]["echo"]["stage"] == "held"


@pytest.mark.asyncio
async def test_fan_out_gathers_handler_outputs():
    engine = CheckpointEngine()
    await engine.setup()
    steps = [
        {
            "name": "fan",
            "type": "fan_out",
            "handlers": ["check_inventory", "check_pricing"],
            "params": {"sku": "SKU-X"},
            "next": "done",
        },
        {"name": "done", "type": "task", "handler": "echo", "params": {"stage": "done"}},
    ]
    graph = build_workflow_graph(steps, HANDLERS, checkpointer=engine.saver)
    config = {"configurable": {"thread_id": "wf-fan-1"}}
    result = await graph.ainvoke(
        {"workflow_id": "wf-fan-1", "input_data": {}, "outputs": {}, "history": []},
        config=config,
    )
    fan = result["outputs"]["fan"]
    assert fan["check_inventory"]["sku"] == "SKU-X"
    assert fan["check_pricing"]["sku"] == "SKU-X"
    assert result["outputs"]["done"]["echo"]["stage"] == "done"
    subs = {h["sub"] for h in result["history"] if h.get("sub")}
    assert subs == {"check_inventory", "check_pricing"}
