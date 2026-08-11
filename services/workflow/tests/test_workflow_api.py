import json
import uuid
from datetime import UTC, datetime

import pytest
from httpx import ASGITransport, AsyncClient

from workflow.app import create_app
from workflow.container import get_workflow_service
from workflow.graph.checkpointer import CheckpointEngine
from workflow.models.workflow import WorkflowModel
from workflow.services.workflow_service import WorkflowService


class FakeWorkflowRepository:
    def __init__(self) -> None:
        self._data: dict[uuid.UUID, WorkflowModel] = {}

    async def create(self, wf: WorkflowModel) -> WorkflowModel:
        if wf.id is None:
            wf.id = uuid.uuid4()
        now = datetime.now(UTC)
        wf.created_at = wf.created_at or now
        wf.updated_at = wf.updated_at or now
        self._data[wf.id] = wf
        return wf

    async def get_by_id(self, workflow_id: str) -> WorkflowModel | None:
        for key, wf in self._data.items():
            if str(key) == workflow_id:
                return wf
        return None

    async def commit(self) -> None:
        return None

    async def list_all(self, status: str | None = None, page: int = 1, page_size: int = 20):
        items = [w for w in self._data.values() if status is None or w.status == status]
        return items, len(items)

    async def update(self, wf: WorkflowModel) -> WorkflowModel:
        self._data[wf.id] = wf
        return wf

    async def delete(self, workflow_id: str) -> bool:
        for key in list(self._data):
            if str(key) == workflow_id:
                self._data.pop(key, None)
                return True
        return False


@pytest.fixture
async def client():
    app = create_app()
    engine = CheckpointEngine()
    service = WorkflowService(FakeWorkflowRepository(), checkpointer=engine)
    app.dependency_overrides[get_workflow_service] = lambda: service
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


async def _create_task(client: AsyncClient, steps: list[dict], name: str = "itest") -> str:
    resp = await client.post("/api/workflows", json={"name": name, "steps": steps})
    assert resp.status_code == 201, resp.text
    return resp.json()["id"]


_SIMPLE_STEP = {"name": "a", "type": "task", "handler": "echo", "params": {}}
_APPROVAL_STEP = {
    "name": "refund", "type": "task", "handler": "process_refund",
    "params": {}, "requires_approval": True,
}


@pytest.mark.asyncio
async def test_run_completes_task_chain(client):
    steps = [
        {"name": "a", "type": "task", "handler": "echo", "params": {"stage": "a"}, "next": "b"},
        {"name": "b", "type": "task", "handler": "echo", "params": {"stage": "b"}},
    ]
    wfid = await _create_task(client, steps)
    resp = await client.post(f"/api/workflows/{wfid}/run", json={"input_data": {"x": 1}})
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["interrupted"] is False
    assert data["workflow"]["status"] == "completed"
    assert data["outputs"]["a"]["echo"]["stage"] == "a"
    assert data["outputs"]["b"]["echo"]["stage"] == "b"
    assert data["current_step"] == "b"


@pytest.mark.asyncio
async def test_approval_interrupt_and_continue(client):
    steps = [
        {"name": "check", "type": "task", "handler": "echo", "params": {}, "next": "refund"},
        {
            "name": "refund",
            "type": "task",
            "handler": "process_refund",
            "params": {"amount": 10, "order_id": "ORD-9"},
            "requires_approval": True,
            "next": "notify",
        },
        {"name": "notify", "type": "task", "handler": "echo", "params": {"stage": "notify"}},
    ]
    wfid = await _create_task(client, steps, name="approve-gate")
    run = await client.post(f"/api/workflows/{wfid}/run", json={})
    data = run.json()
    assert data["interrupted"] is True
    assert data["workflow"]["status"] == "paused"
    assert data["current_step"] == "refund"
    assert "notify" not in data["outputs"]

    fetched = (await client.get(f"/api/workflows/{wfid}")).json()
    assert fetched["pending_approval"] is not None
    assert fetched["pending_approval"]["workflow_step"] == "refund"
    assert fetched["pending_approval"]["handler"] == "process_refund"

    cont = await client.post(
        f"/api/workflows/{wfid}/continue", json={"payload": {"approved": True}}
    )
    cd = cont.json()
    assert cont.status_code == 200, cont.text
    assert cd["workflow"]["status"] == "completed"
    assert cd["outputs"]["refund"]["status"] == "processed"
    assert cd["outputs"]["notify"]["echo"]["stage"] == "notify"
    fetched = (await client.get(f"/api/workflows/{wfid}")).json()
    assert fetched["pending_approval"] is None


@pytest.mark.asyncio
async def test_approval_rejection_skips_handler(client):
    steps = [
        {
            "name": "refund", "type": "task", "handler": "process_refund",
            "params": {}, "requires_approval": True,
        },
    ]
    wfid = await _create_task(client, steps, name="refund-reject")
    await client.post(f"/api/workflows/{wfid}/run", json={})
    cont = await client.post(
        f"/api/workflows/{wfid}/continue", json={"payload": {"approved": False, "reason": "nope"}}
    )
    cd = cont.json()
    assert cd["workflow"]["status"] == "completed"
    assert cd["outputs"]["refund"]["status"] == "rejected"


@pytest.mark.asyncio
async def test_branch_routes_by_input(client):
    steps = [
        {
            "name": "decide", "type": "branch", "field": "$input.sentiment",
            "branches": {"ok": "approve"}, "default": "hold",
        },
        {"name": "approve", "type": "task", "handler": "echo", "params": {"stage": "approved"}},
        {"name": "hold", "type": "task", "handler": "echo", "params": {"stage": "held"}},
    ]
    wfid = await _create_task(client, steps, name="branch-ok")
    resp = await client.post(f"/api/workflows/{wfid}/run", json={"input_data": {"sentiment": "ok"}})
    assert resp.json()["outputs"]["approve"]["echo"]["stage"] == "approved"


@pytest.mark.asyncio
async def test_stream_returns_sse_frames(client):
    steps = [{"name": "a", "type": "task", "handler": "echo", "params": {"stage": "a"}}]
    wfid = await _create_task(client, steps)
    frames = []
    async with client.stream(
        "POST", f"/api/workflows/{wfid}/run", json={"input_data": {}, "stream": True}
    ) as resp:
        assert resp.status_code == 200
        chunks = [c async for c in resp.aiter_lines()]
        for line in chunks:
            if line.startswith("data: "):
                frames.append(line[6:])
    assert frames, "no SSE frames"
    last = json.loads(frames[-1])
    assert last["type"] == "done"
    assert last["data"]["workflow"]["status"] == "completed"


@pytest.mark.asyncio
async def test_fan_out_runs_handlers_in_parallel(client):
    steps = [
        {
            "name": "fan",
            "type": "fan_out",
            "handlers": ["check_inventory", "check_pricing"],
            "params": {"sku": "SKU-1"},
            "next": "done",
        },
        {"name": "done", "type": "task", "handler": "echo", "params": {"stage": "done"}},
    ]
    wfid = await _create_task(client, steps, name="fan-out")
    resp = await client.post(f"/api/workflows/{wfid}/run", json={})
    data = resp.json()
    assert resp.status_code == 200, resp.text
    assert data["workflow"]["status"] == "completed"
    fan = data["outputs"]["fan"]
    assert fan["check_inventory"]["sku"] == "SKU-1"
    assert fan["check_pricing"]["sku"] == "SKU-1"
    assert fan["check_inventory"].get("in_stock") is not None
    assert fan["check_pricing"].get("price_inr") is not None


@pytest.mark.asyncio
async def test_run_on_paused_returns_409(client):
    steps = [_APPROVAL_STEP]
    wfid = await _create_task(client, steps, name="guard-paused")
    await client.post(f"/api/workflows/{wfid}/run", json={})
    assert (await client.get(f"/api/workflows/{wfid}")).json()["status"] == "paused"
    second = await client.post(f"/api/workflows/{wfid}/run", json={})
    assert second.status_code == 409, second.text


@pytest.mark.asyncio
async def test_continue_when_not_paused_returns_conflict(client):
    wfid = await _create_task(client, [_SIMPLE_STEP])
    await client.post(f"/api/workflows/{wfid}/run", json={})
    assert (await client.get(f"/api/workflows/{wfid}")).json()["status"] == "completed"
    cont = await client.post(
        f"/api/workflows/{wfid}/continue", json={"payload": {"approved": True}}
    )
    assert cont.status_code == 409, cont.text


@pytest.mark.asyncio
async def test_rerun_after_completion_resets_thread(client):
    wfid = await _create_task(client, [_SIMPLE_STEP])
    first = (await client.post(f"/api/workflows/{wfid}/run", json={})).json()
    assert first["workflow"]["status"] == "completed"
    second = await client.post(f"/api/workflows/{wfid}/run", json={})
    assert second.status_code == 200, second.text
    assert second.json()["workflow"]["status"] == "completed"


@pytest.mark.asyncio
async def test_delete_workflow(client):
    wfid = await _create_task(client, [_SIMPLE_STEP])
    resp = await client.delete(f"/api/workflows/{wfid}")
    assert resp.status_code == 204
    assert (await client.get(f"/api/workflows/{wfid}")).status_code == 404
