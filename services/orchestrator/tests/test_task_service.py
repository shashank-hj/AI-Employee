from datetime import UTC, datetime

import pytest

from orchestrator.models.task import UserTask
from orchestrator.services.task_service import TaskService, _task_to_dict


class _FakeSession:
    """Minimal AsyncSession stand-in backed by an in-memory dict."""

    def __init__(self, rows: dict[str, UserTask], committed=None):
        self._rows = rows
        self._added: list[UserTask] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def add(self, obj):
        self._added.append(obj)

    async def commit(self):
        for obj in self._added:
            obj.id = obj.id or "t-1"
            obj.created_at = datetime.now(UTC)
            obj.updated_at = datetime.now(UTC)
            self._rows[obj.id] = obj
        self._added = []

    async def refresh(self, obj):
        pass

    async def get(self, model, task_id):
        return self._rows.get(task_id)

    async def execute(self, stmt):
        return _FakeResult(list(self._rows.values()))


class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def scalars(self):
        return self

    def all(self):
        return self._rows


class TestTaskService:
    def test_task_to_dict_shape(self):
        task = UserTask(
            id="t-1",
            title="Write report",
            session_id="s1",
            user_id="u1",
            status="pending",
            priority=2,
        )
        data = _task_to_dict(task)
        assert data["id"] == "t-1"
        assert data["title"] == "Write report"
        assert data["status"] == "pending"
        assert data["due_at"] is None

    @pytest.mark.asyncio
    async def test_create_persists_task(self):
        rows: dict[str, UserTask] = {}

        def factory():
            return _FakeSession(rows)

        svc = TaskService(session_factory=factory)
        task = await svc.create(
            title="Send invoice",
            session_id="s1",
            user_id="u1",
            due_at=datetime(2026, 8, 20, 10, 0, tzinfo=UTC),
        )
        assert task["id"] == "t-1"
        assert task["status"] == "pending"
        assert task["due_at"].startswith("2026-08-20")
        assert rows["t-1"].title == "Send invoice"

    @pytest.mark.asyncio
    async def test_complete_sets_status_and_completed_at(self):
        rows = {"t-1": UserTask(id="t-1", title="Task")}
        svc = TaskService(session_factory=lambda: _FakeSession(rows))
        task = await svc.complete("t-1")
        assert task["status"] == "completed"
        assert task["completed_at"] is not None
        assert rows["t-1"].status == "completed"

    @pytest.mark.asyncio
    async def test_update_fields(self):
        rows = {"t-1": UserTask(id="t-1", title="Task")}
        svc = TaskService(session_factory=lambda: _FakeSession(rows))
        task = await svc.update(
            "t-1",
            title="Renamed",
            status="in_progress",
            priority=5,
        )
        assert task["title"] == "Renamed"
        assert task["status"] == "in_progress"
        assert task["priority"] == 5

    @pytest.mark.asyncio
    async def test_update_rejects_invalid_status(self):
        rows = {"t-1": UserTask(id="t-1", title="Task")}
        svc = TaskService(session_factory=lambda: _FakeSession(rows))
        with pytest.raises(ValueError):
            await svc.update("t-1", status="bogus")

    @pytest.mark.asyncio
    async def test_get_missing_returns_none(self):
        svc = TaskService(session_factory=lambda: _FakeSession({}))
        assert await svc.get("nope") is None

    @pytest.mark.asyncio
    async def test_list_returns_rows(self):
        rows = {
            "t-1": UserTask(id="t-1", title="A", status="pending"),
            "t-2": UserTask(id="t-2", title="B", status="completed"),
        }
        svc = TaskService(session_factory=lambda: _FakeSession(rows))
        tasks = await svc.list()
        assert len(tasks) == 2
