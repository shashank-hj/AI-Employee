"""Tests for the generic shared Redis task queue."""

import asyncio

import pytest

from shared.queue import RedisTaskQueue, RedisTaskWorker


class FakeRedis:
    def __init__(self) -> None:
        self._lists: dict[str, list[str]] = {}

    async def rpush(self, key: str, value: str) -> int:
        self._lists.setdefault(key, []).append(value)
        return len(self._lists[key])

    async def lpop(self, key: str) -> str | None:
        lst = self._lists.get(key)
        if not lst:
            return None
        return lst.pop(0)

    async def llen(self, key: str) -> int:
        return len(self._lists.get(key, []))


class TestRedisTaskQueue:
    @pytest.mark.asyncio
    async def test_enqueue_and_poll_roundtrip(self):
        queue = RedisTaskQueue(FakeRedis(), queue_key="test:q")
        assert await queue.enqueue("greet", {"name": "Ada"}) is True
        job = await queue.poll()
        assert job["task"] == "greet"
        assert job["payload"] == {"name": "Ada"}
        assert job["job_id"]

    @pytest.mark.asyncio
    async def test_fifo_order(self):
        queue = RedisTaskQueue(FakeRedis(), queue_key="test:fifo")
        for i in range(3):
            await queue.enqueue("job", {"i": i})
        ids = []
        while (job := await queue.poll()) is not None:
            ids.append(job["payload"]["i"])
        assert ids == [0, 1, 2]

    @pytest.mark.asyncio
    async def test_empty_queue_polls_none(self):
        queue = RedisTaskQueue(FakeRedis(), queue_key="test:empty")
        assert await queue.poll() is None

    @pytest.mark.asyncio
    async def test_length(self):
        queue = RedisTaskQueue(FakeRedis(), queue_key="test:len")
        await queue.enqueue("a", {})
        await queue.enqueue("b", {})
        assert await queue.length() == 2

    @pytest.mark.asyncio
    async def test_bad_payload_returns_none(self):
        redis = FakeRedis()
        await redis.rpush("test:bad", "{not json")
        queue = RedisTaskQueue(redis, queue_key="test:bad")
        assert await queue.poll() is None


class TestRedisTaskWorker:
    @pytest.mark.asyncio
    async def test_dispatches_by_task_name(self):
        redis = FakeRedis()
        queue = RedisTaskQueue(redis, queue_key="test:worker")
        seen = []

        async def handler(payload):
            seen.append(payload)

        worker = RedisTaskWorker(queue, retry_delay=0.05)
        worker.register("greet", handler)
        worker.start()
        await queue.enqueue("greet", {"name": "Ada"})
        await asyncio.sleep(0.3)
        await worker.stop()
        assert seen == [{"name": "Ada"}]

    @pytest.mark.asyncio
    async def test_unknown_task_is_skipped(self):
        redis = FakeRedis()
        queue = RedisTaskQueue(redis, queue_key="test:skip")
        seen = []

        async def handler(payload):
            seen.append(payload)

        worker = RedisTaskWorker(queue, retry_delay=0.05)
        worker.register("known", handler)
        worker.start()
        await queue.enqueue("known", {"a": 1})
        await queue.enqueue("mystery", {"b": 2})
        await asyncio.sleep(0.4)
        await worker.stop()
        assert seen == [{"a": 1}]

    @pytest.mark.asyncio
    async def test_handler_error_does_not_kill_loop(self):
        redis = FakeRedis()
        queue = RedisTaskQueue(redis, queue_key="test:err")
        seen = []

        async def flaky(payload):
            if payload.get("i") == 0:
                raise RuntimeError("boom")
            seen.append(payload)

        worker = RedisTaskWorker(queue, retry_delay=0.05)
        worker.register("t", flaky)
        worker.start()
        await queue.enqueue("t", {"i": 0})
        await queue.enqueue("t", {"i": 1})
        await asyncio.sleep(0.4)
        await worker.stop()
        assert seen == [{"i": 1}]
