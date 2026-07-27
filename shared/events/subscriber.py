import asyncio
import json
import structlog
from redis.asyncio import Redis
from tenacity import retry, wait_exponential, stop_after_attempt

from shared.events.publisher import Event

logger = structlog.get_logger(__name__)


class EventSubscriber:
    def __init__(self, redis_client: Redis):
        self.redis = redis_client
        self._handlers: dict[str, list] = {}
        self._tasks: list[asyncio.Task] = []

    def register_handler(self, channel: str, handler):
        if channel not in self._handlers:
            self._handlers[channel] = []
        self._handlers[channel].append(handler)

    @retry(wait=wait_exponential(multiplier=1, min=2, max=30), stop=stop_after_attempt(5), reraise=True)
    async def _listen(self, channel: str):
        pubsub = self.redis.pubsub()
        await pubsub.subscribe(channel)
        logger.info("subscribed_to_channel", channel=channel)

        try:
            async for message in pubsub.listen():
                if message["type"] == "message":
                    data = json.loads(message["data"])
                    event = Event(**data)
                    for handler in self._handlers.get(channel, []):
                        try:
                            await handler(event)
                        except Exception:
                            logger.exception("event_handler_failed", channel=channel, event_id=event.id)
        except asyncio.CancelledError:
            await pubsub.unsubscribe(channel)
            raise

    async def start(self, channels: list[str] | None = None):
        if channels is None:
            channels = list(self._handlers.keys())

        for channel in channels:
            task = asyncio.create_task(self._listen(channel))
            self._tasks.append(task)

    async def stop(self):
        for task in self._tasks:
            task.cancel()
        self._tasks.clear()
