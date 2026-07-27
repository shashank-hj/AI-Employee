import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class EventType(str, Enum):
    USER_CREATED = "user.created"
    USER_UPDATED = "user.updated"
    USER_DELETED = "user.deleted"
    EMPLOYEE_ONBOARDED = "employee.onboarded"
    EMPLOYEE_OFFBOARDED = "employee.offboarded"
    TASK_CREATED = "task.created"
    TASK_COMPLETED = "task.completed"
    TASK_UPDATED = "task.updated"
    NOTIFICATION_SENT = "notification.sent"
    SYSTEM_EVENT = "system.event"


class Event(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: EventType
    payload: dict[str, Any] = {}
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source_service: str = ""


class EventBus:
    def __init__(self, redis_client):
        self.redis = redis_client
        self._handlers: dict[str, list] = {}

    async def publish(self, channel: str, event: Event) -> int:
        message = event.model_dump_json()
        return await self.redis.publish(channel, message)

    async def subscribe(self, channel: str, handler):
        if channel not in self._handlers:
            self._handlers[channel] = []
            pubsub = self.redis.pubsub()
            await pubsub.subscribe(channel)
        self._handlers[channel].append(handler)
