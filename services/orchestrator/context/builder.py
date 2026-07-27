from abc import ABC, abstractmethod
from typing import Any


class ContextBuilder(ABC):
    @abstractmethod
    async def build(self, user_input: str, user_id: str | None, session_id: str | None) -> dict[str, Any]:
        """Return dict with 'memory_context', 'document_context', 'user_preferences'."""
        ...


class MockContextBuilder(ContextBuilder):
    async def build(self, user_input: str, user_id: str | None, session_id: str | None) -> dict[str, Any]:
        return {
            "memory_context": [
                {"role": "user", "content": "Previous conversation about project timelines."},
                {"role": "assistant", "content": "I helped schedule a review meeting for next Tuesday."},
            ] if session_id else [],
            "document_context": [],
            "user_preferences": {
                "timezone": "America/Los_Angeles",
                "language": "en",
                "notification_preference": "email",
            },
        }
