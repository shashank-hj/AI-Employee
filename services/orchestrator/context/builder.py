from abc import ABC, abstractmethod
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


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


class MemoryContextBuilder(ContextBuilder):
    """Builds context by fetching real session messages and long-term memories from the Memory Service."""

    MAX_CONTEXT_MESSAGES = 20
    MAX_MEMORIES = 5

    def __init__(self, memory_client):
        self._memory = memory_client

    async def build(self, user_input: str, user_id: str | None, session_id: str | None) -> dict[str, Any]:
        memory_context: list[dict[str, str]] = []
        user_preferences: dict[str, Any] = {}
        user_profile: dict[str, Any] | None = None

        if session_id:
            session = await self._memory.get_session(session_id)
            if session:
                messages = session.get("messages", [])
                recent = messages[-self.MAX_CONTEXT_MESSAGES:]
                memory_context = [{"role": m.get("role", "user"), "content": m.get("content", "")} for m in recent]
                logger.info("memory_context_loaded", session_id=session_id, message_count=len(recent))

        if user_id:
            user_profile = await self._memory.get_profile(user_id)
            if user_profile:
                user_preferences = user_profile.get("preferences") or {}
                logger.info("user_profile_loaded", user_id=user_id)

            memories = await self._memory.search_memories(user_input, user_id=user_id, top_k=self.MAX_MEMORIES)
            if memories:
                memory_context.append({"role": "system", "content": "Relevant past information:"})
                for m in memories:
                    memory_context.append({"role": "system", "content": f"- {m.get('content', '')}"})

        return {
            "memory_context": memory_context,
            "document_context": [],
            "user_preferences": user_preferences,
        }
