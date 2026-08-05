"""HTTP client for the Memory Service (M2, M3, M4 APIs)."""

from typing import Any

import httpx
import structlog

from orchestrator.config import settings

logger = structlog.get_logger(__name__)


class MemoryClient:
    """Async HTTP client for the Memory micro-service."""

    def __init__(
        self,
        base_url: str | None = None,
        timeout: float = 5.0,
    ) -> None:
        self._base_url = (base_url or settings.MEMORY_URL).rstrip("/")
        self._timeout = timeout
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(timeout),
        )

    async def upsert_profile(
        self,
        user_id: str,
        display_name: str | None = None,
        preferences: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Create or update a user profile in the memory service."""
        payload: dict[str, Any] = {"user_id": user_id}
        if display_name is not None:
            payload["display_name"] = display_name
        if preferences is not None:
            payload["preferences"] = preferences
        if metadata is not None:
            payload["metadata"] = metadata

        try:
            response = await self._client.put("/memory/profile", json=payload)
            response.raise_for_status()
            data = response.json()
            logger.info(
                "memory_client_profile_upserted",
                user_id=user_id,
                profile_id=data.get("id"),
            )
            return data
        except httpx.HTTPStatusError as exc:
            logger.warning(
                "memory_client_profile_failed",
                user_id=user_id,
                status=exc.response.status_code,
                detail=exc.response.text[:200],
            )
            return None
        except Exception as exc:
            logger.error("memory_client_profile_error", user_id=user_id, error=str(exc))
            return None

    async def update_profile(
        self,
        user_id: str,
        display_name: str | None = None,
        preferences: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Patch an existing user profile."""
        payload: dict[str, Any] = {}
        if display_name is not None:
            payload["display_name"] = display_name
        if preferences is not None:
            payload["preferences"] = preferences
        if metadata is not None:
            payload["metadata"] = metadata

        if not payload:
            return None

        try:
            response = await self._client.patch(f"/memory/profile/{user_id}", json=payload)
            response.raise_for_status()
            data = response.json()
            logger.info("memory_client_profile_updated", user_id=user_id)
            return data
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                # Profile does not exist — fall back to upsert
                return await self.upsert_profile(
                    user_id=user_id,
                    display_name=display_name,
                    preferences=preferences,
                    metadata=metadata,
                )
            logger.warning(
                "memory_client_profile_update_failed",
                user_id=user_id,
                status=exc.response.status_code,
            )
            return None
        except Exception as exc:
            logger.error("memory_client_profile_update_error", user_id=user_id, error=str(exc))
            return None

    async def store_long_term(
        self,
        user_id: str,
        content: str,
        memory_type: str = "fact",
        importance: float = 0.5,
        metadata: dict[str, Any] | None = None,
        source: str | None = None,
    ) -> dict[str, Any] | None:
        """Store a long-term memory (fact, preference, summary, etc.)."""
        payload = {
            "user_id": user_id,
            "content": content,
            "memory_type": memory_type,
            "importance": importance,
            "metadata": metadata,
            "source": source,
        }
        try:
            response = await self._client.post("/memory/long-term", json=payload)
            response.raise_for_status()
            data = response.json()
            logger.info(
                "memory_client_long_term_stored",
                user_id=user_id,
                memory_id=data.get("id"),
                memory_type=memory_type,
            )
            return data
        except httpx.HTTPStatusError as exc:
            logger.warning(
                "memory_client_long_term_failed",
                user_id=user_id,
                status=exc.response.status_code,
                detail=exc.response.text[:200],
            )
            return None
        except Exception as exc:
            logger.error("memory_client_long_term_error", user_id=user_id, error=str(exc))
            return None

    async def health_check(self) -> bool:
        """Ping the memory service health endpoint."""
        try:
            response = await self._client.get("/health", timeout=2.0)
            return response.status_code == 200
        except Exception:
            return False

    async def upsert_session(
        self,
        session_id: str,
        user_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Create or update a session in the memory service."""
        payload: dict[str, Any] = {"session_id": session_id}
        if user_id is not None:
            payload["user_id"] = user_id
        if metadata is not None:
            payload["metadata"] = metadata

        try:
            response = await self._client.post("/memory/session", json=payload)
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            logger.warning("memory_client_session_upsert_failed", session_id=session_id, error=str(exc))
            return None

    async def get_session(self, session_id: str) -> dict[str, Any] | None:
        """Get conversation messages for a session from PostgreSQL."""
        try:
            response = await self._client.get(f"/memory/conversation/{session_id}")
            response.raise_for_status()
            messages = response.json()
            return {"messages": messages} if isinstance(messages, list) else messages
        except Exception as exc:
            logger.warning("memory_client_get_session_failed", session_id=session_id, error=str(exc))
            return None

    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
    ) -> dict[str, Any] | None:
        """Store a conversation message in PostgreSQL."""
        try:
            response = await self._client.post(
                "/memory/conversation",
                json={"session_id": session_id, "role": role, "content": content},
            )
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            logger.warning("memory_client_add_message_failed", session_id=session_id, error=str(exc))
            return None

    async def get_profile(self, user_id: str) -> dict[str, Any] | None:
        """Get a user profile."""
        try:
            response = await self._client.get(f"/memory/profile/{user_id}")
            if response.status_code == 404:
                return None
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            logger.warning("memory_client_get_profile_failed", user_id=user_id, error=str(exc))
            return None

    async def search_memories(
        self,
        query: str,
        user_id: str | None = None,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Search long-term memories."""
        payload: dict[str, Any] = {"query": query, "top_k": top_k}
        if user_id is not None:
            payload["user_id"] = user_id

        try:
            response = await self._client.post("/memory/search", json=payload)
            response.raise_for_status()
            data = response.json()
            return data if isinstance(data, list) else data.get("results", [])
        except Exception as exc:
            logger.warning("memory_client_search_failed", query=query, error=str(exc))
            return []

    async def aclose(self) -> None:
        await self._client.aclose()
