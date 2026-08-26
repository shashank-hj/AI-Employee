"""Headless bridge to a hosted Sarvam Samvaad voice agent.

Wraps ``sarvam_conv_ai_sdk.AsyncSamvaadAgent`` so the platform can open realtime
sessions with a Samvaad agent (``InteractionType.CALL`` for voice or ``CHAT``
for text) without owning any audio hardware. Server messages are normalised
into JSON-ready dicts and pushed onto a per-session ``outbox`` queue that REST
pollers and WebSocket proxies drain.

The SDK is imported lazily: when it is not installed or the platform is not
configured (``SAMVAAD_ENABLED`` / key / agent id), every capability reports
"disabled" instead of crashing — the same graceful-degradation pattern the RAG
and LLM providers follow.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from sarvam_conv_ai_sdk import AsyncSamvaadAgent, InteractionConfig
    from sarvam_conv_ai_sdk.messages.types import UserIdentifierType

logger = structlog.get_logger(__name__)

_SDK_AVAILABLE = True
try:
    from pydantic import SecretStr  # noqa: F401
    from sarvam_conv_ai_sdk import (
        AsyncSamvaadAgent,  # noqa: F401
        InteractionConfig,  # noqa: F401
        InteractionType,  # noqa: F401
        SarvamToolLanguageName,  # noqa: F401
    )
    from sarvam_conv_ai_sdk.messages.types import UserIdentifierType  # noqa: F401
except Exception as exc:  # pragma: no cover - exercised when SDK is absent
    _SDK_AVAILABLE = False
    _SDK_IMPORT_ERROR = str(exc)

SUPPORTED_SAMPLE_RATES = (16000, 48000)


class SamvaadError(RuntimeError):
    """Raised when the Samvaad integration cannot be used."""


def sdk_available() -> bool:
    return _SDK_AVAILABLE


def _resolve_language(name: str | None) -> Any | None:
    """Map a language name (e.g. "English", "Hindi") to SarvamToolLanguageName."""
    if not name or not _SDK_AVAILABLE:
        return None
    for member in SarvamToolLanguageName:
        if member.value.lower() == name.strip().lower():
            return member
    return None


class SamvaadSession:
    """One live conversation with the hosted Samvaad agent.

    Normalised server messages are pushed onto :attr:`outbox` as JSON-ready
    dicts::

        {"type": "text", "text": str, "status": "pending"|"completed"|"failed"}
        {"type": "transcript", "role": "user"|"bot", "content": str}
        {"type": "audio", "audio_base64": str, "format": str, "sample_rate": int|None}
        {"type": "event", "event": str, "interaction_id": str|None}
    """

    def __init__(
        self,
        session_id: str,
        agent: Any,
        mode: str,
        outbox: asyncio.Queue[dict[str, Any]],
        *,
        max_turns: int = 40,
        max_duration_s: int = 600,
    ) -> None:
        self.session_id = session_id
        self._agent = agent
        self.mode = mode  # "call" | "chat"
        self.outbox = outbox
        self.interaction_id: str | None = None
        self.created_at = time.time()
        self.turn_count = 0
        self.max_turns = max_turns
        self.max_duration_s = max_duration_s

    @property
    def limit_reached(self) -> str | None:
        """Reason the session should be closed, or None if within limits."""
        if self.max_turns and self.turn_count >= self.max_turns:
            return "max_turns"
        if (
            self.max_duration_s
            and (time.time() - self.created_at) >= self.max_duration_s
        ):
            return "max_duration"
        return None

    @property
    def connected(self) -> bool:
        try:
            return bool(self._agent.is_connected())
        except Exception:
            return False

    async def send_text(self, text: str) -> None:
        if not text.strip():
            raise ValueError("text must not be empty")
        await self._agent.send_text(text)

    async def send_audio(self, audio_bytes: bytes) -> None:
        if not audio_bytes:
            raise ValueError("audio bytes must not be empty")
        await self._agent.send_audio(audio_bytes)

    async def close(self) -> None:
        try:
            await self._agent.stop()
        except Exception as exc:
            logger.warning(
                "samvaad_session_close_error",
                session_id=self.session_id,
                error=str(exc),
            )

    def drain(self, limit: int = 200) -> list[dict[str, Any]]:
        """Pop up to ``limit`` pending outbox items without blocking."""
        items: list[dict[str, Any]] = []
        while len(items) < limit:
            try:
                items.append(self.outbox.get_nowait())
            except asyncio.QueueEmpty:
                break
        return items


class SamvaadSessionManager:
    """Opens and tracks Samvaad sessions against one configured agent."""

    def __init__(
        self,
        *,
        api_key: str,
        agent_id: str,
        org_id: str,
        workspace_id: str,
        base_url: str,
        sample_rate: int,
        default_language: str,
        version: int | None = None,
        connect_timeout: float = 15.0,
        enabled: bool = True,
        max_turns: int = 40,
        max_duration_s: int = 600,
    ) -> None:
        self._api_key = api_key
        self._agent_id = agent_id
        self._org_id = org_id
        self._workspace_id = workspace_id
        self._base_url = base_url.rstrip("/") + "/"
        self._sample_rate = sample_rate if sample_rate in SUPPORTED_SAMPLE_RATES else 16000
        self._default_language = default_language or "English"
        self._version = version
        self._connect_timeout = connect_timeout
        self._enabled = enabled
        self._max_turns = max_turns
        self._max_duration_s = max_duration_s
        self._sessions: dict[str, SamvaadSession] = {}

    @property
    def enabled(self) -> bool:
        return (
            self._enabled
            and _SDK_AVAILABLE
            and bool(self._api_key and self._agent_id)
        )

    def unavailable_reason(self) -> str | None:
        if not self._enabled:
            return "Samvaad is disabled (set SAMVAAD_ENABLED=true)"
        if not self._api_key:
            return "SAMVAAD_API_KEY is not set"
        if not self._agent_id:
            return "SAMVAAD_AGENT_ID is not set"
        if not _SDK_AVAILABLE:
            return f"sarvam-conv-ai-sdk unavailable: {_SDK_IMPORT_ERROR}"
        return None

    def active_sessions(self) -> int:
        return len(self._sessions)

    def get(self, session_id: str) -> SamvaadSession | None:
        return self._sessions.get(session_id)

    def list_sessions(self) -> list[SamvaadSession]:
        return list(self._sessions.values())

    async def open_session(
        self,
        *,
        user_identifier: str,
        mode: str = "chat",
        language: str | None = None,
        agent_variables: dict[str, Any] | None = None,
        session_id: str | None = None,
    ) -> SamvaadSession:
        if not self.enabled:
            raise SamvaadError(self.unavailable_reason() or "Samvaad disabled")
        if mode not in ("call", "chat"):
            raise ValueError("mode must be 'call' or 'chat'")

        session_id = session_id or f"samvaad-{uuid.uuid4().hex[:12]}"
        outbox: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=1000)
        lang = _resolve_language(language or self._default_language)

        from pydantic import SecretStr
        from sarvam_conv_ai_sdk import (
            AsyncSamvaadAgent,
            InteractionConfig,
            InteractionType,
        )
        from sarvam_conv_ai_sdk.messages.types import UserIdentifierType

        config: InteractionConfig = InteractionConfig(
            user_identifier_type=UserIdentifierType.CUSTOM,
            user_identifier=user_identifier,
            org_id=self._org_id,
            workspace_id=self._workspace_id,
            app_id=self._agent_id,
            version=self._version,
            interaction_type=(
                InteractionType.CALL if mode == "call" else InteractionType.CHAT
            ),
            sample_rate=self._sample_rate,
            agent_variables=agent_variables or {},
            initial_language_name=lang,
        )

        session = SamvaadSession(
            session_id=session_id,
            agent=None,
            mode=mode,
            outbox=outbox,
            max_turns=self._max_turns,
            max_duration_s=self._max_duration_s,
        )

        agent: AsyncSamvaadAgent = AsyncSamvaadAgent(
            api_key=SecretStr(self._api_key),
            config=config,
            base_url=self._base_url,
            audio_callback=lambda msg: self._on_audio(session, msg),
            text_callback=lambda msg: self._on_text(session, msg),
            transcript_callback=lambda msg: self._on_transcript(session, msg),
            event_callback=lambda event: self._on_event(session, event),
        )
        session._agent = agent

        try:
            await agent.start()
            connected = await agent.wait_for_connect(timeout=self._connect_timeout)
            if not connected:
                raise SamvaadError(
                    "Samvaad agent did not connect (check agent id, org/workspace, "
                    "and that the agent has a committed version)"
                )
        except SamvaadError:
            await session.close()
            raise
        except Exception as exc:
            await session.close()
            logger.error("samvaad_open_error", error=str(exc))
            raise SamvaadError(f"Failed to open Samvaad session: {exc}") from exc

        self._sessions[session_id] = session
        logger.info(
            "samvaad_session_opened",
            session_id=session_id,
            mode=mode,
            interaction_id=session.interaction_id,
        )
        return session

    async def close_session(self, session_id: str) -> bool:
        session = self._sessions.pop(session_id, None)
        if session is None:
            return False
        await session.close()
        logger.info("samvaad_session_closed", session_id=session_id)
        return True

    async def close_all(self) -> None:
        for session_id in list(self._sessions):
            await self.close_session(session_id)

    # ── callbacks ──

    async def _on_text(self, session: SamvaadSession, msg: Any) -> None:
        session.outbox.put_nowait(
            {
                "type": "text",
                "text": getattr(msg, "text", ""),
                "status": getattr(getattr(msg, "status", None), "value", "completed"),
            }
        )

    async def _on_transcript(self, session: SamvaadSession, msg: Any) -> None:
        role = getattr(getattr(msg, "role", None), "value", "bot")
        content = getattr(msg, "content", "")
        if role == "user":
            session.turn_count += 1
        session.outbox.put_nowait(
            {
                "type": "transcript",
                "role": role,
                "content": content,
            }
        )
        if reason := session.limit_reached:
            logger.info(
                "samvaad_session_limit",
                session_id=session.session_id,
                reason=reason,
                turns=session.turn_count,
            )
            session.outbox.put_nowait(
                {
                    "type": "event",
                    "event": "limit_reached",
                    "reason": reason,
                    "turns": session.turn_count,
                }
            )
            await session.close()

    async def _on_audio(self, session: SamvaadSession, msg: Any) -> None:
        audio_b64 = getattr(msg, "audio_base64", "")
        sample_rate = getattr(msg, "sample_rate", None)
        logger.info(
            "samvaad_audio_chunk",
            session_id=session.session_id,
            bytes=len(audio_b64),
            sample_rate=sample_rate,
        )
        session.outbox.put_nowait(
            {
                "type": "audio",
                "audio_base64": audio_b64,
                "format": getattr(getattr(msg, "format", None), "value", "audio/wav"),
                "sample_rate": sample_rate,
            }
        )

    async def _on_event(self, session: SamvaadSession, event: Any) -> None:
        interaction_id = getattr(event, "interaction_id", None)
        if interaction_id:
            session.interaction_id = interaction_id
        session.outbox.put_nowait(
            {
                "type": "event",
                "event": getattr(getattr(event, "type", None), "value", "unknown"),
                "interaction_id": interaction_id,
            }
        )
