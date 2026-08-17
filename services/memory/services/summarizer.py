"""Session summarization providers.

A summarizer turns the most recent transcript of a session into a short digest
that is persisted on the session's ``context.summary`` and, optionally, as a
long-term ``summary`` memory.
"""

from abc import ABC, abstractmethod

import structlog

from shared.llm.base import LLMProvider

logger = structlog.get_logger(__name__)

_SUMMARY_SYSTEM_PROMPT = (
    "You are a conversation summarizer. Summarize the user-assistant transcript "
    "in 2-4 concise sentences. Capture the user's request, what was resolved, "
    "any decisions, commitments, and unresolved follow-ups. Use plain text, no "
    "bullets, no preamble."
)


class BaseSessionSummarizer(ABC):
    @abstractmethod
    async def summarize(self, transcript: str) -> str: ...


class MockSessionSummarizer(BaseSessionSummarizer):
    """Deterministic fallback digest (no LLM): head + tail of the transcript."""

    def __init__(self, max_lines: int = 8, tail_lines: int = 2) -> None:
        self._max_lines = max_lines
        self._tail_lines = tail_lines

    async def summarize(self, transcript: str) -> str:
        lines = [ln.strip() for ln in transcript.splitlines() if ln.strip()]
        if not lines:
            return ""
        head = lines[: self._max_lines]
        tail = lines[-self._tail_lines:] if len(lines) > self._max_lines else []
        parts = [*head, *(["..."] if tail else []), *tail]
        return "\n".join(parts)


class LLMSessionSummarizer(BaseSessionSummarizer):
    """LLM-backed summarizer; returns an empty string on any failure."""

    def __init__(self, llm: LLMProvider | None) -> None:
        self._llm = llm

    async def summarize(self, transcript: str) -> str:
        if self._llm is None or not transcript.strip():
            return ""
        try:
            response = await self._llm.generate(
                system_prompt=_SUMMARY_SYSTEM_PROMPT,
                user_message=transcript[:12000],
            )
            return (response.content or "").strip()
        except Exception as exc:
            logger.error("session_summarizer_failed", error=str(exc))
            return ""
