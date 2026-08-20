"""Cost estimation for Samvaad voice-agent usage.

Sarvam's Voice Agents platform bills per-turn usage (STT per audio second,
LLM per token, TTS per character), not a flat per-minute rate. Because the
agent harness re-sends the *entire growing transcript* as LLM input on every
turn, cost scales roughly with the square of the number of turns. The
estimator below reproduces that model from the interaction analytics
(turn count + duration) so the dashboard can show live rupees-per-session
without hitting Sarvam's billing endpoint (which is not exposed via API).

The figures are estimates, not Sarvam's ledger. They use the published list
rates (docs.sarvam.ai/api/getting-started/pricing) and two LLM scenarios:
the cheaper Sarvam-105B and the pricier auto-routed GLM-5.2.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import structlog

from orchestrator.config import settings

logger = structlog.get_logger(__name__)

# Published list rates (INR). STT is billed per second of audio (~30/hour).
STT_RS_PER_HOUR = 30.0

# Sarvam-105B: input/cached-input/output per 1M tokens.
LLM_105B_IN = 29.28
LLM_105B_OUT = 73.2

# GLM-5.2 (beta): input/output per 1M tokens; reasoning billed as output.
LLM_GLM_IN = 128.1
LLM_GLM_OUT = 402.6

# Rough token model: avg chars per message and tokens per char for Indic text.
AVG_CHARS_PER_MSG = 90.0
TOKENS_PER_CHAR = 0.35
AVG_OUT_TOKENS_PER_TURN = 40.0


def estimate_session_cost(turns: int, duration_seconds: float) -> dict[str, Any]:
    """Estimate the cost of one Samvaad session from turn count + duration.

    The harness re-sends the full transcript on every turn, so cumulative
    input tokens ~= (turns * (turns+1) / 2) * avg_msg_tokens. Output tokens
    are a small fixed amount per turn. STT is charged on the call audio
    duration (billed per second).
    """
    turns = max(int(turns or 0), 0)
    duration_seconds = max(float(duration_seconds or 0), 0.0)

    avg_msg_tokens = AVG_CHARS_PER_MSG * TOKENS_PER_CHAR
    input_tokens = round(turns * (turns + 1) / 2 * avg_msg_tokens)
    output_tokens = round(turns * AVG_OUT_TOKENS_PER_TURN)

    stt_rs = duration_seconds / 3600.0 * STT_RS_PER_HOUR

    cost_105b = (input_tokens / 1e6) * LLM_105B_IN + (output_tokens / 1e6) * LLM_105B_OUT + stt_rs
    cost_glm = (input_tokens / 1e6) * LLM_GLM_IN + (output_tokens / 1e6) * LLM_GLM_OUT + stt_rs

    return {
        "turns": turns,
        "duration_seconds": round(duration_seconds, 1),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "stt_rs": round(stt_rs, 2),
        "cost_105b_rs": round(cost_105b, 2),
        "cost_glm_rs": round(cost_glm, 2),
    }


def _analytics_base_url() -> str:
    org = settings.SAMVAAD_ORG_ID
    ws = settings.SAMVAAD_WORKSPACE_ID
    app = settings.SAMVAAD_AGENT_ID
    return (
        f"https://apps.sarvam.ai/api/analytics/v1/{org}/{ws}/{app}/interactions"
    )


class SamvaadUsageClient:
    """Pulls interaction analytics and estimates per-session + total cost."""

    def __init__(
        self, api_key: str, timeout: float = 30.0, transport: Any | None = None
    ) -> None:
        self._api_key = api_key
        self._timeout = timeout
        self._client = httpx.AsyncClient(timeout=timeout, transport=transport)

    async def _fetch_interactions(
        self, start: datetime, end: datetime, limit: int = 100
    ) -> list[dict[str, Any]]:
        params = {
            "start_datetime": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end_datetime": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "limit": str(limit),
        }
        headers = {"X-API-Key": self._api_key} if self._api_key else {}
        resp = await self._client.get(_analytics_base_url(), params=params, headers=headers)
        if resp.status_code == 401:
            raise PermissionError("Sarvam analytics returned 401 (bad API key)")
        if resp.status_code == 402:
            raise ConnectionError("Sarvam credits exhausted (402)")
        resp.raise_for_status()
        data = resp.json()
        return data.get("items") or []

    def _enabled(self) -> str | None:
        if not self._api_key:
            return "SAMVAAD_API_KEY is not set"
        if not all(
            [
                settings.SAMVAAD_ORG_ID,
                settings.SAMVAAD_WORKSPACE_ID,
                settings.SAMVAAD_AGENT_ID,
            ]
        ):
            return "SAMVAAD_ORG_ID / SAMVAAD_WORKSPACE_ID / SAMVAAD_AGENT_ID are not set"
        return None

    async def estimate_window(
        self, days: int = 14
    ) -> dict[str, Any]:
        """Estimate total cost across the last ``days`` days of interactions."""
        reason = self._enabled()
        if reason:
            return {
                "available": False,
                "reason": reason,
                "total_105b_rs": 0.0,
                "total_glm_rs": 0.0,
                "stt_rs": 0.0,
                "sessions": [],
                "session_count": 0,
                "days": days,
                "ts": int(time.time()),
            }

        end = datetime.now(UTC)
        start = end - timedelta(days=days)
        try:
            items = await self._fetch_interactions(start, end)
        except Exception as exc:
            logger.warning("samvaad_usage_fetch_failed", error=str(exc))
            return {
                "available": False,
                "reason": f"Analytics unavailable: {exc}",
                "total_105b_rs": 0.0,
                "total_glm_rs": 0.0,
                "stt_rs": 0.0,
                "sessions": [],
                "session_count": 0,
                "days": days,
                "ts": int(time.time()),
            }

        sessions: list[dict[str, Any]] = []
        total_105b = 0.0
        total_glm = 0.0
        total_stt = 0.0
        for it in items:
            turns = int(it.get("num_messages") or 0)
            dur = float(it.get("duration_in_seconds") or 0)
            est = estimate_session_cost(turns, dur)
            sessions.append(
                {
                    "interaction_id": it.get("interaction_id"),
                    "start_datetime": it.get("start_datetime"),
                    **est,
                }
            )
            total_105b += est["cost_105b_rs"]
            total_glm += est["cost_glm_rs"]
            total_stt += est["stt_rs"]

        sessions.sort(key=lambda s: s["cost_glm_rs"], reverse=True)
        return {
            "available": True,
            "reason": None,
            "total_105b_rs": round(total_105b, 2),
            "total_glm_rs": round(total_glm, 2),
            "stt_rs": round(total_stt, 2),
            "session_count": len(sessions),
            "sessions": sessions,
            "days": days,
            "ts": int(time.time()),
        }

    async def aclose(self) -> None:
        await self._client.aclose()


_usage_client: SamvaadUsageClient | None = None


async def get_usage_client() -> SamvaadUsageClient:
    """Module-level singleton so concurrent dashboard polls reuse one client."""
    global _usage_client
    if _usage_client is None:
        _usage_client = SamvaadUsageClient(api_key=settings.SAMVAAD_API_KEY)
    return _usage_client
