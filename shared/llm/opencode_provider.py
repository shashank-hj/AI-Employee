"""opencode-backed LLM provider.

Calls the opencode HTTP server (``opencode serve``) as a plain chat/classifier
backend. The server exposes an OpenAPI spec at ``/doc``:

* ``GET /global/health``         — health/version
* ``POST /session``              — create a session
* ``POST /session/:id/message``  — send a message (blocks until assistant reply)

To keep opencode deterministic as an LLM engine (not an agent), messages are sent
with ``tools: []`` so it never attempts to call tools / modify the repo. A short
per-call session is created and discarded so context never leaks between turns.
"""

import base64
import json
import time
from collections.abc import Awaitable, Callable
from typing import Any

import httpx
import structlog

from shared.llm.base import IntentClassification, LLMProvider, LLMResponse
from shared.llm.openai_compatible_provider import (
    OpenAICompatibleProvider,
    DEFAULT_TIMEOUT,
)
from shared.llm.prompts import INTENT_CONTEXT_BLOCK, INTENT_SYSTEM_PROMPT
from shared.llm.schemas import IntentClassificationResult
from shared.usage.pricing import estimate_tokens
from shared.usage.records import UsageRecord

logger = structlog.get_logger(__name__)

DEFAULT_BASE_URL = "http://localhost:4096"
DEFAULT_AGENT = "general"
MAX_RETRIES = 3

# JSON Schema that mirrors IntentClassificationResult, used to request
# structured output from opencode for reliable intent parsing.
INTENT_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "intent": {"type": "string"},
        "confidence": {"type": "number"},
        "requires_human": {"type": "boolean"},
        "reason": {"type": "string"},
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "type": {"type": "string"},
                    "value": {"type": "string"},
                },
                "required": ["name", "type", "value"],
            },
        },
        "suggested_tools": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["intent", "confidence", "requires_human", "reason"],
}


class OpencodeProvider(LLMProvider):
    """LLMProvider backed by the opencode HTTP server."""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        model: str = "",
        agent: str = DEFAULT_AGENT,
        password: str | None = None,
        username: str = "opencode",
        timeout: float = DEFAULT_TIMEOUT,
        max_tokens: int = 1024,
        temperature: float = 0.1,
        usage_hook: Callable[[UsageRecord], Awaitable[None]] | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._agent = agent
        self._timeout = timeout
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._usage_hook = usage_hook

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if password:
            token = base64.b64encode(f"{username}:{password}".encode()).decode()
            headers["Authorization"] = f"Basic {token}"

        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers=headers,
            timeout=httpx.Timeout(timeout),
            transport=transport,
        )

    # ── LLMProvider contract ──────────────────────────────────────────

    async def health_check(self) -> bool:
        try:
            response = await self._client.get("/global/health")
            if response.status_code != 200:
                return False
            data = response.json()
            return bool(data.get("healthy", False))
        except Exception:
            return False

    async def generate(self, system_prompt: str, user_message: str) -> LLMResponse:
        start = time.perf_counter()
        logger.info("opencode_generate_started", agent=self._agent, model=self._model or "default")

        content = await self._message(system_prompt, user_message, operation="generate")

        duration_ms = (time.perf_counter() - start) * 1000
        await self._emit_usage(
            operation="generate",
            content=content,
            system_prompt=system_prompt,
            user_message=user_message,
            duration_ms=duration_ms,
        )
        logger.info(
            "opencode_generate_completed",
            model=self._model or "default",
            output_tokens=len(content.split()),
            duration_ms=round(duration_ms, 2),
        )
        return LLMResponse(
            content=content,
            model=self._model or "opencode",
            output_tokens=len(content.split()),
            duration_ms=round(duration_ms, 2),
        )

    async def classify_intent(
        self,
        user_input: str,
        context: str | None = None,
    ) -> IntentClassification:
        start = time.perf_counter()
        logger.info(
            "opencode_intent_classification_started",
            input_length=len(user_input),
            agent=self._agent,
        )

        try:
            user_message = (
                INTENT_CONTEXT_BLOCK.format(context=context, user_input=user_input)
                if context
                else user_input
            )
            content = await self._message(
                INTENT_SYSTEM_PROMPT,
                user_message,
                operation="classify_intent",
            )
            parsed = OpenAICompatibleProvider._parse_intent_response(content)

            duration_ms = (time.perf_counter() - start) * 1000
            logger.info(
                "opencode_intent_classification_completed",
                intent=parsed.intent,
                confidence=parsed.confidence,
                duration_ms=round(duration_ms, 2),
            )
            return parsed
        except Exception as exc:
            duration_ms = (time.perf_counter() - start) * 1000
            logger.error(
                "opencode_intent_classification_failed",
                error=str(exc),
                duration_ms=round(duration_ms, 2),
            )
            return IntentClassification(
                intent="general",
                confidence=0.3,
                reason=f"Classification failed: {str(exc)}",
            )

    async def close(self) -> None:
        await self._client.aclose()

    # ── opencode server interaction ───────────────────────────────────

    async def _message(
        self,
        system_prompt: str,
        user_message: str,
        operation: str = "generate",
        format_schema: dict[str, Any] | None = None,
    ) -> str:
        """Create a short-lived session, send one message, return assistant text."""
        created = await self._create_session()
        session_id = created.get("id") or created.get("sessionID") or ""
        try:
            logger.debug("opencode_session_created", session_id=session_id)
            parts = [{"type": "text", "text": user_message}]
            body: dict[str, Any] = {
                "system": system_prompt,
                "parts": parts,
                # opencode expects `tools` to be an object or null; omit it so the
                # server does not attempt any tool execution and opencode behaves
                # as a pure chat/classifier model rather than an agent.
                "agent": self._agent,
            }
            model_params = self._model_params()
            if model_params:
                body["model"] = model_params
            if format_schema:
                body["format"] = {
                    "type": "json_schema",
                    "schema": format_schema,
                    "retryCount": 2,
                }

            response = await self._post_message(session_id, body)
            data = response.json()
            content = self._extract_text(data)
            if not content:
                logger.warning(
                    "opencode_empty_reply",
                    session_id=session_id,
                    response_preview=str(data)[:300],
                )
            return content
        finally:
            try:
                await self._delete_session(session_id)
            except Exception:
                pass

    def _model_params(self) -> dict[str, str] | None:
        """Map the configured ``providerID/modelID`` string to the object the
        opencode server expects on the message payload.

        opencode 1.18+ validates ``model`` as an object (or null), not a bare
        string. A bare model id (no slash) is assumed to live on the built-in
        ``opencode`` provider.
        """
        if not self._model:
            return None
        provider_id, sep, model_id = self._model.partition("/")
        if not sep:
            provider_id = "opencode"
            model_id = self._model
        return {"providerID": provider_id, "modelID": model_id}

    async def _create_session(self) -> dict[str, Any]:
        # Do NOT pass a client-chosen `id` — opencode assigns its own internal
        # session id (ses_...) and passing an arbitrary id breaks the subsequent
        # message call with a 500 on the server side.
        response = await self._client.post(
            "/session",
            json={"title": "ai-employee-turn"},
        )
        response.raise_for_status()
        return response.json()

    async def _post_message(
        self,
        session_id: str,
        body: dict[str, Any],
    ) -> httpx.Response:
        last_exc: Exception | None = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = await self._client.post(
                    f"/session/{session_id}/message",
                    json=body,
                )
                if response.status_code != 200:
                    logger.error(
                        "opencode_api_error",
                        status=response.status_code,
                        body=response.text[:300],
                        attempt=attempt,
                    )
                    response.raise_for_status()
                return response
            except (httpx.TimeoutException, httpx.ConnectError) as exc:
                last_exc = exc
                logger.warning(
                    "opencode_retryable_error",
                    error=str(exc),
                    attempt=attempt,
                )
                if attempt < MAX_RETRIES:
                    import asyncio

                    await asyncio.sleep(2 ** (attempt - 1))
        if last_exc:
            raise last_exc
        raise RuntimeError("opencode request failed")

    async def _delete_session(self, session_id: str) -> None:
        response = await self._client.delete(f"/session/{session_id}")
        response.raise_for_status()

    @staticmethod
    def _extract_text(data: dict[str, Any]) -> str:
        """Extract assistant text from a ``{info, parts}`` message payload.

        Prefers ``info.structured_output`` (opencode's validated JSON when a
        ``format`` schema was requested); otherwise concatenates text parts.
        """
        info = data.get("info") or {}
        structured = info.get("structured_output")
        if structured:
            return json.dumps(structured, ensure_ascii=False)

        parts = data.get("parts") or []
        texts: list[str] = []
        for part in parts:
            if part.get("type") == "text":
                texts.append(part.get("text", ""))
        return "".join(texts)

    async def _emit_usage(
        self,
        *,
        operation: str,
        content: str,
        system_prompt: str,
        user_message: str,
        duration_ms: float,
    ) -> None:
        if self._usage_hook is None:
            return
        model_key = self._pricing_model_key()
        record = UsageRecord(
            category="llm",
            operation=operation,
            model=model_key,
            unit="tokens",
            input_units=estimate_tokens(system_prompt + user_message),
            output_units=estimate_tokens(content),
            duration_ms=round(duration_ms, 2),
        )
        try:
            await self._usage_hook(record)
        except Exception as exc:
            logger.debug("opencode_usage_hook_failed", error=str(exc))

    def _pricing_model_key(self) -> str:
        """Return a pricing-table key for the active model.

        OPENCODE_MODEL may be a bare id (``deepseek-v4-flash``) or a
        provider/model pair (``opencode-go/deepseek-v4-flash``). The pricing
        table keys on the model id, so we strip any provider prefix.
        """
        if not self._model:
            return "opencode"
        return self._model.split("/")[-1]
