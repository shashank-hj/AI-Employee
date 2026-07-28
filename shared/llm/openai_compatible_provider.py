import json
import re
import time
from typing import Any

import httpx
import structlog
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

from shared.llm.base import IntentClassification, LLMProvider, LLMResponse
from shared.llm.prompts import INTENT_SYSTEM_PROMPT
from shared.llm.schemas import IntentClassificationResult

logger = structlog.get_logger(__name__)

DEFAULT_TIMEOUT = 30.0
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_TOKENS = 1024
MAX_RETRIES = 3


def _is_retryable(exception: Exception) -> bool:
    if isinstance(exception, (httpx.TimeoutException, httpx.ConnectError)):
        return True
    if isinstance(exception, httpx.HTTPStatusError):
        return exception.response.status_code not in (400, 401, 403, 404)
    return False


class OpenAICompatibleProvider(LLMProvider):
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "",
        timeout: float = DEFAULT_TIMEOUT,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout
        self._temperature = temperature
        self._max_tokens = max_tokens

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        if extra_headers:
            headers.update(extra_headers)

        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers=headers,
            timeout=httpx.Timeout(timeout),
        )

    async def classify_intent(self, user_input: str) -> IntentClassification:
        start = time.perf_counter()
        logger.info("intent_classification_started", input_length=len(user_input), model=self._model)

        try:
            response = await self._chat_completion(
                system_prompt=INTENT_SYSTEM_PROMPT,
                user_message=user_input,
            )
            parsed = self._parse_intent_response(response.content)

            duration_ms = (time.perf_counter() - start) * 1000
            logger.info(
                "intent_classification_completed",
                intent=parsed.intent,
                confidence=parsed.confidence,
                duration_ms=round(duration_ms, 2),
                model=self._model,
            )
            return parsed

        except Exception as exc:
            duration_ms = (time.perf_counter() - start) * 1000
            logger.error(
                "intent_classification_failed",
                error=str(exc),
                duration_ms=round(duration_ms, 2),
                model=self._model,
            )
            return IntentClassification(
                intent="general",
                confidence=0.3,
                reason=f"Classification failed: {str(exc)}",
            )

    async def generate(self, system_prompt: str, user_message: str) -> LLMResponse:
        start = time.perf_counter()
        logger.info("llm_generate_started", model=self._model)

        try:
            return await self._chat_completion(
                system_prompt=system_prompt,
                user_message=user_message,
            )
        except Exception as exc:
            duration_ms = (time.perf_counter() - start) * 1000
            logger.error("llm_generate_failed", error=str(exc), model=self._model)
            if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 404:
                return LLMResponse(
                    content=(
                        f"I was unable to generate a response because the model '{self._model}' "
                        "was not found. Please ensure the model is installed: "
                        f"run 'ollama pull {self._model}' and try again."
                    ),
                    model=self._model,
                    duration_ms=round(duration_ms, 2),
                )
            return LLMResponse(
                content="I'm sorry, I encountered an error processing your request.",
                model=self._model,
                duration_ms=round(duration_ms, 2),
            )

    async def health_check(self) -> bool:
        try:
            response = await self._client.get(f"{self._base_url}/v1/models")
            return response.status_code == 200
        except Exception:
            try:
                response = await self._client.get(f"{self._base_url}/api/tags")
                return response.status_code == 200
            except Exception:
                return False

    async def close(self) -> None:
        await self._client.aclose()

    @retry(
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.TimeoutException, httpx.ConnectError)),
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        before_sleep=before_sleep_log(logger, "warning"),
        reraise=True,
    )
    async def _chat_completion(
        self,
        system_prompt: str,
        user_message: str,
    ) -> LLMResponse:
        start = time.perf_counter()

        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }

        logger.debug("llm_api_request", model=self._model, message_count=len(messages))

        response = await self._client.post(
            "/v1/chat/completions",
            json=payload,
        )

        if response.status_code != 200:
            logger.error(
                "llm_api_error",
                status=response.status_code,
                body=response.text[:500],
                model=self._model,
            )
            response.raise_for_status()

        data = response.json()
        duration_ms = (time.perf_counter() - start) * 1000

        content = ""
        input_tokens = 0
        output_tokens = 0

        choices = data.get("choices", [])
        if choices:
            message = choices[0].get("message", {})
            content = message.get("content") or ""
            if not content:
                content = message.get("reasoning", "") or ""
                if content:
                    logger.debug("llm_using_reasoning_as_content", content_len=len(content))

        usage = data.get("usage", {})
        if usage:
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)

        logger.debug(
            "llm_api_response",
            model=data.get("model", self._model),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            duration_ms=round(duration_ms, 2),
        )

        return LLMResponse(
            content=content,
            model=data.get("model", self._model),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            duration_ms=round(duration_ms, 2),
        )

    @staticmethod
    def _parse_intent_response(content: str) -> IntentClassification:
        if not content:
            return IntentClassification(
                intent="general", confidence=0.2, reason="Empty response from LLM",
            )

        text = content.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{[\s\S]*\}", text)
            if match:
                try:
                    data = json.loads(match.group(0))
                except json.JSONDecodeError:
                    return IntentClassification(
                        intent="general", confidence=0.2, reason="Failed to parse LLM JSON response",
                    )
            else:
                return IntentClassification(
                    intent="general", confidence=0.2, reason="Failed to parse LLM JSON response",
                )

        validated = IntentClassificationResult(**data)
        return IntentClassification(
            intent=validated.intent,
            confidence=validated.confidence,
            requires_human=validated.requires_human,
            reason=validated.reason,
            entities=[e.model_dump() for e in validated.entities],
            suggested_tools=validated.suggested_tools,
        )
