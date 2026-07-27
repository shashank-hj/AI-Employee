import json
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
from shared.llm.schemas import IntentClassificationResult

logger = structlog.get_logger(__name__)

DEFAULT_BASE_URL = "https://api.sarvam.ai"
DEFAULT_MODEL = "sarvam-105b"
DEFAULT_TIMEOUT = 30.0
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_TOKENS = 1024
MAX_RETRIES = 3

INTENT_SYSTEM_PROMPT = """You are an intent classification system for an AI employee platform.
Analyze the user's message and classify it into exactly one of these intents:

- sales: Product inquiries, pricing, purchasing, upgrades, trials, comparisons
- support: Order status, delivery tracking, returns, troubleshooting, account help
- booking: Appointments, demos, reservations, scheduling, calendar
- general: Greetings, FAQs, chitchat, math questions, factual queries, redirection
- complaint: Refund demands, legal threats, aggressive dissatisfaction, formal complaints
- escalate: Explicit "talk to human", "real person", "transfer me to an agent"

Available tools you can suggest:
- search_documents: Search the company knowledge base for policies, guides, documentation
- calculator: Evaluate mathematical expressions
- schedule_meeting: Schedule a meeting with date/time/attendees
- get_weather: Get current weather for a location
- send_email: Send an email (use sparingly, only when explicitly requested)

Classify these examples:

User: "My order ORD-7891 hasn't arrived yet — where is it?"
→ {"intent": "support", "confidence": 0.96, "requires_human": false, "reason": "User is asking about a specific order's delivery status", "entities": [{"name": "ORD-7891", "type": "order_id", "value": "ORD-7891"}], "suggested_tools": ["search_documents"]}

User: "What are your enterprise pricing plans?"
→ {"intent": "sales", "confidence": 0.94, "requires_human": false, "reason": "User is asking about pricing information for enterprise tier", "entities": [], "suggested_tools": ["search_documents"]}

User: "Schedule a demo for next Tuesday at 3pm"
→ {"intent": "booking", "confidence": 0.93, "requires_human": false, "reason": "User wants to schedule a demo at a specific date/time", "entities": [{"name": "next Tuesday 3pm", "type": "datetime", "value": "next Tuesday 3pm"}], "suggested_tools": ["schedule_meeting"]}

User: "I want to talk to a real person"
→ {"intent": "escalate", "confidence": 0.99, "requires_human": true, "reason": "User explicitly requests human agent", "entities": [], "suggested_tools": []}

User: "What is 5 + 5?"
→ {"intent": "general", "confidence": 0.98, "requires_human": false, "reason": "Simple math question", "entities": [], "suggested_tools": ["calculator"]}

User: "Hello there"
→ {"intent": "general", "confidence": 0.97, "requires_human": false, "reason": "Simple greeting", "entities": [], "suggested_tools": []}

User: "I'm very unhappy and want a full refund immediately"
→ {"intent": "complaint", "confidence": 0.92, "requires_human": true, "reason": "Aggressive demand for refund with dissatisfaction", "entities": [], "suggested_tools": []}

User: "What is the weather in Mumbai?"
→ {"intent": "general", "confidence": 0.90, "requires_human": false, "reason": "Weather query for a specific city", "entities": [{"name": "Mumbai", "type": "location", "value": "Mumbai"}], "suggested_tools": ["get_weather"]}

Return ONLY valid JSON with no markdown wrapping, no code fences, no additional text.
Use the exact schema shown in the examples above."""


class SarvamProvider(LLMProvider):
    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        timeout: float = DEFAULT_TIMEOUT,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "api-subscription-key": self._api_key,
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(timeout),
        )

    async def classify_intent(self, user_input: str) -> IntentClassification:
        start = time.perf_counter()
        logger.info("intent_classification_started", input_length=len(user_input))

        try:
            response = await self._chat_completion(
                system_prompt=INTENT_SYSTEM_PROMPT,
                user_message=user_input,
                expect_json=True,
            )
            parsed = self._parse_intent_response(response.content)

            duration_ms = (time.perf_counter() - start) * 1000
            logger.info(
                "intent_classification_completed",
                intent=parsed.intent,
                confidence=parsed.confidence,
                duration_ms=round(duration_ms, 2),
            )
            return parsed

        except Exception as exc:
            duration_ms = (time.perf_counter() - start) * 1000
            logger.error(
                "intent_classification_failed",
                error=str(exc),
                duration_ms=round(duration_ms, 2),
            )
            return IntentClassification(
                intent="general",
                confidence=0.3,
                reason=f"Classification failed: {str(exc)}",
            )

    async def generate(self, system_prompt: str, user_message: str) -> LLMResponse:
        start = time.perf_counter()
        logger.info("llm_generate_started")

        try:
            response = await self._chat_completion(
                system_prompt=system_prompt,
                user_message=user_message,
                expect_json=False,
            )
            return response
        except Exception as exc:
            duration_ms = (time.perf_counter() - start) * 1000
            logger.error("llm_generate_failed", error=str(exc))
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
        expect_json: bool = False,
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

        logger.debug("sarvam_api_request", model=self._model, message_count=len(messages))

        response = await self._client.post(
            "/v1/chat/completions",
            json=payload,
        )

        if response.status_code != 200:
            logger.error(
                "sarvam_api_error",
                status=response.status_code,
                body=response.text[:500],
            )
            response.raise_for_status()

        data = response.json()
        duration_ms = (time.perf_counter() - start) * 1000

        content = ""
        input_tokens = 0
        output_tokens = 0

        choices = data.get("choices", [])
        if choices:
            content = choices[0].get("message", {}).get("content") or ""

        usage = data.get("usage", {})
        if usage:
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)

        logger.debug(
            "sarvam_api_response",
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

    def _parse_intent_response(self, content: str) -> IntentClassification:
        if not content:
            return IntentClassification(
                intent="general",
                confidence=0.2,
                reason="Empty response from LLM",
            )

        text = content.strip()

        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            import re
            match = re.search(r"\{[\s\S]*\}", text)
            if match:
                try:
                    data = json.loads(match.group(0))
                except json.JSONDecodeError:
                    return IntentClassification(
                        intent="general",
                        confidence=0.2,
                        reason="Failed to parse LLM JSON response",
                    )
            else:
                return IntentClassification(
                    intent="general",
                    confidence=0.2,
                    reason="Failed to parse LLM JSON response",
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
