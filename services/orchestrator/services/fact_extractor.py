"""Extract structured facts, preferences, and learnings from conversation transcripts via LLM."""

import json
import re
import structlog
from dataclasses import dataclass, field
from typing import Any

from shared.llm.base import LLMProvider

logger = structlog.get_logger(__name__)


def _extract_balanced_json(text: str) -> str:
    """Extract the first balanced JSON object from text, preferring code-fenced blocks."""
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1)

    start = text.find("{")
    if start == -1:
        return text

    depth = 0
    in_string = False
    escape_next = False

    for i in range(start, len(text)):
        ch = text[i]

        if escape_next:
            escape_next = False
            continue

        if ch == "\\":
            escape_next = True
            continue

        if ch == '"':
            in_string = not in_string
            continue

        if in_string:
            continue

        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    return text[start:]

EXTRACTION_SYSTEM_PROMPT = (
    "You are a structured fact extraction engine. Your job is to analyze a customer service "
    "conversation transcript and extract structured learnings about the user.\n\n"
    "Rules:\n"
    "1. Extract only factual, verifiable information — do not hallucinate.\n"
    "2. If the user mentions their name, extract it as display_name.\n"
    "3. Identify preferences (language, communication style, product interests, etc.).\n"
    "4. Note the overall sentiment of the conversation (positive, neutral, negative).\n"
    "5. Summarize the conversation in 1-2 sentences for episodic memory.\n"
    "6. List key topics discussed.\n"
    "7. Extract individual facts as short, self-contained sentences.\n\n"
    "Return ONLY a JSON object matching this schema — no extra commentary:\n"
    '{\n'
    '  "display_name": "string or null",\n'
    '  "preferences": {"key": "value", ...},\n'
    '  "facts": ["fact 1", "fact 2", ...],\n'
    '  "sentiment": "positive|neutral|negative",\n'
    '  "summary": "1-2 sentence summary",\n'
    '  "topics": ["topic 1", "topic 2", ...]\n'
    '}'
)


@dataclass
class ExtractedFacts:
    """Structured output from the fact extraction pipeline."""

    display_name: str | None = None
    preferences: dict[str, Any] = field(default_factory=dict)
    facts: list[str] = field(default_factory=list)
    sentiment: str = "neutral"
    summary: str = ""
    topics: list[str] = field(default_factory=list)


class FactExtractor:
    """Uses an LLM to extract structured facts from a conversation transcript."""

    def __init__(self, llm_provider: LLMProvider | None = None) -> None:
        self._llm = llm_provider

    async def extract(self, transcript: dict[str, Any]) -> ExtractedFacts:
        """Extract facts from a conversation transcript.\n\n"
        "Args:\n"
        "    transcript: dict with keys like user_input, final_response, tool_results, execution_log\n\n"
        "Returns:\n"
        "    ExtractedFacts dataclass\n"
        """
        if self._llm is None:
            logger.debug("fact_extractor_no_llm", message="LLM not configured; skipping extraction")
            return ExtractedFacts()

        user_message = self._build_transcript_prompt(transcript)

        try:
            response = await self._llm.generate(
                system_prompt=EXTRACTION_SYSTEM_PROMPT,
                user_message=user_message,
            )
            return self._parse_response(response.content)
        except Exception as exc:
            logger.error("fact_extractor_failed", error=str(exc))
            return ExtractedFacts()

    @staticmethod
    def _build_transcript_prompt(transcript: dict[str, Any]) -> str:
        parts: list[str] = []
        parts.append(f"User said: {transcript.get('user_input', '')}")

        final_response = transcript.get("final_response", "")
        if final_response:
            parts.append(f"Agent responded: {final_response}")

        tool_results = transcript.get("tool_results", [])
        if tool_results:
            tool_lines: list[str] = []
            for tr in tool_results:
                name = tr.get("tool_name", "unknown")
                if tr.get("success"):
                    data = tr.get("data", {})
                    tool_lines.append(f"  - {name}: {json.dumps(data, default=str)}")
                else:
                    tool_lines.append(f"  - {name}: FAILED — {tr.get('error', 'unknown error')}")
            parts.append("Tools used:\n" + "\n".join(tool_lines))

        return "\n\n".join(parts)

    @staticmethod
    def _parse_response(content: str) -> ExtractedFacts:
        """Parse JSON from LLM response, handling markdown fences and balanced braces."""
        json_str = _extract_balanced_json(content)

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as exc:
            logger.warning("fact_extractor_json_parse_failed", raw=content[:200], error=str(exc))
            return ExtractedFacts()

        return ExtractedFacts(
            display_name=data.get("display_name") or None,
            preferences=data.get("preferences") or {},
            facts=data.get("facts") or [],
            sentiment=data.get("sentiment", "neutral"),
            summary=data.get("summary", ""),
            topics=data.get("topics") or [],
        )
