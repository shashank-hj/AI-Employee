"""Synthesize a natural-language answer from retrieved RAG chunks.

The generator hands the top-ranked chunks (each labeled with its source
document) to an LLM and asks it to answer the user's question *only* from
those chunks while explicitly naming the document each claim comes from.
This lets the RAG endpoint both "answer from previous documents" and tell the
user *which* document the answer was drawn from.
"""

from __future__ import annotations

import structlog

from rag.schemas.documents import AnswerResult, SearchResult, Source
from shared.llm.base import LLMProvider
from shared.usage.pricing import estimate_tokens
from shared.usage.recorder import UsageRecorder
from shared.usage.records import UsageRecord

logger = structlog.get_logger(__name__)

SYSTEM_PROMPT = (
    "You are a precise document assistant. You are given a user question and a "
    "set of text excerpts retrieved from a knowledge base of uploaded documents. "
    "Each excerpt is prefixed with its source document title in the form "
    "[doc:N] <title>.\n\n"
    "Rules:\n"
    "- Answer ONLY from the provided excerpts. Do not use outside knowledge.\n"
    "- For every claim, name the source document it came from, e.g. "
    "\"According to 'annual_report.pdf', ...\".\n"
    "- Give a complete, well-structured answer. Synthesize all relevant excerpts "
    "into a coherent explanation (define the subject, then cover its key points, "
    "features, and examples found in the excerpts). Do not merely list chunks.\n"
    "- If the excerpts do not contain the answer, say so clearly and list which "
    "documents (if any) were closest.\n"
    "- Be factual and accurate. Do not invent citations or details not present "
    "in the excerpts."
)


class AnswerGenerator:
    def __init__(
        self,
        llm: LLMProvider,
        usage_recorder: UsageRecorder | None = None,
    ) -> None:
        self._llm = llm
        self._usage = usage_recorder

    @property
    def _model(self) -> str:
        return getattr(self._llm, "_model", "") or "opencode"

    def _dedup(self, results: list[SearchResult]) -> list[SearchResult]:
        """Drop duplicate chunks (same content, e.g. the same PDF uploaded twice)
        so the LLM context and source list are clean."""
        seen: set[str] = set()
        deduped: list[SearchResult] = []
        for r in results:
            key = " ".join(r.content.split()).lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(r)
        return deduped

    def _build_user_message(self, query: str, results: list[SearchResult]) -> str:
        lines = [f"Question: {query}", "", "Excerpts:"]
        for i, r in enumerate(self._dedup(results), start=1):
            lines.append(
                f"[doc:{i}] {r.document_title} (score {r.score})\n"
                f"{r.content}\n"
            )
        return "\n".join(lines)

    @staticmethod
    def _sources(results: list[SearchResult]) -> list[Source]:
        # Collapse same-title documents (e.g. the same file uploaded twice):
        # keep the highest-scoring entry per title.
        seen: dict[str, Source] = {}
        for r in results:
            key = (r.document_title or "").strip().lower()
            existing = seen.get(key)
            if existing is not None and existing.score >= r.score:
                continue
            seen[key] = Source(
                document_id=r.document_id,
                document_title=r.document_title,
                snippet=" ".join(r.content.split())[:200],
                score=r.score,
            )
        return list(seen.values())

    async def generate(self, query: str, results: list[SearchResult]) -> AnswerResult:
        if not results:
            return AnswerResult(
                answer=(
                    "I could not find any matching content in the uploaded "
                    "documents to answer your question."
                ),
                sources=[],
            )

        user_message = self._build_user_message(query, results)
        response = await self._llm.generate(SYSTEM_PROMPT, user_message)

        if self._usage is not None:
            try:
                await self._usage.record(UsageRecord(
                    service="rag",
                    category="llm",
                    operation="generate_answer",
                    model=self._model,
                    unit="tokens",
                    input_units=estimate_tokens(SYSTEM_PROMPT + user_message),
                    output_units=estimate_tokens(response.content),
                    status="success",
                ))
            except Exception as exc:  # pragma: no cover - usage must never break the answer
                logger.warning("answer_usage_record_failed", error=str(exc))

        logger.info(
            "answer_generated",
            query=query[:50],
            results_count=len(results),
            model=self._model,
        )
        return AnswerResult(
            answer=response.content.strip(),
            sources=self._sources(results),
        )
