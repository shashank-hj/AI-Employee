"""Agentic query refinement for retrieval.

A small LLM step rewrites/expands the user query into a focused search query
before embedding. On any failure (or when no LLM is configured) the original
query is returned unchanged, so retrieval never degrades below baseline.
"""

import structlog

from shared.llm.base import LLMProvider

logger = structlog.get_logger(__name__)

_REFINE_SYSTEM_PROMPT = (
    "You rewrite user queries into focused search queries for document retrieval. "
    "Expand abbreviations, resolve pronouns, and add key domain synonyms. "
    "Reply with only the rewritten query, nothing else."
)


class QueryRefiner:
    def __init__(self, llm_provider: LLMProvider) -> None:
        self._llm = llm_provider

    async def refine(self, query: str) -> str:
        query = query.strip()
        if not query:
            return query
        try:
            response = await self._llm.generate(_REFINE_SYSTEM_PROMPT, query)
            refined = response.content.strip()
            if not refined:
                return query
            logger.info(
                "query_refined",
                original=query[:80],
                refined=refined[:80],
                model=response.model,
            )
            return refined
        except Exception as exc:
            logger.warning("query_refine_failed", error=str(exc)[:200], fallback=True)
            return query
