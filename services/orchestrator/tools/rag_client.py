from typing import Any, Protocol, runtime_checkable

import httpx
import structlog

logger = structlog.get_logger(__name__)


@runtime_checkable
class RAGClient(Protocol):
    async def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]: ...

    async def health_check(self) -> bool: ...


class MockRAGClient:
    _MOCK_DOCUMENTS: list[dict[str, Any]] = [
        {"title": "Company Policy - Remote Work", "snippet": "Employees may work remotely up to 3 days per week...", "score": 0.95},
        {"title": "Expense Reimbursement", "snippet": "Submit expenses within 30 days via the internal portal...", "score": 0.87},
        {"title": "Onboarding Guide", "snippet": "New hires must complete security training in the first week...", "score": 0.82},
        {"title": "API Documentation", "snippet": "All internal APIs use OAuth 2.0 for authentication...", "score": 0.78},
        {"title": "Code Review Guidelines", "snippet": "All PRs require at least one approval before merge...", "score": 0.74},
    ]

    async def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        q = query.lower() if query else ""
        scored: list[dict[str, Any]] = []
        for doc in self._MOCK_DOCUMENTS:
            content = (doc["title"] + " " + doc["snippet"]).lower()
            score = doc["score"] * (0.5 if q and q in content else 0.1)
            scored.append({**doc, "score": round(score, 3)})
        scored.sort(key=lambda d: d["score"], reverse=True)
        return scored[:top_k]

    async def health_check(self) -> bool:
        return True


class HttpRAGClient:
    def __init__(
        self,
        base_url: str,
        query_path: str = "/api/v1/documents/query",
        health_path: str = "/health",
        timeout: float = 5.0,
        fallback_client: RAGClient | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._query_path = query_path
        self._health_path = health_path
        self._timeout = timeout
        self._fallback = fallback_client

    async def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    f"{self._base_url}{self._query_path}",
                    json={"query": query, "top_k": top_k},
                )
                response.raise_for_status()
                data = response.json()
                results = data.get("results", [])
                logger.info(
                    "rag_query_success",
                    query=query[:80],
                    results_count=len(results),
                )
                return [
                    {
                        "title": r.get("document_title", r.get("title", "Untitled")),
                        "snippet": r.get("content", "")[:200],
                        "score": r.get("score", 0.0),
                    }
                    for r in results
                ]

        except Exception as exc:
            logger.warning(
                "rag_query_failed_falling_back",
                base_url=self._base_url,
                error=str(exc),
            )
            if self._fallback is not None:
                return await self._fallback.search(query, top_k)
            return [{"title": "Search Unavailable", "snippet": "The knowledge base is temporarily unavailable. Please try again shortly.", "score": 0.0}]

    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.get(f"{self._base_url}{self._health_path}")
                return response.status_code == 200
        except Exception:
            return False
