import math
import random
from abc import ABC, abstractmethod

import httpx
import structlog

logger = structlog.get_logger(__name__)


class BaseEmbeddingProvider(ABC):
    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]: ...

    @property
    @abstractmethod
    def dimension(self) -> int: ...


class MockEmbeddingProvider(BaseEmbeddingProvider):
    DIMENSION = 768

    @property
    def dimension(self) -> int:
        return self.DIMENSION

    async def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            random.seed(hash(text) & 0xFFFFFFFF)
            vec = [random.gauss(0, 1) for _ in range(self.DIMENSION)]
            norm = math.sqrt(sum(x * x for x in vec))
            vectors.append([x / norm for x in vec] if norm > 0 else [0.0] * self.DIMENSION)
        return vectors


class OllamaEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "nomic-embed-text",
        timeout: float = 30.0,
        batch_size: int = 64,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout
        self._batch_size = max(1, batch_size)
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(timeout),
            transport=transport,
        )
        self._dimension: int | None = None

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            return 768  # default before first call
        return self._dimension

    async def _embed_batch(self, texts: list[str]) -> list[list[float]] | None:
        """Embed a batch of texts with the Ollama `/api/embed` endpoint.

        Returns a list of vectors aligned to ``texts``, or ``None`` on failure.
        Falls back to the legacy single-prompt endpoint if `/api/embed` is not
        available (older Ollama versions).
        """
        try:
            response = await self._client.post(
                "/api/embed",
                json={"model": self._model, "input": texts},
            )
            response.raise_for_status()
            data = response.json()
            embeddings = data.get("embeddings") or []
            if embeddings and self._dimension is None:
                self._dimension = len(embeddings[0])
            if len(embeddings) == len(texts):
                return embeddings
            logger.warning(
                "ollama_embed_batch_mismatch",
                model=self._model,
                sent=len(texts),
                received=len(embeddings),
            )
            return None
        except Exception as exc:
            logger.warning(
                "ollama_embed_batch_failed",
                model=self._model,
                batch_size=len(texts),
                error=str(exc)[:300],
            )
            return None

    async def _embed_one(self, text: str) -> list[float]:
        """Legacy per-text fallback via `/api/embeddings`."""
        try:
            response = await self._client.post(
                "/api/embeddings",
                json={"model": self._model, "prompt": text},
            )
            response.raise_for_status()
            data = response.json()
            embedding = data.get("embedding", [])
            if embedding and self._dimension is None:
                self._dimension = len(embedding)
            return embedding
        except Exception as exc:
            logger.error(
                "ollama_embedding_failed",
                model=self._model,
                text_len=len(text),
                error=str(exc),
            )
            return self._zero_vector()

    def _zero_vector(self) -> list[float]:
        return [0.0] * (self._dimension or 768)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        vectors: list[list[float] | None] = [None] * len(texts)
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            batch_vectors = await self._embed_batch(batch)
            if batch_vectors is not None:
                vectors[start : start + len(batch)] = batch_vectors
            else:
                # Fall back to per-text embedding for this batch.
                for i, text in enumerate(batch):
                    vectors[start + i] = await self._embed_one(text)

        return [v if v is not None else self._zero_vector() for v in vectors]

    async def health_check(self) -> bool:
        try:
            response = await self._client.post(
                "/api/embeddings",
                json={"model": self._model, "prompt": "health check"},
            )
            return response.status_code == 200
        except Exception:
            return False

    async def close(self) -> None:
        await self._client.aclose()
