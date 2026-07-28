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
    DIMENSION = 1536

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
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(timeout),
        )
        self._dimension: int | None = None

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            return 768  # default before first call
        return self._dimension

    async def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
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
                vectors.append(embedding)
            except Exception as exc:
                logger.error(
                    "ollama_embedding_failed",
                    model=self._model,
                    text_len=len(text),
                    error=str(exc),
                )
                # fallback: zero vector
                dim = self._dimension or 768
                vectors.append([0.0] * dim)
        return vectors

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
