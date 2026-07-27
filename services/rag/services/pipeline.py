from abc import ABC, abstractmethod
import math
import random
import re
from typing import Optional


class TextChunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def chunk(self, text: str) -> list[str]:
        if not text.strip():
            return []

        paragraphs = self._split_paragraphs(text)
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0

        for para in paragraphs:
            para_len = len(para)
            if current_len + para_len > self._chunk_size and current:
                chunks.append("\n\n".join(current))
                overlap_text = self._get_overlap(current)
                current = [overlap_text] if overlap_text else []
                current_len = len(overlap_text) if overlap_text else 0
            current.append(para)
            current_len += para_len

        if current:
            chunks.append("\n\n".join(current))

        if not chunks:
            chunks.append(text[: self._chunk_size])

        return chunks

    def _split_paragraphs(self, text: str) -> list[str]:
        raw = re.split(r"\n\s*\n", text)
        return [p.strip() for p in raw if p.strip()]

    def _get_overlap(self, paragraphs: list[str]) -> str:
        combined = "\n\n".join(paragraphs)
        start = max(0, len(combined) - self._chunk_overlap)
        return combined[start:]


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


class DocumentIngester:
    def __init__(self, chunker: TextChunker) -> None:
        self._chunker = chunker

    def ingest(self, content: str) -> list[str]:
        return self._chunker.chunk(content)
