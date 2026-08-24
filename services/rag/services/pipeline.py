import re

from shared.llm.embedding_provider import BaseEmbeddingProvider, MockEmbeddingProvider


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
            if para_len > self._chunk_size:
                # A single paragraph larger than the chunk size must be split
                # (e.g. long PDF paragraphs / tables). Flush anything pending
                # first, then split this paragraph on its own.
                if current:
                    chunks.append("\n\n".join(current))
                    current = []
                    current_len = 0
                chunks.extend(self._split_long_paragraph(para))
                continue
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

    def _split_long_paragraph(self, para: str) -> list[str]:
        """Split an oversized paragraph into ``chunk_size``-bounded pieces with
        ``chunk_overlap`` between consecutive pieces."""
        pieces: list[str] = []
        start = 0
        n = len(para)
        while start < n:
            end = min(start + self._chunk_size, n)
            piece = para[start:end].strip()
            if piece:
                pieces.append(piece)
            if end >= n:
                break
            next_start = end - self._chunk_overlap
            if next_start <= start:
                next_start = end  # guarantee forward progress
            start = next_start
        return pieces

    def _split_paragraphs(self, text: str) -> list[str]:
        raw = re.split(r"\n\s*\n", text)
        return [p.strip() for p in raw if p.strip()]

    def _get_overlap(self, paragraphs: list[str]) -> str:
        combined = "\n\n".join(paragraphs)
        start = max(0, len(combined) - self._chunk_overlap)
        return combined[start:]


class DocumentIngester:
    def __init__(self, chunker: TextChunker) -> None:
        self._chunker = chunker

    def ingest(self, content: str) -> list[str]:
        return self._chunker.chunk(content)


__all__ = ["BaseEmbeddingProvider", "MockEmbeddingProvider", "TextChunker", "DocumentIngester"]
