import pytest

from rag.schemas.documents import DocumentUpload


class _ZeroEmbedder:
    @property
    def dimension(self) -> int:
        return 768

    async def embed(self, texts):
        return [[0.0] * 768 for _ in texts]


class _ShortEmbedder:
    @property
    def dimension(self) -> int:
        return 768

    async def embed(self, texts):
        return [[0.1] * 768] if texts else []


class TestIngestFailurePaths:
    @pytest.mark.asyncio
    async def test_all_zero_embeddings_marks_document_failed(self, mock_service):
        mock_service._embedder = _ZeroEmbedder()
        resp = await mock_service.ingest_document(DocumentUpload(
            title="t",
            content="Some content that will be chunked into a paragraph.",
        ))
        assert resp.status == "failed"

    @pytest.mark.asyncio
    async def test_embedder_length_mismatch_marks_document_failed(self, mock_service):
        mock_service._embedder = _ShortEmbedder()
        content = "This sentence is long enough to produce several chunks. " * 60
        resp = await mock_service.ingest_document(DocumentUpload(
            title="t",
            content=content,
        ))
        assert resp.status == "failed"

    @pytest.mark.asyncio
    async def test_usage_recorder_failure_does_not_fail_document(self, mock_service):
        class BoomRecorder:
            async def record(self, *args, **kwargs):
                raise RuntimeError("db down")

        mock_service._usage = BoomRecorder()
        resp = await mock_service.ingest_document(DocumentUpload(
            title="t",
            content="Valid content that should ingest fine.",
        ))
        assert resp.status == "ready"
