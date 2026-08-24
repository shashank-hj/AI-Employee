import pytest


class TestDocumentIngestion:
    @pytest.mark.asyncio
    async def test_ingest_document_success(self, client):
        response = await client.post("/api/documents", json={
            "title": "Test Doc",
            "content": "This is some test content that needs to be chunked.",
        })
        assert response.status_code == 201
        data = response.json()
        assert data["title"] == "Test Doc"
        assert data["status"] == "ready"
        assert data["chunks_count"] >= 1
        assert "id" in data
        assert "created_at" in data

    @pytest.mark.asyncio
    async def test_ingest_document_missing_title(self, client):
        response = await client.post("/api/documents", json={
            "content": "Some content",
        })
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_ingest_document_empty_content(self, client):
        response = await client.post("/api/documents", json={
            "title": "Empty",
            "content": "",
        })
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_ingest_document_with_metadata(self, client):
        response = await client.post("/api/documents", json={
            "title": "Meta Doc",
            "content": "Content with metadata.",
            "metadata": {"author": "test", "tags": ["important"]},
        })
        assert response.status_code == 201
        data = response.json()
        assert data["metadata"] == {"author": "test", "tags": ["important"]}

    @pytest.mark.asyncio
    async def test_ingest_long_document_multiple_chunks(self, client):
        paragraphs = [f"Paragraph {i}: " + "word " * 100 for i in range(20)]
        long_content = "\n\n".join(paragraphs)
        response = await client.post("/api/documents", json={
            "title": "Long Doc",
            "content": long_content,
        })
        assert response.status_code == 201
        data = response.json()
        assert data["chunks_count"] >= 2

    @pytest.mark.asyncio
    async def test_get_document(self, client):
        create = await client.post("/api/documents", json={
            "title": "Get Test",
            "content": "Content for get test.",
        })
        doc_id = create.json()["id"]
        response = await client.get(f"/api/documents/{doc_id}")
        assert response.status_code == 200
        assert response.json()["title"] == "Get Test"

    @pytest.mark.asyncio
    async def test_get_nonexistent_document(self, client):
        response = await client.get("/api/documents/nonexistent")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_list_documents(self, client):
        for i in range(3):
            await client.post("/api/documents", json={
                "title": f"Doc {i}",
                "content": f"Content {i}",
            })
        response = await client.get("/api/documents")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["data"]["total"] >= 3

    @pytest.mark.asyncio
    async def test_delete_document(self, client):
        create = await client.post("/api/documents", json={
            "title": "Delete Test",
            "content": "Content to be deleted.",
        })
        doc_id = create.json()["id"]
        delete = await client.delete(f"/api/documents/{doc_id}")
        assert delete.status_code == 204
        get_resp = await client.get(f"/api/documents/{doc_id}")
        assert get_resp.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_nonexistent_document(self, client):
        response = await client.delete("/api/documents/nonexistent")
        assert response.status_code == 404


class TestQuery:
    @pytest.mark.asyncio
    async def test_query_returns_results(self, client):
        await client.post("/api/documents", json={
            "title": "Query Test Doc",
            "content": "This document contains information about machine learning and artificial intelligence.",
        })
        response = await client.post("/api/query", json={
            "query": "What is machine learning?",
            "top_k": 3,
        })
        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "What is machine learning?"
        assert len(data["results"]) >= 1
        assert "total_found" in data

    @pytest.mark.asyncio
    async def test_query_result_structure(self, client):
        await client.post("/api/documents", json={
            "title": "Structure Test",
            "content": "Content for structure validation.",
        })
        response = await client.post("/api/query", json={
            "query": "structure validation",
        })
        data = response.json()
        for result in data["results"]:
            assert "chunk_id" in result
            assert "document_id" in result
            assert "document_title" in result
            assert "chunk_index" in result
            assert "content" in result
            assert "score" in result
            assert 0.0 <= result["score"] <= 1.0

    @pytest.mark.asyncio
    async def test_query_empty_string_rejected(self, client):
        response = await client.post("/api/query", json={
            "query": "",
        })
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_query_respects_top_k(self, client):
        await client.post("/api/documents", json={
            "title": "TopK Test",
            "content": "Content " * 500,
        })
        response = await client.post("/api/query", json={
            "query": "test query",
            "top_k": 2,
        })
        data = response.json()
        assert len(data["results"]) <= 2

    @pytest.mark.asyncio
    async def test_query_no_documents(self, client):
        response = await client.post("/api/query", json={
            "query": "nothing here",
        })
        assert response.status_code == 200
        data = response.json()
        assert data["total_found"] >= 0


class TestTextChunker:
    def test_chunk_basic(self):
        from rag.services.pipeline import TextChunker
        chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        text = "Short text that fits in one chunk."
        chunks = chunker.chunk(text)
        assert len(chunks) == 1

    def test_chunk_multiple_paragraphs(self):
        from rag.services.pipeline import TextChunker
        chunker = TextChunker(chunk_size=50, chunk_overlap=10)
        text = "A" * 60 + "\n\n" + "B" * 60 + "\n\n" + "C" * 60
        chunks = chunker.chunk(text)
        assert len(chunks) >= 2

    def test_chunk_empty_text(self):
        from rag.services.pipeline import TextChunker
        chunker = TextChunker()
        chunks = chunker.chunk("")
        assert chunks == []

    def test_chunk_whitespace_only(self):
        from rag.services.pipeline import TextChunker
        chunker = TextChunker()
        chunks = chunker.chunk("   \n\n   ")
        assert chunks == []

    def test_chunker_validation(self):
        from rag.services.pipeline import TextChunker
        with pytest.raises(ValueError):
            TextChunker(chunk_size=100, chunk_overlap=100)

    def test_chunk_overlap_preserved(self):
        from rag.services.pipeline import TextChunker
        chunker = TextChunker(chunk_size=50, chunk_overlap=10)
        text = "\n\n".join([f"Paragraph {i}: " + "X" * 45 for i in range(10)])
        chunks = chunker.chunk(text)
        assert len(chunks) >= 3

    def test_chunk_splits_oversized_single_paragraph(self):
        from rag.services.pipeline import TextChunker
        chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        text = "Y" * 250  # single paragraph far larger than chunk_size
        chunks = chunker.chunk(text)
        assert len(chunks) >= 3
        assert all(len(c) <= 100 for c in chunks)

    def test_chunk_mixed_oversized_and_normal_paragraphs(self):
        from rag.services.pipeline import TextChunker
        chunker = TextChunker(chunk_size=50, chunk_overlap=10)
        text = "Z" * 200 + "\n\n" + "A" * 30 + "\n\n" + "B" * 30
        chunks = chunker.chunk(text)
        assert len(chunks) >= 5
        assert all(len(c) <= 50 for c in chunks)


class TestEmbeddingProvider:
    def test_dimension(self):
        from rag.services.pipeline import MockEmbeddingProvider
        provider = MockEmbeddingProvider()
        assert provider.dimension == 768

    def test_embed_single(self):
        import asyncio

        from rag.services.pipeline import MockEmbeddingProvider
        provider = MockEmbeddingProvider()
        async def run():
            vectors = await provider.embed(["hello world"])
            return vectors
        vectors = asyncio.new_event_loop().run_until_complete(run())
        assert len(vectors) == 1
        assert len(vectors[0]) == 768

    def test_embed_deterministic(self):
        import asyncio

        from rag.services.pipeline import MockEmbeddingProvider
        provider = MockEmbeddingProvider()
        async def run():
            v1 = await provider.embed(["test"])
            v2 = await provider.embed(["test"])
            v3 = await provider.embed(["different"])
            return v1, v2, v3
        v1, v2, v3 = asyncio.new_event_loop().run_until_complete(run())
        assert v1 == v2
        assert v1 != v3

    def test_embed_batch(self):
        import asyncio

        from rag.services.pipeline import MockEmbeddingProvider
        provider = MockEmbeddingProvider()
        async def run():
            vectors = await provider.embed(["a", "b", "c"])
            return vectors
        vectors = asyncio.new_event_loop().run_until_complete(run())
        assert len(vectors) == 3
        assert all(len(v) == 768 for v in vectors)


class TestDocumentIngester:
    def test_ingester_delegates_to_chunker(self):
        from rag.services.pipeline import DocumentIngester, TextChunker
        chunker = TextChunker(chunk_size=500, chunk_overlap=50)
        ingester = DocumentIngester(chunker)
        text = "A" * 600 + "\n\n" + "B" * 600
        chunks = ingester.ingest(text)
        assert len(chunks) >= 2
