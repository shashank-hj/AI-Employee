import pytest


class TestSearchEndpoint:
    @pytest.mark.asyncio
    async def test_search_returns_results(self, client):
        for i in range(3):
            await client.post("/memory/long-term", json={
                "user_id": "user-1",
                "content": f"Fact number {i}",
                "memory_type": "fact",
            })
        response = await client.post("/memory/search", json={
            "query": "What facts do we have?",
            "top_k": 5,
        })
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 1

    @pytest.mark.asyncio
    async def test_search_empty_query_rejected(self, client):
        response = await client.post("/memory/search", json={
            "query": "",
        })
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_search_respects_top_k(self, client):
        for i in range(5):
            await client.post("/memory/long-term", json={
                "user_id": "user-1",
                "content": f"Item {i}",
                "memory_type": "fact",
            })
        response = await client.post("/memory/search", json={
            "query": "items",
            "top_k": 2,
        })
        data = response.json()
        assert len(data) <= 2

    @pytest.mark.asyncio
    async def test_search_filters_by_user(self, client):
        await client.post("/memory/long-term", json={
            "user_id": "user-a",
            "content": "User A memory",
            "memory_type": "fact",
        })
        response = await client.post("/memory/search", json={
            "query": "memory",
            "user_id": "user-b",
        })
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_search_result_has_score(self, client):
        await client.post("/memory/long-term", json={
            "user_id": "user-1",
            "content": "Test memory for scoring",
            "memory_type": "fact",
        })
        response = await client.post("/memory/search", json={
            "query": "scoring test",
            "top_k": 1,
        })
        data = response.json()
        if data:
            assert "score" in data[0]
            assert 0.0 <= data[0]["score"] <= 1.0


class TestConversationEndpoints:
    @pytest.mark.asyncio
    async def test_store_message(self, client):
        response = await client.post("/memory/conversation", json={
            "session_id": "sess-1",
            "user_id": "user-1",
            "role": "user",
            "content": "Hello, world!",
        })
        assert response.status_code == 201
        data = response.json()
        assert data["session_id"] == "sess-1"
        assert data["role"] == "user"

    @pytest.mark.asyncio
    async def test_get_conversation(self, client):
        for i in range(3):
            await client.post("/memory/conversation", json={
                "session_id": "sess-2",
                "role": "user" if i % 2 == 0 else "assistant",
                "content": f"Message {i}",
            })
        response = await client.get("/memory/conversation/sess-2")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 3

    @pytest.mark.asyncio
    async def test_get_empty_conversation(self, client):
        response = await client.get("/memory/conversation/nonexistent-session")
        assert response.status_code == 200
        assert response.json() == []

    @pytest.mark.asyncio
    async def test_invalid_role_rejected(self, client):
        response = await client.post("/memory/conversation", json={
            "session_id": "sess-1",
            "role": "invalid-role",
            "content": "test",
        })
        assert response.status_code == 422


class TestProfileEndpoints:
    PROFILE = {
        "user_id": "user-1",
        "display_name": "Alice",
        "preferences": {"theme": "dark", "language": "en"},
    }

    @pytest.mark.asyncio
    async def test_upsert_profile(self, client):
        response = await client.put("/memory/profile", json=self.PROFILE)
        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == "user-1"
        assert data["display_name"] == "Alice"
        assert data["preferences"]["theme"] == "dark"

    @pytest.mark.asyncio
    async def test_get_profile(self, client):
        await client.put("/memory/profile", json=self.PROFILE)
        response = await client.get("/memory/profile/user-1")
        assert response.status_code == 200
        assert response.json()["display_name"] == "Alice"

    @pytest.mark.asyncio
    async def test_get_nonexistent_profile(self, client):
        response = await client.get("/memory/profile/ghost-user")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_update_profile(self, client):
        await client.put("/memory/profile", json=self.PROFILE)
        response = await client.patch("/memory/profile/user-1", json={
            "display_name": "Alice Updated",
            "preferences": {"theme": "light"},
        })
        assert response.status_code == 200
        data = response.json()
        assert data["display_name"] == "Alice Updated"
        assert data["preferences"]["theme"] == "light"

    @pytest.mark.asyncio
    async def test_missing_user_id_rejected(self, client):
        response = await client.put("/memory/profile", json={
            "display_name": "Bob",
        })
        assert response.status_code == 422


class TestEmbeddingService:
    def test_embedding_dimensions(self):
        from memory.services.stores import MockEmbeddingService
        svc = MockEmbeddingService()
        async def run():
            vec = await svc.embed("hello world")
            return vec
        import asyncio
        vec = asyncio.new_event_loop().run_until_complete(run())
        assert len(vec) == 1536
        assert all(isinstance(x, float) for x in vec)

    def test_embedding_deterministic(self):
        from memory.services.stores import MockEmbeddingService
        svc = MockEmbeddingService()
        async def run():
            v1 = await svc.embed("test text")
            v2 = await svc.embed("test text")
            v3 = await svc.embed("different text")
            return v1, v2, v3
        import asyncio
        v1, v2, v3 = asyncio.new_event_loop().run_until_complete(run())
        assert v1 == v2
        assert v1 != v3
