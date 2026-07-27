import pytest


SESSION_PAYLOAD = {
    "user_id": "user-1",
    "messages": [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ],
    "context": {"topic": "greeting"},
}


class TestSessionEndpoints:
    @pytest.mark.asyncio
    async def test_create_session(self, client):
        response = await client.post("/memory/session", json=SESSION_PAYLOAD)
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] is not None
        assert data["message_count"] == 2
        assert len(data["messages"]) == 2

    @pytest.mark.asyncio
    async def test_get_session(self, client):
        create = await client.post("/memory/session", json=SESSION_PAYLOAD)
        sid = create.json()["session_id"]
        response = await client.get(f"/memory/session/{sid}")
        assert response.status_code == 200
        assert response.json()["session_id"] == sid
        assert response.json()["message_count"] == 2

    @pytest.mark.asyncio
    async def test_update_existing_session(self, client):
        create = await client.post("/memory/session", json=SESSION_PAYLOAD)
        sid = create.json()["session_id"]
        update = await client.post("/memory/session", json={
            "session_id": sid,
            "user_id": "user-1",
            "messages": [{"role": "user", "content": "Updated"}],
        })
        assert update.status_code == 200
        assert update.json()["message_count"] == 1

    @pytest.mark.asyncio
    async def test_get_nonexistent_session(self, client):
        response = await client.get("/memory/session/nonexistent")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_session(self, client):
        create = await client.post("/memory/session", json=SESSION_PAYLOAD)
        sid = create.json()["session_id"]
        delete = await client.delete(f"/memory/session/{sid}")
        assert delete.status_code == 204
        get_resp = await client.get(f"/memory/session/{sid}")
        assert get_resp.status_code == 404

    @pytest.mark.asyncio
    async def test_add_message_to_session(self, client):
        create = await client.post("/memory/session", json=SESSION_PAYLOAD)
        sid = create.json()["session_id"]
        response = await client.post(f"/memory/session/{sid}/message", json={
            "role": "user",
            "content": "A third message",
        })
        assert response.status_code == 200
        assert response.json()["message_count"] == 3

    @pytest.mark.asyncio
    async def test_empty_session_created(self, client):
        response = await client.post("/memory/session", json={
            "messages": [],
        })
        assert response.status_code == 200
        assert response.json()["message_count"] == 0

    @pytest.mark.asyncio
    async def test_invalid_message_role_rejected(self, client):
        response = await client.post("/memory/session", json={
            "messages": [{"role": "invalid", "content": "test"}],
        })
        assert response.status_code == 422


class TestLongTermEndpoints:
    LT_PAYLOAD = {
        "user_id": "user-1",
        "content": "The company was founded in 2020.",
        "memory_type": "fact",
        "importance": 0.8,
    }

    @pytest.mark.asyncio
    async def test_store_long_term(self, client):
        response = await client.post("/memory/long-term", json=self.LT_PAYLOAD)
        assert response.status_code == 201
        data = response.json()
        assert data["user_id"] == "user-1"
        assert data["memory_type"] == "fact"
        assert "id" in data
        assert "created_at" in data

    @pytest.mark.asyncio
    async def test_get_long_term(self, client):
        create = await client.post("/memory/long-term", json=self.LT_PAYLOAD)
        mid = create.json()["id"]
        response = await client.get(f"/memory/long-term/{mid}")
        assert response.status_code == 200
        assert response.json()["id"] == mid

    @pytest.mark.asyncio
    async def test_get_nonexistent_long_term(self, client):
        response = await client.get("/memory/long-term/nonexistent")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_list_long_term(self, client):
        await client.post("/memory/long-term", json=self.LT_PAYLOAD)
        await client.post("/memory/long-term", json={
            "user_id": "user-1",
            "content": "Another fact",
            "memory_type": "fact",
            "importance": 0.5,
        })
        response = await client.get("/memory/long-term?user_id=user-1")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["data"]["total"] >= 2

    @pytest.mark.asyncio
    async def test_delete_long_term(self, client):
        create = await client.post("/memory/long-term", json=self.LT_PAYLOAD)
        mid = create.json()["id"]
        delete = await client.delete(f"/memory/long-term/{mid}")
        assert delete.status_code == 204

    @pytest.mark.asyncio
    async def test_invalid_importance_rejected(self, client):
        response = await client.post("/memory/long-term", json={
            **self.LT_PAYLOAD, "importance": 1.5,
        })
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_missing_user_id_rejected(self, client):
        response = await client.post("/memory/long-term", json={
            "content": "test",
            "memory_type": "fact",
        })
        assert response.status_code == 422
