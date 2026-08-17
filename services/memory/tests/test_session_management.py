import pytest

SESSION_PAYLOAD = {
    "user_id": "user-sess",
    "messages": [
        {"role": "user", "content": "Remember the product is due Friday"},
        {"role": "assistant", "content": "Noted, Friday it is."},
    ],
}


class TestSessionManagement:
    @pytest.mark.asyncio
    async def test_list_sessions_empty(self, client):
        response = await client.get("/memory/session")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["data"]["total"] == 0
        assert data["data"]["items"] == []

    @pytest.mark.asyncio
    async def test_list_sessions_after_create(self, client):
        await client.post("/memory/session", json=SESSION_PAYLOAD)
        response = await client.get("/memory/session?user_id=user-sess")
        assert response.status_code == 200
        data = response.json()
        assert data["data"]["total"] == 1
        assert data["data"]["items"][0]["session_id"]
        assert "messages" not in data["data"]["items"][0]

    @pytest.mark.asyncio
    async def test_list_sessions_filters_by_user(self, client):
        await client.post("/memory/session", json=SESSION_PAYLOAD)
        await client.post("/memory/session", json={
            "user_id": "other-user",
            "messages": [{"role": "user", "content": "Hello"}],
        })
        response = await client.get("/memory/session?user_id=user-sess")
        assert response.json()["data"]["total"] == 1

    @pytest.mark.asyncio
    async def test_patch_session_state(self, client):
        create = await client.post("/memory/session", json=SESSION_PAYLOAD)
        sid = create.json()["session_id"]
        response = await client.patch(f"/memory/session/{sid}", json={
            "context": {"topic": "deadline"},
            "metadata": {"source": "api"},
        })
        assert response.status_code == 200
        data = response.json()
        assert data["context"]["topic"] == "deadline"
        assert data["metadata"]["source"] == "api"

    @pytest.mark.asyncio
    async def test_patch_nonexistent_session(self, client):
        response = await client.patch("/memory/session/nope", json={"context": {"a": 1}})
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_clear_session_messages(self, client):
        create = await client.post("/memory/session", json=SESSION_PAYLOAD)
        sid = create.json()["session_id"]
        response = await client.post(f"/memory/session/{sid}/clear")
        assert response.status_code == 200
        data = response.json()
        assert data["message_count"] == 0
        assert data["messages"] == []

    @pytest.mark.asyncio
    async def test_clear_nonexistent_session(self, client):
        response = await client.post("/memory/session/nope/clear")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_expire_session(self, client):
        create = await client.post("/memory/session", json=SESSION_PAYLOAD)
        sid = create.json()["session_id"]
        expire = await client.post(f"/memory/session/{sid}/expire")
        assert expire.status_code == 204
        get_resp = await client.get(f"/memory/session/{sid}")
        assert get_resp.status_code == 404

    @pytest.mark.asyncio
    async def test_expire_nonexistent_session(self, client):
        response = await client.post("/memory/session/nope/expire")
        assert response.status_code == 404


class TestSessionSummary:
    @pytest.mark.asyncio
    async def test_summarize_session(self, client, mock_service):
        create = await client.post("/memory/session", json=SESSION_PAYLOAD)
        sid = create.json()["session_id"]
        response = await client.post(f"/memory/session/{sid}/summary", json={
            "message_limit": 10,
            "store": True,
        })
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == sid
        assert data["summary"] != ""
        assert data["message_count"] == 2

        get_resp = await client.get(f"/memory/session/{sid}")
        assert get_resp.json()["context"]["summary"] == data["summary"]
        lt_items = await mock_service._lt_repo.list_by_user("user-sess")
        assert any(m.memory_type == "summary" for m in lt_items[0])

    @pytest.mark.asyncio
    async def test_summarize_without_store(self, client):
        create = await client.post("/memory/session", json=SESSION_PAYLOAD)
        sid = create.json()["session_id"]
        response = await client.post(f"/memory/session/{sid}/summary", json={"store": False})
        assert response.status_code == 200
        assert response.json()["summary"] != ""

    @pytest.mark.asyncio
    async def test_summarize_nonexistent_session(self, client):
        response = await client.post("/memory/session/nope/summary", json={})
        assert response.status_code == 404
