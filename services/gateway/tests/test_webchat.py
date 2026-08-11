"""Tests for the CH5 web chat page routes."""

import pytest


@pytest.mark.asyncio
async def test_chat_page_served(client):
    response = await client.get("/chat")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "AI Employee" in response.text
    assert "api/channels/web" in response.text


@pytest.mark.asyncio
async def test_root_redirects_to_chat(client):
    response = await client.get("/", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/chat"
