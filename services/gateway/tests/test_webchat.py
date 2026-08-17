"""Tests for the CH5 web chat routes (now redirected into the dashboard)."""

import pytest


@pytest.mark.asyncio
async def test_chat_page_redirects_to_dashboard(client, monkeypatch):
    monkeypatch.setattr(
        "gateway.routers.webchat._DASHBOARD_URL", "http://localhost:8001/dashboard"
    )
    response = await client.get("/chat", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "http://localhost:8001/dashboard"


@pytest.mark.asyncio
async def test_root_redirects_to_dashboard(client, monkeypatch):
    monkeypatch.setattr(
        "gateway.routers.webchat._DASHBOARD_URL", "http://localhost:8001/dashboard"
    )
    response = await client.get("/", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "http://localhost:8001/dashboard"
