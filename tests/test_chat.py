"""
Tests for POST /api/v1/chat
"""

import pytest


@pytest.mark.asyncio
async def test_chat_returns_200(client):
    payload = {
        "messages": [
            {"role": "user", "content": "I need to assess Java developers."}
        ],
        "max_recommendations": 3,
    }
    response = await client.post("/api/v1/chat", json=payload)
    assert response.status_code == 200

    body = response.json()
    assert body["success"] is True
    assert "reply" in body["data"]
    assert isinstance(body["data"]["recommendations"], list)


@pytest.mark.asyncio
async def test_chat_rejects_empty_messages(client):
    payload = {"messages": []}
    response = await client.post("/api/v1/chat", json=payload)
    assert response.status_code == 422  # Pydantic validation error


@pytest.mark.asyncio
async def test_chat_rejects_invalid_role(client):
    payload = {
        "messages": [
            {"role": "invalid_role", "content": "Hello"}
        ],
    }
    response = await client.post("/api/v1/chat", json=payload)
    assert response.status_code == 422
