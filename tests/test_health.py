"""
Tests for GET /api/v1/health
"""

import pytest


@pytest.mark.asyncio
async def test_health_returns_200(client):
    response = await client.get("/api/v1/health")
    assert response.status_code == 200

    body = response.json()
    assert body["success"] is True
    assert body["data"]["status"] == "healthy"
    assert "version" in body["data"]
    assert "environment" in body["data"]
