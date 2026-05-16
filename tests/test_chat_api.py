import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_chat_endpoint():
    response = client.post(
        "/api/v1/chat",
        json={
            "messages": [
                {"role": "user", "content": "I need a Python developer assessment."}
            ]
        }
    )
    print(response.status_code)
    print(response.json())

test_chat_endpoint()
