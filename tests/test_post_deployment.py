import os
import requests
import pytest

# Environment variables set dynamically in GitHub Actions
BASE_URL = os.getenv("ECS_APP_URL")  # e.g., http://<task-public-ip>:8080

@pytest.mark.smoke
def test_homepage_reachable():
    """
    Verify homepage is reachable after ECS deployment.
    """
    response = requests.get(f"{BASE_URL}/", timeout=10)
    assert response.status_code == 200
    assert "Document Portal" in response.text


@pytest.mark.smoke
def test_healthcheck_endpoint():
    """
    Verify /health endpoint for service health.
    """
    response = requests.get(f"{BASE_URL}/health", timeout=10)
    assert response.status_code == 200
    data = response.json()
    assert data.get("status") == "healthy"


@pytest.mark.smoke
def test_api_inference_response():
    """
    Ensure RAG inference endpoint returns valid response.
    """
    payload = {"query": "Summarize the device safety information."}
    response = requests.post(f"{BASE_URL}/api/infer", json=payload, timeout=20)
    assert response.status_code == 200
    result = response.json()
    assert "summary" in result
    assert isinstance(result["summary"], str)
    assert len(result["summary"]) > 10
