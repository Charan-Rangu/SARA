import sys
import os
from fastapi.testclient import TestClient

# Ensure backend package can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    
    data = response.json()
    assert data.get("status") == "ok", f"Expected status 'ok', got {data.get('status')}"
    assert data.get("project") == "SARA", f"Expected project 'SARA', got {data.get('project')}"
    assert data.get("version") == "0.1.0", f"Expected version '0.1.0', got {data.get('version')}"
