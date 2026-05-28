from fastapi.testclient import TestClient

from api.main import app


def test_validation_error_uses_standard_error_schema():
    # Invalid symbol format triggers request validation before handler body.
    payload = {"symbols": ["BAD SYMBOL"], "model_type": "all", "walk_forward": False}
    with TestClient(app) as client:
        resp = client.post("/api/predictions/train", json=payload)

    assert resp.status_code == 422
    body = resp.json()
    assert body["error_code"] == "VALIDATION_ERROR"
    assert "message" in body
    assert isinstance(body.get("details"), list)
