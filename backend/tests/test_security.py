from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from moe_tools_suite.lab import MODEL_ID
from moe_tools_suite.main import create_app
from moe_tools_suite.settings import Settings
from pydantic import ValidationError

ACCESS_TOKEN = "correct-horse-battery-staple-token"


def _settings(tmp_path: Path, **overrides) -> Settings:
    return Settings(
        mode="mock",
        data_dir=tmp_path / "data",
        frontend_dist=tmp_path / "frontend",
        require_auth=True,
        auth_token=ACCESS_TOKEN,
        **overrides,
    )


def test_auth_token_is_required_and_must_be_strong(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="AUTH_TOKEN is required"):
        Settings(require_auth=True, data_dir=tmp_path)
    with pytest.raises(ValidationError, match="at least 24"):
        Settings(require_auth=True, auth_token="too-short", data_dir=tmp_path)
    with pytest.raises(ValidationError, match="unresolved placeholder"):
        Settings(
            require_auth=True,
            auth_token="{{ RUNPOD_SECRET_moe_tools_auth_token }}",
            data_dir=tmp_path,
        )


def test_login_cookie_and_csrf_protect_api_mutations(tmp_path: Path) -> None:
    client = TestClient(create_app(_settings(tmp_path)))

    session = client.get("/api/session")
    assert session.json() == {
        "auth_required": True,
        "authenticated": False,
        "csrf_token": None,
        "expires_at": None,
    }
    assert client.get("/api/models").status_code == 401
    assert client.post("/api/session/login", json={"token": "wrong"}).status_code == 401

    login = client.post("/api/session/login", json={"token": ACCESS_TOKEN})
    assert login.status_code == 200
    csrf_token = login.json()["csrf_token"]
    cookie = login.headers["set-cookie"]
    assert "HttpOnly" in cookie
    assert "SameSite=strict" in cookie
    assert client.get("/api/models").status_code == 200

    payload = {"model_id": MODEL_ID}
    assert client.post("/api/model-sessions", json=payload).status_code == 403
    loaded = client.post(
        "/api/model-sessions",
        json=payload,
        headers={"X-CSRF-Token": csrf_token},
    )
    assert loaded.status_code == 202

    logout = client.post("/api/session/logout", headers={"X-CSRF-Token": csrf_token})
    assert logout.status_code == 204
    assert client.get("/api/models").status_code == 401


def test_runpod_cookie_is_marked_secure(tmp_path: Path) -> None:
    client = TestClient(create_app(_settings(tmp_path, cookie_secure=True)))

    response = client.post("/api/session/login", json={"token": ACCESS_TOKEN})

    assert "Secure" in response.headers["set-cookie"]
