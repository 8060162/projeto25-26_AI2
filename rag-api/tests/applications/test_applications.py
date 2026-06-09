"""
Testes do router de Applications

Cobertura:
- POST   /v1/applications          : criar, validação, sem admin
- GET    /v1/applications          : listar
- GET    /v1/applications/{id}     : obter, não encontrado
- PATCH  /v1/applications/{id}     : editar, não encontrado
- DELETE /v1/applications/{id}     : remover, não encontrado
- POST   /v1/applications/{id}/keys          : gerar key
- POST   /v1/applications/{id}/keys/{h}/rotate : rodar key
- DELETE /v1/applications/{id}/keys/{h}      : revogar key
"""
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from rag_api.api.main import create_app
from rag_api.auth.strategies import APIKeyAuthStrategy
from rag_api.schemas.applications import ApplicationRecord
from rag_api.schemas.identity import ClientIdentity

# ── Fixtures ──────────────────────────────────────────────────────────────────

ADMIN_IDENTITY = ClientIdentity(
    client_id  = "admin-client",
    scopes     = ["rag:admin", "rag:query"],
    rate_limit = 1000,
)

USER_IDENTITY = ClientIdentity(
    client_id  = "user-client",
    scopes     = ["rag:query"],   # sem rag:admin
    rate_limit = 100,
)

def make_app_record(app_id: str = "app-1") -> ApplicationRecord:
    return ApplicationRecord(
        id          = app_id,
        name        = "Test App",
        description = "App de teste",
        scopes      = ["rag:query"],
        rate_limit  = 100,
        active      = True,
        created_at  = datetime.now(timezone.utc),
        created_by  = "admin-client",
    )


def make_client(identity: ClientIdentity = ADMIN_IDENTITY):
    app = create_app()

    mock_strategy   = AsyncMock(spec=APIKeyAuthStrategy)
    mock_strategy.authenticate.return_value = identity

    mock_limiter        = MagicMock()
    mock_limiter.check  = AsyncMock()

    mock_app_repo       = AsyncMock()
    mock_key_repo       = AsyncMock()

    app.state.auth_strategy    = mock_strategy
    app.state.rate_limiter     = mock_limiter
    app.state.application_repo = mock_app_repo
    app.state.key_repo         = mock_key_repo
    app.state.rag_controller   = MagicMock()

    return TestClient(app, raise_server_exceptions=False), mock_app_repo, mock_key_repo


AUTH = {"Authorization": "Bearer rag_test"}


# ── POST /v1/applications ─────────────────────────────────────────────────────

def test_create_application_returns_201():
    client, repo, _ = make_client()
    repo.create = AsyncMock(return_value=make_app_record())

    resp = client.post("/v1/applications",
                       json={"name": "Test App", "scopes": ["rag:query"]},
                       headers=AUTH)
    assert resp.status_code == 201
    assert resp.json()["name"] == "Test App"


def test_create_application_without_admin_returns_403():
    client, _, _ = make_client(identity=USER_IDENTITY)
    resp = client.post("/v1/applications",
                       json={"name": "Test App"},
                       headers=AUTH)
    assert resp.status_code == 403
    assert resp.json()["error"] == "insufficient_scope"


def test_create_application_missing_name_returns_422():
    client, _, _ = make_client()
    resp = client.post("/v1/applications", json={}, headers=AUTH)
    assert resp.status_code == 422


# ── GET /v1/applications ──────────────────────────────────────────────────────

def test_list_applications_returns_200():
    client, repo, _ = make_client()
    repo.get_all = AsyncMock(return_value=[make_app_record("a1"), make_app_record("a2")])

    resp = client.get("/v1/applications", headers=AUTH)
    assert resp.status_code == 200
    assert resp.json()["total"] == 2


def test_list_applications_without_admin_returns_403():
    client, _, _ = make_client(identity=USER_IDENTITY)
    resp = client.get("/v1/applications", headers=AUTH)
    assert resp.status_code == 403


# ── GET /v1/applications/{id} ─────────────────────────────────────────────────

def test_get_application_returns_200():
    client, repo, _ = make_client()
    repo.get_by_id = AsyncMock(return_value=make_app_record("app-1"))

    resp = client.get("/v1/applications/app-1", headers=AUTH)
    assert resp.status_code == 200
    assert resp.json()["id"] == "app-1"


def test_get_application_not_found_returns_404():
    client, repo, _ = make_client()
    repo.get_by_id = AsyncMock(return_value=None)

    resp = client.get("/v1/applications/unknown", headers=AUTH)
    assert resp.status_code == 404
    assert resp.json()["error"] == "application_not_found"


# ── PATCH /v1/applications/{id} ───────────────────────────────────────────────

def test_update_application_returns_200():
    client, repo, _ = make_client()
    updated = make_app_record()
    updated.name = "Updated Name"
    repo.update = AsyncMock(return_value=updated)

    resp = client.patch("/v1/applications/app-1",
                        json={"name": "Updated Name"}, headers=AUTH)
    assert resp.status_code == 200


def test_update_application_not_found_returns_404():
    client, repo, _ = make_client()
    repo.update = AsyncMock(return_value=None)

    resp = client.patch("/v1/applications/unknown",
                        json={"name": "X"}, headers=AUTH)
    assert resp.status_code == 404


# ── DELETE /v1/applications/{id} ──────────────────────────────────────────────

def test_delete_application_returns_204():
    client, repo, _ = make_client()
    repo.delete = AsyncMock(return_value=True)

    resp = client.delete("/v1/applications/app-1", headers=AUTH)
    assert resp.status_code == 204


def test_delete_application_not_found_returns_404():
    client, repo, _ = make_client()
    repo.delete = AsyncMock(return_value=False)

    resp = client.delete("/v1/applications/unknown", headers=AUTH)
    assert resp.status_code == 404


# ── POST /v1/applications/{id}/keys ──────────────────────────────────────────

def test_generate_key_returns_201_with_key():
    client, app_repo, key_repo = make_client()
    app_repo.get_by_id = AsyncMock(return_value=make_app_record())
    key_repo.create    = AsyncMock(return_value=MagicMock())

    resp = client.post("/v1/applications/app-1/keys", headers=AUTH)
    assert resp.status_code == 201
    body = resp.json()
    assert body["key"].startswith("rag_")
    assert "hint" in body
    assert body["application_id"] == "app-1"


def test_generate_key_app_not_found_returns_404():
    client, app_repo, _ = make_client()
    app_repo.get_by_id = AsyncMock(return_value=None)

    resp = client.post("/v1/applications/unknown/keys", headers=AUTH)
    assert resp.status_code == 404


# ── POST /v1/applications/{id}/keys/{hint}/rotate ────────────────────────────

def test_rotate_key_returns_200_with_new_key():
    client, app_repo, key_repo = make_client()
    app_repo.get_by_id = AsyncMock(return_value=make_app_record())
    key_repo.revoke    = AsyncMock(return_value=True)
    key_repo.create    = AsyncMock(return_value=MagicMock())

    resp = client.post("/v1/applications/app-1/keys/rag_oldhint/rotate", headers=AUTH)
    assert resp.status_code == 200
    assert resp.json()["key"].startswith("rag_")


def test_rotate_key_not_found_returns_404():
    client, app_repo, key_repo = make_client()
    app_repo.get_by_id = AsyncMock(return_value=make_app_record())
    key_repo.revoke    = AsyncMock(return_value=False)

    resp = client.post("/v1/applications/app-1/keys/rag_unknown/rotate", headers=AUTH)
    assert resp.status_code == 404


# ── DELETE /v1/applications/{id}/keys/{hint} ─────────────────────────────────

def test_revoke_key_returns_204():
    client, _, key_repo = make_client()
    key_repo.revoke = AsyncMock(return_value=True)

    resp = client.delete("/v1/applications/app-1/keys/rag_hint123", headers=AUTH)
    assert resp.status_code == 204


def test_revoke_key_not_found_returns_404():
    client, _, key_repo = make_client()
    key_repo.revoke = AsyncMock(return_value=False)

    resp = client.delete("/v1/applications/app-1/keys/rag_unknown", headers=AUTH)
    assert resp.status_code == 404


# ── Formato de erro uniforme ──────────────────────────────────────────────────

def test_all_404_errors_have_required_fields():
    client, repo, _ = make_client()
    repo.get_by_id = AsyncMock(return_value=None)

    resp = client.get("/v1/applications/nonexistent", headers=AUTH)
    body = resp.json()
    assert "error"    in body
    assert "message"  in body
    assert "trace_id" in body