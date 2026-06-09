"""
Testes do Módulo 1 — API

Cobertura:
- POST /v1/query      : happy path, sem auth, scope errado, input inválido
- POST /v1/feedback   : happy path, rating inválido
- GET  /v1/health     : sem auth, sempre 200
- Error handlers      : formato ErrorResponse uniforme
- Access log          : trace_id no header de resposta
"""
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from rag_api.api.main import create_app
from rag_api.auth.strategies import APIKeyAuthStrategy, InvalidKeyError
from rag_api.rag import StubRAGController
from rag_api.schemas.identity import ClientIdentity
from rag_api.schemas.responses import QueryResponse


# ── Setup ─────────────────────────────────────────────────────────────────────

VALID_IDENTITY = ClientIdentity(
    client_id  = "test-client",
    scopes     = ["rag:query"],
    rate_limit = 100,
)


def make_app(identity: ClientIdentity = VALID_IDENTITY, auth_raises=None):
    """
    Factory de app de teste.
    Injeta mocks no app.state — sem DB, sem Redis, sem LLM.
    """
    app = create_app()

    # Mock da strategy de autenticação
    mock_strategy = AsyncMock(spec=APIKeyAuthStrategy)
    if auth_raises:
        mock_strategy.authenticate.side_effect = auth_raises
    else:
        mock_strategy.authenticate.return_value = identity

    # Mock do rate limiter — passa sempre
    mock_limiter = MagicMock()
    mock_limiter.check = AsyncMock()

    app.state.auth_strategy  = mock_strategy
    app.state.rate_limiter   = mock_limiter
    app.state.rag_controller = StubRAGController()

    return app


@pytest.fixture
def client():
    return TestClient(make_app(), raise_server_exceptions=False)


AUTH_HEADER = {"Authorization": "Bearer rag_testtoken"}


# ── GET /v1/health ─────────────────────────────────────────────────────────────

def test_health_returns_200_without_auth():
    app    = make_app()
    client = TestClient(app)
    resp   = client.get("/v1/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_health_has_trace_id_header(client):
    resp = client.get("/v1/health")
    assert "x-trace-id" in resp.headers


# ── POST /v1/query — happy path ────────────────────────────────────────────────

def test_query_returns_200_with_valid_request(client):
    resp = client.post(
        "/v1/query",
        json    = {"question": "Qual o prazo de matrícula?"},
        headers = AUTH_HEADER,
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "answer"     in body
    assert "sources"    in body
    assert "trace_id"   in body
    assert "session_id" in body


def test_query_response_trace_id_matches_header(client):
    resp = client.post(
        "/v1/query",
        json    = {"question": "Qual o prazo?"},
        headers = AUTH_HEADER,
    )
    assert resp.json()["trace_id"] == resp.headers["x-trace-id"]


def test_query_accepts_optional_session_id(client):
    session = str(uuid.uuid4())
    resp = client.post(
        "/v1/query",
        json    = {"question": "Qual o prazo?", "session_id": session},
        headers = AUTH_HEADER,
    )
    assert resp.status_code == 200


# ── POST /v1/query — autenticação ─────────────────────────────────────────────

def test_query_without_auth_returns_401():
    app    = make_app(auth_raises=InvalidKeyError("missing_bearer_token"))
    client = TestClient(app, raise_server_exceptions=False)
    resp   = client.post("/v1/query", json={"question": "teste"})
    assert resp.status_code == 401
    assert resp.json()["error"] == "missing_bearer_token"


def test_query_with_invalid_key_returns_401():
    app    = make_app(auth_raises=InvalidKeyError("invalid_key"))
    client = TestClient(app, raise_server_exceptions=False)
    resp   = client.post(
        "/v1/query",
        json    = {"question": "teste"},
        headers = {"Authorization": "Bearer rag_invalid"},
    )
    assert resp.status_code == 401
    assert resp.json()["error"] == "invalid_key"


def test_query_with_insufficient_scope_returns_403():
    identity_no_scope = ClientIdentity(
        client_id  = "limited-client",
        scopes     = [],            # sem rag:query
        rate_limit = 100,
    )
    app    = make_app(identity=identity_no_scope)
    client = TestClient(app, raise_server_exceptions=False)
    resp   = client.post(
        "/v1/query",
        json    = {"question": "teste"},
        headers = AUTH_HEADER,
    )
    assert resp.status_code == 403
    assert resp.json()["error"] == "insufficient_scope"


# ── POST /v1/query — validação de input ───────────────────────────────────────

def test_query_with_empty_question_returns_422(client):
    resp = client.post(
        "/v1/query",
        json    = {"question": ""},
        headers = AUTH_HEADER,
    )
    assert resp.status_code == 422
    assert resp.json()["error"] == "validation_error"


def test_query_with_missing_question_returns_422(client):
    resp = client.post(
        "/v1/query",
        json    = {},
        headers = AUTH_HEADER,
    )
    assert resp.status_code == 422


def test_query_with_question_too_long_returns_422(client):
    resp = client.post(
        "/v1/query",
        json    = {"question": "x" * 1001},
        headers = AUTH_HEADER,
    )
    assert resp.status_code == 422


# ── POST /v1/feedback ─────────────────────────────────────────────────────────

def test_feedback_positive_returns_200(client):
    resp = client.post(
        "/v1/feedback",
        json    = {"trace_id": "trace-abc", "rating": "positive"},
        headers = AUTH_HEADER,
    )
    assert resp.status_code == 200
    assert resp.json()["accepted"] is True


def test_feedback_negative_with_reason_returns_200(client):
    resp = client.post(
        "/v1/feedback",
        json    = {"trace_id": "trace-abc", "rating": "negative", "reason": "Resposta incorrecta"},
        headers = AUTH_HEADER,
    )
    assert resp.status_code == 200


def test_feedback_invalid_rating_returns_422(client):
    resp = client.post(
        "/v1/feedback",
        json    = {"trace_id": "trace-abc", "rating": "meh"},
        headers = AUTH_HEADER,
    )
    assert resp.status_code == 422
    assert resp.json()["error"] == "validation_error"


# ── Error format ──────────────────────────────────────────────────────────────

def test_all_error_responses_have_required_fields(client):
    """Garante que o contrato ErrorResponse é respeitado em todos os erros."""
    error_cases = [
        dict(url="/v1/query", json={"question": ""},       headers=AUTH_HEADER),
        dict(url="/v1/query", json={"question": "teste"},  headers={}),
    ]
    for case in error_cases:
        resp = client.post(**case)
        assert resp.status_code >= 400
        body = resp.json()
        assert "error"    in body, f"Missing 'error' in {body}"
        assert "message"  in body, f"Missing 'message' in {body}"
        assert "trace_id" in body, f"Missing 'trace_id' in {body}"