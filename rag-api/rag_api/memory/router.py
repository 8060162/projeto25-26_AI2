"""
Router — Módulo 4 — Memória Longa.

Endpoints:
  POST   /v1/memory                       → criar entrada (autenticado ou anónimo)
  GET    /v1/memory/{user_key}            → listar memórias activas
  PATCH  /v1/memory/{user_key}/{mem_id}   → actualizar entrada
  DELETE /v1/memory/{user_key}/{mem_id}   → eliminar entrada

Segurança — duas camadas, padrão do projecto (ver query.py):
  1. require_scope("rag:query") — só aplicações activas com o scope correcto acedem.
     Autenticação + verificação de scope + audit log encapsulados na dependência.
  2. rate_limit() — protege contra abuso por aplicação.

Ownership — terceira camada:
  client_id da identidade deve coincidir com client_id do documento.
  Verificado antes de qualquer operação de leitura ou escrita.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from rag_api.api.middleware.rate_limit import rate_limit
from rag_api.auth.middleware import require_scope
from rag_api.schemas.identity import ClientIdentity

from .keys import derive_user_key
from .schemas import (
    InteractionResponse,
    MemoryCreateRequest,
    MemoryListResponse,
    MemoryResponse,
    MemoryUpdateRequest,
)
from .store import MemoryRecord, MemoryRepository


router = APIRouter(prefix="/v1/memory", tags=["memory"])

_MODE_AUTHENTICATED = "authenticated"
_MODE_ANONYMOUS     = "anonymous"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _repo(request: Request) -> MemoryRepository:
    return request.app.state.memory_repo

def _pepper(request: Request) -> str:
    return request.app.state.memory_hmac_pepper

def _ttl_days(request: Request) -> int:
    return request.app.state.memory_ttl_days

def _to_response(rec: MemoryRecord) -> MemoryResponse:
    return MemoryResponse(
        mem_id       = rec.mem_id,
        session_id   = rec.session_id,
        client_id    = rec.client_id,
        mode         = rec.mode,
        type         = rec.type,
        topic        = rec.topic,
        importance   = rec.importance,
        interaction  = InteractionResponse(**rec.interaction),
        last_seen_at = rec.last_seen_at,
        created_at   = rec.created_at,
        expires_at   = rec.expires_at,
    )

def _assert_ownership(rec: MemoryRecord, identity: ClientIdentity) -> None:
    """Falha rápida se a aplicação autenticada não detém o documento."""
    if rec.client_id != identity.client_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="forbidden")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("", response_model=MemoryResponse, status_code=status.HTTP_201_CREATED)
async def create_memory(
    body:     MemoryCreateRequest,
    request:  Request,
    identity: ClientIdentity = Depends(require_scope("rag:query")),
    _:        ClientIdentity = Depends(rate_limit()),
) -> MemoryResponse:
    """
    Cria uma entrada de memória longa.
    Canal 1 — user_token presente → modo autenticado (contexto pessoal do aluno).
    Canal 2 — user_token ausente  → modo anónimo (métricas e análise da aplicação).
    """
    mode       = _MODE_AUTHENTICATED if body.user_token else _MODE_ANONYMOUS
    user_key   = derive_user_key(_pepper(request), identity.client_id, body.user_token)
    expires_at = datetime.now(timezone.utc) + timedelta(days=_ttl_days(request))

    record = await _repo(request).create(
        user_key   = user_key,
        client_id  = identity.client_id,
        mode       = mode,
        req        = body,
        expires_at = expires_at,
    )
    return _to_response(record)


@router.get("/{user_key}", response_model=MemoryListResponse)
async def list_memories(
    user_key:       str,
    request:        Request,
    identity:       ClientIdentity = Depends(require_scope("rag:query")),
    _:              ClientIdentity = Depends(rate_limit()),
    type:           str | None     = Query(default=None, description="Filtrar por type"),
    min_importance: float | None   = Query(default=None, ge=0.0, le=10.0),
    limit:          int            = Query(default=20, ge=1, le=100),
    skip:           int            = Query(default=0, ge=0),
) -> MemoryListResponse:
    records, total = await _repo(request).list_active(
        user_key       = user_key,
        type_filter    = type,
        min_importance = min_importance,
        limit          = limit,
        skip           = skip,
    )
    # Sem registos não há ownership a verificar — lista vazia é resposta válida.
    # Com registos, o primeiro é suficiente: todos partilham client_id por
    # construção (user_key é derivado de client_id + user_token).
    if records:
        _assert_ownership(records[0], identity)

    return MemoryListResponse(
        items=[_to_response(r) for r in records],
        total=total,
    )


@router.patch("/{user_key}/{mem_id}", response_model=MemoryResponse)
async def update_memory(
    user_key: str,
    mem_id:   str,
    body:     MemoryUpdateRequest,
    request:  Request,
    identity: ClientIdentity = Depends(require_scope("rag:query")),
    _:        ClientIdentity = Depends(rate_limit()),
) -> MemoryResponse:
    # get antes de update para verificar ownership sem depender do resultado do update
    record = await _repo(request).get(user_key, mem_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="memory_not_found")
    _assert_ownership(record, identity)

    updated = await _repo(request).update(user_key, mem_id, body)
    if updated is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="memory_not_found")
    return _to_response(updated)


@router.delete("/{user_key}/{mem_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_memory(
    user_key: str,
    mem_id:   str,
    request:  Request,
    identity: ClientIdentity = Depends(require_scope("rag:query")),
    _:        ClientIdentity = Depends(rate_limit()),
) -> None:
    record = await _repo(request).get(user_key, mem_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="memory_not_found")
    _assert_ownership(record, identity)
    await _repo(request).delete(user_key, mem_id)