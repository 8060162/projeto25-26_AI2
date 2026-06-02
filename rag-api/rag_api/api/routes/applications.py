import structlog
from fastapi import APIRouter, Depends, HTTPException, Request

from rag_api.applications.store import ApplicationRepositoryProtocol
from rag_api.auth.keys import generate_api_key, hash_key, build_create_key_response
from rag_api.auth.middleware import require_scope
from rag_api.auth.store import KeyRepositoryProtocol
from rag_api.auth.audit import AuditEvent, record as audit_record
from rag_api.schemas.applications import (
    ApplicationListResponse,
    ApplicationResponse,
    CreateApplicationRequest,
    GenerateKeyResponse,
    KeyInfoResponse,
    UpdateApplicationRequest,
)
from rag_api.schemas.identity import ClientIdentity, CreateKeyRequest

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/v1/applications", tags=["applications"])

# Todos os endpoints deste router requerem rag:admin
_admin = Depends(require_scope("rag:admin"))


# ── CRUD de Applications ──────────────────────────────────────────────────────

@router.post("", response_model=ApplicationResponse, status_code=201)
async def create_application(
    body:     CreateApplicationRequest,
    request:  Request,
    identity: ClientIdentity = _admin,
) -> ApplicationResponse:
    """Cria uma nova Application. Apenas Admin."""
    repo: ApplicationRepositoryProtocol = request.app.state.application_repo

    app = await repo.create(body, created_by=identity.client_id)

    audit_record(AuditEvent.KEY_CREATED, identity.client_id,
                 request.state.trace_id, {"application_id": app.id})

    logger.info("application_created", application_id=app.id,
                created_by=identity.client_id, trace_id=request.state.trace_id)

    return ApplicationResponse(**app.model_dump())


@router.get("", response_model=ApplicationListResponse)
async def list_applications(
    request:  Request,
    identity: ClientIdentity = _admin,
) -> ApplicationListResponse:
    """Lista todas as Applications activas. Apenas Admin."""
    repo: ApplicationRepositoryProtocol = request.app.state.application_repo
    apps = await repo.get_all()
    return ApplicationListResponse(
        items=[ApplicationResponse(**a.model_dump()) for a in apps],
        total=len(apps),
    )


@router.get("/{app_id}", response_model=ApplicationResponse)
async def get_application(
    app_id:   str,
    request:  Request,
    identity: ClientIdentity = _admin,
) -> ApplicationResponse:
    """Obtém uma Application por ID. Apenas Admin."""
    repo: ApplicationRepositoryProtocol = request.app.state.application_repo
    app = await repo.get_by_id(app_id)

    if not app:
        raise HTTPException(status_code=404, detail={
            "error":    "application_not_found",
            "message":  f"Application '{app_id}' not found.",
            "trace_id": request.state.trace_id,
        })
    return ApplicationResponse(**app.model_dump())


@router.patch("/{app_id}", response_model=ApplicationResponse)
async def update_application(
    app_id:   str,
    body:     UpdateApplicationRequest,
    request:  Request,
    identity: ClientIdentity = _admin,
) -> ApplicationResponse:
    """Edita nome, descrição, scopes ou rate_limit. Apenas Admin."""
    repo: ApplicationRepositoryProtocol = request.app.state.application_repo
    app = await repo.update(app_id, body)

    if not app:
        raise HTTPException(status_code=404, detail={
            "error":    "application_not_found",
            "message":  f"Application '{app_id}' not found.",
            "trace_id": request.state.trace_id,
        })

    logger.info("application_updated", application_id=app_id,
                updated_by=identity.client_id, trace_id=request.state.trace_id)
    return ApplicationResponse(**app.model_dump())


@router.delete("/{app_id}", status_code=204)
async def delete_application(
    app_id:   str,
    request:  Request,
    identity: ClientIdentity = _admin,
) -> None:
    """Remove uma Application (soft delete). Apenas Admin."""
    repo: ApplicationRepositoryProtocol = request.app.state.application_repo
    deleted = await repo.delete(app_id)

    if not deleted:
        raise HTTPException(status_code=404, detail={
            "error":    "application_not_found",
            "message":  f"Application '{app_id}' not found.",
            "trace_id": request.state.trace_id,
        })

    audit_record(AuditEvent.KEY_REVOKED, identity.client_id,
                 request.state.trace_id, {"application_id": app_id})
    logger.info("application_deleted", application_id=app_id,
                deleted_by=identity.client_id, trace_id=request.state.trace_id)


# ── Ciclo de vida das API Keys ────────────────────────────────────────────────

@router.post("/{app_id}/keys", response_model=GenerateKeyResponse, status_code=201)
async def generate_api_key_endpoint(
    app_id:   str,
    request:  Request,
    identity: ClientIdentity = _admin,
) -> GenerateKeyResponse:
    """
    Gera uma nova API Key para a Application.
    A key em claro é devolvida uma única vez — não é recuperável depois.
    Apenas Admin.
    """
    app_repo: ApplicationRepositoryProtocol = request.app.state.application_repo
    key_repo: KeyRepositoryProtocol         = request.app.state.key_repo

    app = await app_repo.get_by_id(app_id)
    if not app:
        raise HTTPException(status_code=404, detail={
            "error":    "application_not_found",
            "message":  f"Application '{app_id}' not found.",
            "trace_id": request.state.trace_id,
        })

    raw_key, key_hash, key_hint = generate_api_key()

    await key_repo.create(
        request  = CreateKeyRequest(
            client_id  = app_id,
            scopes     = app.scopes,
            rate_limit = app.rate_limit,
        ),
        key_hash = key_hash,
        key_hint = key_hint,
    )

    audit_record(AuditEvent.KEY_CREATED, identity.client_id,
                 request.state.trace_id, {"application_id": app_id, "hint": key_hint})

    return GenerateKeyResponse(
        key            = raw_key,
        hint           = key_hint,
        application_id = app_id,
        scopes         = app.scopes,
        expires_at     = None,
    )


@router.post("/{app_id}/keys/{hint}/rotate", response_model=GenerateKeyResponse)
async def rotate_api_key(
    app_id:   str,
    hint:     str,
    request:  Request,
    identity: ClientIdentity = _admin,
) -> GenerateKeyResponse:
    """
    Rotação de API Key: revoga a key identificada pelo hint e gera uma nova.
    Ambas coexistem durante a transição — a nova é devolvida de imediato.
    Apenas Admin.
    """
    app_repo:   ApplicationRepositoryProtocol = request.app.state.application_repo
    key_repo:   KeyRepositoryProtocol         = request.app.state.key_repo
    auth_cache                                = request.app.state.auth_strategy._cache

    app = await app_repo.get_by_id(app_id)
    if not app:
        raise HTTPException(status_code=404, detail={
            "error":    "application_not_found",
            "message":  f"Application '{app_id}' not found.",
            "trace_id": request.state.trace_id,
        })

    # Revoga a key antiga
    revoked = await key_repo.revoke(hint, client_id=app_id)
    if not revoked:
        raise HTTPException(status_code=404, detail={
            "error":    "key_not_found",
            "message":  f"Key '{hint}' not found or already revoked.",
            "trace_id": request.state.trace_id,
        })

    # Gera a nova key
    raw_key, key_hash, key_hint = generate_api_key()
    await key_repo.create(
        request  = CreateKeyRequest(
            client_id  = app_id,
            scopes     = app.scopes,
            rate_limit = app.rate_limit,
        ),
        key_hash = key_hash,
        key_hint = key_hint,
    )

    audit_record(AuditEvent.KEY_REVOKED, identity.client_id,
                 request.state.trace_id, {"application_id": app_id, "rotated_hint": hint})
    audit_record(AuditEvent.KEY_CREATED, identity.client_id,
                 request.state.trace_id, {"application_id": app_id, "new_hint": key_hint})

    return GenerateKeyResponse(
        key            = raw_key,
        hint           = key_hint,
        application_id = app_id,
        scopes         = app.scopes,
        expires_at     = None,
    )


@router.delete("/{app_id}/keys/{hint}", status_code=204)
async def revoke_api_key(
    app_id:   str,
    hint:     str,
    request:  Request,
    identity: ClientIdentity = _admin,
) -> None:
    """
    Revoga uma API Key específica pelo seu hint.
    A revogação invalida imediatamente o cache Redis.
    Apenas Admin.
    """
    key_repo: KeyRepositoryProtocol = request.app.state.key_repo

    revoked = await key_repo.revoke(hint, client_id=app_id)
    if not revoked:
        raise HTTPException(status_code=404, detail={
            "error":    "key_not_found",
            "message":  f"Key '{hint}' not found or already revoked.",
            "trace_id": request.state.trace_id,
        })

    audit_record(AuditEvent.KEY_REVOKED, identity.client_id,
                 request.state.trace_id, {"application_id": app_id, "hint": hint})
    logger.info("key_revoked", application_id=app_id, hint=hint,
                revoked_by=identity.client_id, trace_id=request.state.trace_id)