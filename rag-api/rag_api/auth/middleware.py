from fastapi import Depends, HTTPException, Request, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from rag_api.auth.audit import AuditEvent, record
from rag_api.auth.strategies import APIKeyAuthStrategy, AuthStrategyProtocol, InvalidKeyError, InsufficientScopeError
from rag_api.schemas.identity import ClientIdentity

_bearer_scheme = HTTPBearer(auto_error=False)


# ── Dependência para injectar a strategy do app.state ────────────────────────

def get_auth_strategy(request: Request) -> AuthStrategyProtocol:
    """Lê a strategy registada no app.state — injectável e testável."""
    return request.app.state.auth_strategy


# ── Dependência base — autentica e injeta ClientIdentity ─────────────────────

async def get_current_client(
    request:  Request,
    _:        HTTPAuthorizationCredentials | None = Security(_bearer_scheme),
    strategy: AuthStrategyProtocol = Depends(get_auth_strategy),
) -> ClientIdentity:
    """
    FastAPI Depends — injectada em qualquer route que precise de autenticação.
    Não levanta HTTPException directamente: converte erros de domínio em HTTP.
    """
    trace_id = getattr(request.state, "trace_id", "unknown")

    try:
        identity = await strategy.authenticate(request)
    except InvalidKeyError as exc:
        raise HTTPException(status_code=401, detail={
            "error":    str(exc),
            "message":  "Authentication failed.",
            "trace_id": trace_id,
        })

    # Injeta no request.state para o access log poder usar
    request.state.client_id = identity.client_id
    return identity


# ── Factory de dependências com scope — para proteger endpoints específicos ───

def require_scope(scope: str):
    """
    Uso: Depends(require_scope("rag:query"))
    Verificação de scope após autenticação — separação de concerns.
    """
    async def _check(
        request:  Request,
        identity: ClientIdentity = Depends(get_current_client),
    ) -> ClientIdentity:
        if scope not in identity.scopes:
            record(
                AuditEvent.SCOPE_VIOLATION,
                identity.client_id,
                getattr(request.state, "trace_id", "unknown"),
                {"required_scope": scope, "client_scopes": identity.scopes},
            )
            raise HTTPException(status_code=403, detail={
                "error":    "insufficient_scope",
                "message":  f"Scope '{scope}' required.",
                "trace_id": getattr(request.state, "trace_id", "unknown"),
            })
        return identity

    return _check