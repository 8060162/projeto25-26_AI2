from dataclasses import dataclass, field
from pydantic import BaseModel
from datetime import datetime


# ── Contrato que o módulo auth entrega ao módulo API ──────────────────────────

@dataclass(frozen=True)
class ClientIdentity:
    """
    Injectado pelo auth middleware em cada pedido autenticado.
    Imutável por design — nada downstream pode alterar a identidade.
    """
    client_id:  str
    scopes:     list[str]
    rate_limit: int         # pedidos por minuto


# ── Erros estruturados — sem expor internos ───────────────────────────────────

class AuthErrorResponse(BaseModel):
    error:    str       # código máquina: invalid_key, insufficient_scope, ...
    message:  str       # descrição legível
    trace_id: str


# ── Gestão de API Keys ────────────────────────────────────────────────────────

class APIKeyRecord(BaseModel):
    """Representa uma key na base de dados — nunca contém a key em claro."""
    id:         str
    client_id:  str
    key_hash:   str
    key_hint:   str         # primeiros 12 chars — identificação visual
    scopes:     list[str]
    rate_limit: int
    expires_at: datetime | None
    active:     bool


class CreateKeyRequest(BaseModel):
    client_id:  str
    scopes:     list[str]
    rate_limit: int = 100
    expires_at: datetime | None = None


class CreateKeyResponse(BaseModel):
    """
    Devolvida uma única vez após criação.
    A key em claro nunca mais é acessível após este momento.
    """
    key:        str         # rag_xxxx... — mostrar ao cliente uma vez
    hint:       str         # primeiros 12 chars para identificação futura
    client_id:  str
    scopes:     list[str]
    expires_at: datetime | None