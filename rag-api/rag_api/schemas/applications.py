from datetime import datetime
from pydantic import BaseModel, Field


# ── Application — entidade de primeiro nível ──────────────────────────────────

class ApplicationRecord(BaseModel):
    """
    Representa uma aplicação cliente registada no sistema.
    Uma Application pode ter múltiplas API Keys activas (para rotação sem downtime).
    """
    id:          str
    name:        str
    description: str | None
    scopes:      list[str]
    rate_limit:  int
    active:      bool
    created_at:  datetime
    created_by:  str            # client_id do admin que a criou


class CreateApplicationRequest(BaseModel):
    name:        str  = Field(..., min_length=1, max_length=100)
    description: str | None = Field(default=None, max_length=500)
    scopes:      list[str] = Field(default=["rag:query"])
    rate_limit:  int       = Field(default=100, ge=1, le=10000)


class UpdateApplicationRequest(BaseModel):
    name:        str | None = Field(default=None, min_length=1, max_length=100)
    description: str | None = Field(default=None, max_length=500)
    scopes:      list[str] | None = None
    rate_limit:  int | None       = Field(default=None, ge=1, le=10000)


class ApplicationResponse(BaseModel):
    id:          str
    name:        str
    description: str | None
    scopes:      list[str]
    rate_limit:  int
    active:      bool
    created_at:  datetime


class ApplicationListResponse(BaseModel):
    items: list[ApplicationResponse]
    total: int


# ── API Key responses no contexto de Applications ────────────────────────────

class GenerateKeyResponse(BaseModel):
    """
    Devolvida uma única vez — a key em claro nunca mais é acessível.
    """
    key:            str
    hint:           str
    application_id: str
    scopes:         list[str]
    expires_at:     datetime | None


class KeyInfoResponse(BaseModel):
    """
    Informação sobre uma key sem expor o valor — para listagem.
    """
    hint:       str
    active:     bool
    created_at: datetime
    expires_at: datetime | None