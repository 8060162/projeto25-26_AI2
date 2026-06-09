"""
Schemas Pydantic — Módulo 4 — Memória Longa.

Um único MemoryCreateRequest cobre os dois canais:
  user_token presente → modo autenticado (contexto pessoal do aluno)
  user_token ausente  → modo anónimo (métricas e análise da aplicação)

InteractionPayload alinha com QueryResponse existente (answer, sources, trace_id)
para que o query.py consiga construir o payload sem transformações adicionais.
"""
from __future__ import annotations

from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, Field, model_validator


# ── Sub-schemas ───────────────────────────────────────────────────────────────

class InteractionPayload(BaseModel):
    """
    Par pergunta+resposta como unidade atómica.
    Campos alinhados com QueryResponse — sem transformação no caller.
    sources e trace_id opcionais para memórias criadas fora do pipeline.
    """
    question: str       = Field(..., min_length=1, max_length=1000)
    answer:   str       = Field(..., min_length=1, max_length=8000)
    sources:  list[str] = Field(default_factory=list)
    trace_id: str | None = Field(default=None)


# ── Requests ──────────────────────────────────────────────────────────────────

class MemoryCreateRequest(BaseModel):
    """
    Contrato de criação — usado pelo router /v1/memory e pelo query.py.
    user_token ausente → modo anónimo; presente → modo autenticado.
    topic e importance são responsabilidade do caller (LLM ou pipeline).
    """
    user_token:  str | None = Field(default=None, min_length=1, max_length=256)
    session_id:  str        = Field(..., min_length=1, max_length=128)
    type:        str        = Field(..., pattern=r"^[a-z_]{1,64}$")
    topic:       str        = Field(..., min_length=1, max_length=256)
    importance:  Annotated[float, Field(ge=0.0, le=10.0)]
    interaction: InteractionPayload


class MemoryUpdateRequest(BaseModel):
    """
    Apenas campos mutáveis após criação.
    Requer pelo menos um campo — evita writes vazios.
    """
    topic:       str | None                = Field(default=None, min_length=1, max_length=256)
    importance:  Annotated[float | None, Field(default=None, ge=0.0, le=10.0)]
    interaction: InteractionPayload | None = Field(default=None)

    @model_validator(mode="after")
    def at_least_one_field(self) -> "MemoryUpdateRequest":
        if all(v is None for v in (self.topic, self.importance, self.interaction)):
            raise ValueError("pelo menos um campo deve ser fornecido")
        return self


# ── Responses ─────────────────────────────────────────────────────────────────

class InteractionResponse(BaseModel):
    question: str
    answer:   str
    sources:  list[str]
    trace_id: str | None


class MemoryResponse(BaseModel):
    mem_id:       str
    session_id:   str
    client_id:    str
    mode:         str        # "authenticated" | "anonymous"
    type:         str
    topic:        str
    importance:   float
    interaction:  InteractionResponse
    last_seen_at: datetime
    created_at:   datetime
    expires_at:   datetime
    # user_key e user_token nunca aparecem na resposta


class MemoryListResponse(BaseModel):
    items: list[MemoryResponse]
    total: int