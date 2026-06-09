from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question:   str       = Field(..., min_length=1, max_length=1000)
    session_id: str | None = Field(default=None, description="Para memória curta — opcional")
    # Canal autenticado: user_token presente → memória pessoal do aluno
    # Canal anónimo:     user_token ausente  → memória da aplicação para métricas
    user_token: str | None = Field(default=None, min_length=1, max_length=256,
                                   description="Identificador opaco do utilizador — nunca persistido")


class FeedbackRequest(BaseModel):
    trace_id: str  = Field(..., description="trace_id da resposta a avaliar")
    rating:   str  = Field(..., pattern="^(positive|negative)$")
    reason:   str | None = Field(default=None, max_length=500)