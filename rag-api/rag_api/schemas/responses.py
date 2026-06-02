from pydantic import BaseModel


class QueryResponse(BaseModel):
    answer:     str
    sources:    list[str]   # referências dos documentos usados
    trace_id:   str
    session_id: str         # criado ou confirmado — para memória curta


class FeedbackResponse(BaseModel):
    accepted:  bool
    trace_id:  str


class HealthResponse(BaseModel):
    status:  str            # "ok"
    version: str


# Formato de erro uniforme em toda a API — acordado no módulo 2
class ErrorResponse(BaseModel):
    error:    str           # código máquina
    message:  str           # descrição legível
    trace_id: str