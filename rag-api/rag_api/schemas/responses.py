from pydantic import BaseModel
from datetime import datetime


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


# ── Metrics ───────────────────────────────────────────────────────────────────

class CategoryMetrics(BaseModel):
    category:          str
    total_queries:     int
    avg_score:         float
    avg_latency_ms:    float
    positive_feedback: int
    negative_feedback: int


class GlobalMetricsSummary(BaseModel):
    from_dt:             datetime
    to_dt:               datetime
    total_queries:       int
    avg_score:           float
    avg_latency_ms:      float
    avg_docs_retrieved:  float
    total_tokens_input:  int
    total_tokens_output: int


class ClientMetricsResponse(BaseModel):
    client_id:  str
    from_dt:    datetime
    to_dt:      datetime
    categories: list[CategoryMetrics]