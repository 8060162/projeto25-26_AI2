from pydantic import BaseModel
from datetime import datetime


class QueryResponse(BaseModel):
    answer:     str
    sources:    list[str]
    trace_id:   str
    session_id: str


class FeedbackResponse(BaseModel):
    accepted:  bool
    trace_id:  str


class HealthResponse(BaseModel):
    status:  str
    version: str


class ErrorResponse(BaseModel):
    error:    str
    message:  str
    trace_id: str


# ── Metrics ───────────────────────────────────────────────────────────────────

class CategoryMetrics(BaseModel):
    category:          str
    total_queries:     int
    avg_score:         float
    avg_latency_ms:    float
    positive_feedback: int
    negative_feedback: int


class TopCategory(BaseModel):
    category: str
    count:    int


class EvidenceDistribution(BaseModel):
    strong: int
    weak:   int


class GroundingDistribution(BaseModel):
    strong_alignment: int
    mismatch:         int


class GlobalMetricsSummary(BaseModel):
    from_dt:                datetime
    to_dt:                  datetime
    generated_at:           datetime | None = None
    total_queries:          int
    success_rate:           float | None = None
    grounded_rate:          float | None = None
    avg_score:              float
    avg_latency_ms:         float
    avg_docs_retrieved:     float
    total_tokens_input:     int
    total_tokens_output:    int
    low_score_rate:         float | None = None
    active_clients:         int | None = None
    top_categories:         list[TopCategory] = []
    peak_hour_lisbon:       int | None = None
    queries_by_day:         dict[str, int] = {}
    evidence_distribution:  EvidenceDistribution | None = None
    grounding_distribution: GroundingDistribution | None = None


class ClientMetricsResponse(BaseModel):
    client_id:  str
    from_dt:    datetime
    to_dt:      datetime
    categories: list[CategoryMetrics]