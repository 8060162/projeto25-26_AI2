from datetime import datetime, timezone, timedelta

import structlog
from fastapi import APIRouter, Request, Query

from rag_api.auth.middleware import require_scope
from rag_api.schemas.responses import GlobalMetricsSummary, ClientMetricsResponse
from rag_api.signals.reader import MetricsReader

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/v1/metrics", tags=["metrics"])


def _default_window() -> tuple[datetime, datetime]:
    now = datetime.now(timezone.utc)
    return now - timedelta(hours=24), now


@router.get("/signals", response_model=GlobalMetricsSummary)
async def get_global_metrics(
    request: Request,
    from_dt: datetime | None = Query(default=None),
    to_dt:   datetime | None = Query(default=None),
    _identity=require_scope("rag:admin"),
) -> GlobalMetricsSummary:
    default_from, default_to = _default_window()
    from_dt = from_dt or default_from
    to_dt   = to_dt   or default_to

    reader: MetricsReader = request.app.state.metrics_reader
    raw = await reader.global_summary(from_dt, to_dt)

    return GlobalMetricsSummary(
        from_dt=from_dt,
        to_dt=to_dt,
        total_queries=raw.get("total_queries", 0),
        avg_score=round(raw.get("avg_score") or 0.0, 4),
        avg_latency_ms=round(raw.get("avg_latency_ms") or 0.0, 2),
        avg_docs_retrieved=round(raw.get("avg_docs") or 0.0, 2),
        total_tokens_input=raw.get("total_tokens_in", 0),
        total_tokens_output=raw.get("total_tokens_out", 0),
    )


@router.get("/signals/{client_id}", response_model=ClientMetricsResponse)
async def get_client_metrics(
    client_id: str,
    request: Request,
    from_dt: datetime | None = Query(default=None),
    to_dt:   datetime | None = Query(default=None),
    _identity=require_scope("rag:admin"),
) -> ClientMetricsResponse:
    default_from, default_to = _default_window()
    from_dt = from_dt or default_from
    to_dt   = to_dt   or default_to

    reader: MetricsReader = request.app.state.metrics_reader
    raw = await reader.client_summary(client_id, from_dt, to_dt)

    categories = [
        {
            "category":          cat.get("_id") or "uncategorized",
            "total_queries":     cat.get("total_queries", 0),
            "avg_score":         round(cat.get("avg_score") or 0.0, 4),
            "avg_latency_ms":    round(cat.get("avg_latency_ms") or 0.0, 2),
            "positive_feedback": cat.get("positive_feedback", 0),
            "negative_feedback": cat.get("negative_feedback", 0),
        }
        for cat in raw.get("categories", [])
    ]

    return ClientMetricsResponse(
        client_id=client_id,
        from_dt=from_dt,
        to_dt=to_dt,
        categories=categories,
    )