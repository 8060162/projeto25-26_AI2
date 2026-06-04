from datetime import datetime, timezone, timedelta

import structlog
from fastapi import APIRouter, Request, Query, HTTPException

from rag_api.auth.middleware import require_scope
from rag_api.reporting import SnapshotAggregator, Granularity
from rag_api.reporting.aggregator import _window_for
from rag_api.reporting.models import period_key
from rag_api.schemas.responses import GlobalMetricsSummary, ClientMetricsResponse
from rag_api.signals.reader import MetricsReader

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/v1/metrics", tags=["metrics"])


@router.get("/signals", response_model=GlobalMetricsSummary)
async def get_global_metrics(
    request:     Request,
    granularity: Granularity = Query(default=Granularity.REALTIME),
    period:      str | None  = Query(default=None),
    _identity=require_scope("rag:admin"),
) -> GlobalMetricsSummary:
    """
    Lê snapshot pré-computado pelo scheduler.
    Se o snapshot ainda não existir, computa on-demand.
    """
    aggregator: SnapshotAggregator = request.app.state.snapshot_aggregator
    target_period = period or period_key(granularity)

    snapshot = await aggregator.get_snapshot(granularity, target_period)
    if snapshot is None:
        window   = _window_for(granularity)
        snapshot = await aggregator.compute_and_store(window)

    return _to_response(snapshot)


@router.post("/refresh", status_code=200)
async def refresh_metrics(
    request:     Request,
    granularity: Granularity = Query(default=Granularity.REALTIME),
    _identity=require_scope("rag:admin"),
) -> GlobalMetricsSummary:
    """
    Força recomputação imediata do snapshot — independente do scheduler.
    """
    aggregator: SnapshotAggregator = request.app.state.snapshot_aggregator
    window   = _window_for(granularity)
    snapshot = await aggregator.compute_and_store(window)
    logger.info("metrics.manual_refresh", granularity=granularity)
    return _to_response(snapshot)


@router.get("/signals/{client_id}", response_model=ClientMetricsResponse)
async def get_client_metrics(
    client_id: str,
    request:   Request,
    from_dt:   datetime | None = Query(default=None),
    to_dt:     datetime | None = Query(default=None),
    _identity=require_scope("rag:admin"),
) -> ClientMetricsResponse:
    """
    Dados históricos por cliente — query directa ao query_signals.
    """
    now    = datetime.now(timezone.utc)
    from_dt = from_dt or now - timedelta(hours=24)
    to_dt   = to_dt   or now

    if from_dt >= to_dt:
        raise HTTPException(status_code=422, detail="from_dt deve ser anterior a to_dt.")

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


def _to_response(snapshot: dict) -> GlobalMetricsSummary:
    """Mapeia snapshot MongoDB para o contrato da API."""
    return GlobalMetricsSummary(
        from_dt=snapshot.get("from_dt", datetime.now(timezone.utc)),
        to_dt=snapshot.get("to_dt", datetime.now(timezone.utc)),
        generated_at=snapshot.get("generated_at"),
        total_queries=snapshot.get("total_queries", 0),
        success_rate=snapshot.get("success_rate"),
        avg_score=snapshot.get("avg_retrieval_score", 0.0),
        avg_latency_ms=snapshot.get("avg_latency_ms", 0.0),
        avg_docs_retrieved=0.0,
        total_tokens_input=snapshot.get("total_tokens_input", 0),
        total_tokens_output=snapshot.get("total_tokens_output", 0),
        low_score_rate=snapshot.get("low_score_rate"),
        active_clients=snapshot.get("active_clients"),
        top_categories=snapshot.get("top_categories", []),
        peak_hour_lisbon=snapshot.get("peak_hour_lisbon"),
        queries_by_day=snapshot.get("queries_by_day", {}),
    )