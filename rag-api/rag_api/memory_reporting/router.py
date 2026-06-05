"""
Router — Memory Reporting.

Segue o padrão de metrics.py:
  - require_scope("rag:admin") + rate_limit() em todos os endpoints
  - compute on-demand quando snapshot não existe

Dois endpoints:
  GET /v1/memory-reports/daily/{period}   → snapshot de um dia (com on-demand fallback)
  GET /v1/memory-reports/range            → agregação de intervalo arbitrário
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel

from rag_api.api.middleware.rate_limit import rate_limit
from rag_api.auth.middleware import require_scope
from rag_api.memory_reporting.aggregator import _daily_window
from rag_api.memory_reporting.models import memory_period_key
from rag_api.schemas.identity import ClientIdentity

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/v1/memory-reports", tags=["memory-reports"])


# ── Response schemas ──────────────────────────────────────────────────────────

class TopicSummary(BaseModel):
    topic:              str
    count:              int
    avg_importance:     float
    clarification_rate: float = 0.0


class SourceSummary(BaseModel):
    source: str
    count:  int


class DailyBreakdown(BaseModel):
    period:              str
    total_interactions:  int
    authenticated_count: int
    anonymous_count:     int
    avg_importance:      float


class MemoryDailyReport(BaseModel):
    period:              str
    from_dt:             datetime
    to_dt:               datetime
    total_interactions:  int
    authenticated_count: int
    anonymous_count:     int
    unique_users:        int
    unique_clients:      int
    avg_importance:      float
    low_importance_rate: float
    no_sources_rate:     float
    top_topics:          list[TopicSummary]
    weak_topics:         list[TopicSummary]
    top_sources:         list[SourceSummary]


class MemoryRangeReport(BaseModel):
    from_dt:             datetime
    to_dt:               datetime
    days_covered:        int
    total_interactions:  int
    authenticated_count: int
    anonymous_count:     int
    unique_users:        int
    unique_clients:      int
    avg_importance:      float
    low_importance_rate: float
    no_sources_rate:     float
    top_topics:          list[TopicSummary]
    weak_topics:         list[TopicSummary]
    top_sources:         list[SourceSummary]
    daily_breakdown:     list[DailyBreakdown]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/daily/{period}", response_model=MemoryDailyReport)
async def get_daily_report(
    period:     str,
    request:    Request,
    _identity:  ClientIdentity = Depends(require_scope("rag:admin")),
    _ratelimit: ClientIdentity = Depends(rate_limit()),
) -> MemoryDailyReport:
    """
    Snapshot de um dia específico — formato YYYY-MM-DD.
    Se o snapshot ainda não existir, computa on-demand.
    """
    reader     = request.app.state.memory_report_reader
    aggregator = request.app.state.memory_aggregator

    snapshot = await reader.get_daily(period)
    if snapshot is None:
        # On-demand — mesmo padrão do metrics.py
        try:
            today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            is_today  = (period == today_str)
            dt        = datetime.strptime(period, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            window    = _daily_window(at=dt, today=is_today) if is_today else _daily_window(at=dt + timedelta(days=1))
            snapshot  = await aggregator.compute_and_store(window)
        except ValueError:
            raise HTTPException(
                status_code = status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail      = "Formato de período inválido — usar YYYY-MM-DD",
            )

    if snapshot is None:
        raise HTTPException(
            status_code = status.HTTP_404_NOT_FOUND,
            detail      = f"Sem dados para o período '{period}'",
        )

    return MemoryDailyReport(**snapshot)


@router.post("/refresh", status_code=200)
async def refresh_daily_report(
    request:    Request,
    period:     str | None     = Query(default=None, description="YYYY-MM-DD — omitir para hoje"),
    _identity:  ClientIdentity = Depends(require_scope("rag:admin")),
    _ratelimit: ClientIdentity = Depends(rate_limit()),
) -> MemoryDailyReport:
    """
    Força recomputação imediata do snapshot diário — mesmo padrão do /metrics/refresh.
    """
    aggregator = request.app.state.memory_aggregator

    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if period:
        try:
            # Se o período é hoje, usa janela actual (dados parciais do dia)
            # Se é um dia passado, usa janela completa desse dia
            is_today = (period == today_str)
            dt       = datetime.strptime(period, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            window   = _daily_window(at=dt, today=is_today) if is_today else _daily_window(at=dt + timedelta(days=1))
        except ValueError:
            raise HTTPException(
                status_code = status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail      = "Formato de período inválido — usar YYYY-MM-DD",
            )
    else:
        # Sem período — Refresh manual usa sempre o dia actual
        window = _daily_window(today=True)

    snapshot = await aggregator.compute_and_store(window)
    logger.info("memory_report.manual_refresh", period=window.period)
    return MemoryDailyReport(**snapshot)


@router.get("/range", response_model=MemoryRangeReport)
async def get_range_report(
    request:    Request,
    _identity:  ClientIdentity = Depends(require_scope("rag:admin")),
    _ratelimit: ClientIdentity = Depends(rate_limit()),
    from_dt:    datetime       = Query(..., description="Início do intervalo (UTC)"),
    to_dt:      datetime       = Query(..., description="Fim do intervalo (UTC)"),
) -> MemoryRangeReport:
    """
    Agrega snapshots diários no intervalo pedido.
    Permite consultar qualquer período — semana, mês, semestre, ano académico.
    """
    if from_dt >= to_dt:
        raise HTTPException(
            status_code = status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail      = "from_dt deve ser anterior a to_dt",
        )

    reader = request.app.state.memory_report_reader
    data   = await reader.get_aggregated(from_dt, to_dt)
    return MemoryRangeReport(**data)