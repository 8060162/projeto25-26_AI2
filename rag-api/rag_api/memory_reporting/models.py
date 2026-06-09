"""
Models — Memory Reporting.

Granularidade simplificada face ao reporting de signals:
  - Apenas DAILY como unidade base de computação
  - A agregação por intervalos arbitrários é feita no reader em runtime
    — sem necessidade de snapshots hourly ou realtime para métricas de memória

period_key reutiliza a mesma convenção do reporting existente
(Europe/Lisbon, formato determinístico) para consistência entre módulos.
"""
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from zoneinfo import ZoneInfo

LISBON = ZoneInfo("Europe/Lisbon")


class MemoryGranularity(StrEnum):
    DAILY = "daily"


def memory_period_key(at: datetime | None = None) -> str:
    """Chave diária determinística — base do upsert idempotente."""
    now = (at or datetime.now(timezone.utc)).astimezone(LISBON)
    return now.strftime("%Y-%m-%d")


@dataclass(frozen=True)
class MemorySnapshotWindow:
    from_dt: datetime
    to_dt:   datetime
    period:  str       # chave diária — ex: "2025-10-15"