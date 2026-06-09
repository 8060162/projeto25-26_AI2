from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from zoneinfo import ZoneInfo


class Granularity(StrEnum):
    REALTIME = "realtime"
    HOURLY   = "hourly"
    DAILY    = "daily"
    MONTHLY  = "monthly"


def period_key(granularity: Granularity, at: datetime | None = None) -> str:
    """
    Chave determinística por granularidade — garante upsert idempotente.
    Calculada em Europe/Lisbon para corresponder ao ciclo académico do IPP.
    """
    now = (at or datetime.now(timezone.utc)).astimezone(ZoneInfo("Europe/Lisbon"))
    match granularity:
        case Granularity.REALTIME: return "realtime"
        case Granularity.HOURLY:   return now.strftime("%Y-%m-%dT%H")
        case Granularity.DAILY:    return now.strftime("%Y-%m-%d")
        case Granularity.MONTHLY:  return now.strftime("%Y-%m")


@dataclass(frozen=True)
class SnapshotWindow:
    granularity: Granularity
    from_dt:     datetime
    to_dt:       datetime
    period:      str