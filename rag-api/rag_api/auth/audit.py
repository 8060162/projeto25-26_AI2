from datetime import datetime, timezone
from enum import StrEnum

import structlog

logger = structlog.get_logger(__name__)


class AuditEvent(StrEnum):
    KEY_CREATED       = "key_created"
    KEY_REVOKED       = "key_revoked"
    AUTH_FAILED       = "auth_failed"
    SCOPE_VIOLATION   = "scope_violation"
    RATE_LIMIT_HIT    = "rate_limit_hit"


def record(
    event: AuditEvent,
    client_id: str | None,
    trace_id: str,
    detail: dict | None = None,
) -> None:
    """
    Regista um evento de segurança.
    Usa structlog — o destino (stdout, ficheiro, SIEM) é configurado externamente.
    Append-only por design: nunca edita ou apaga registos de audit.
    """
    logger.warning(
        event,
        audit=True,             # campo para filtrar audit events nos logs
        client_id=client_id,
        trace_id=trace_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        **(detail or {}),
    )