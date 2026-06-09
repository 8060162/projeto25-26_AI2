import time
import uuid

import structlog
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

logger = structlog.get_logger(__name__)


class AccessLogMiddleware(BaseHTTPMiddleware):
    """
    Regista metadados de cada pedido HTTP.
    Nunca regista: corpo do pedido, corpo da resposta, authorization header.
    trace_id gerado aqui — propaga para todos os componentes via request.state.
    """

    async def dispatch(self, request: Request, call_next):
        trace_id = str(uuid.uuid4())
        request.state.trace_id = trace_id

        start = time.monotonic()
        response = await call_next(request)
        latency_ms = round((time.monotonic() - start) * 1000, 2)

        # client_id injectado pelo auth middleware — pode não existir em /health
        client_id = getattr(request.state, "client_id", "anonymous")

        logger.info(
            "api_access",
            trace_id   = trace_id,
            client_id  = client_id,
            method     = request.method,
            path       = request.url.path,
            status     = response.status_code,
            latency_ms = latency_ms,
        )

        response.headers["X-Trace-Id"] = trace_id
        return response