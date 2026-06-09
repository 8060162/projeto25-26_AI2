from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException


def _trace_id(request: Request) -> str:
    return getattr(request.state, "trace_id", "unavailable")


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """
    Garante que todos os HTTPException seguem o formato ErrorResponse.
    Os detalhes do módulo 2 (auth, rate limit) já chegam neste formato.
    """
    detail = exc.detail
    if isinstance(detail, dict):
        body = detail
    else:
        body = {
            "error":    "http_error",
            "message":  str(detail),
            "trace_id": _trace_id(request),
        }
    return JSONResponse(status_code=exc.status_code, content=body,
                        headers=getattr(exc, "headers", None))


async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """
    Erros de validação Pydantic → formato ErrorResponse.
    Extrai o primeiro erro para a mensagem — sem expor o stack interno.
    """
    first = exc.errors()[0] if exc.errors() else {}
    field = " → ".join(str(loc) for loc in first.get("loc", []))
    msg   = first.get("msg", "Validation error")

    return JSONResponse(
        status_code=422,
        content={
            "error":    "validation_error",
            "message":  f"{field}: {msg}" if field else msg,
            "trace_id": _trace_id(request),
        },
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Último recurso — nunca expõe internos ao cliente.
    O erro real é logado pelo access log middleware.
    """
    import structlog
    logger = structlog.get_logger(__name__)
    logger.error(
        "unhandled_exception",
        trace_id  = _trace_id(request),
        exception = repr(exc),
    )
    return JSONResponse(
        status_code=500,
        content={
            "error":    "internal_error",
            "message":  "An unexpected error occurred.",
            "trace_id": _trace_id(request),
        },
    )