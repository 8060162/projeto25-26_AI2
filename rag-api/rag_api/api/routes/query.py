import time
import structlog
from fastapi import APIRouter, Depends, Request

from rag_api.api.middleware.rate_limit import rate_limit
from rag_api.auth.middleware import require_scope
from rag_api.rag import RAGControllerProtocol
from rag_api.schemas.identity import ClientIdentity
from rag_api.schemas.requests import FeedbackRequest, QueryRequest
from rag_api.schemas.responses import FeedbackResponse, QueryResponse
from rag_api.signals.models import QuerySignal
from rag_api.signals.store import SignalStore

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/v1")


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Submete uma pergunta ao RAG",
)
async def query(
    body:     QueryRequest,
    request:  Request,
    identity: ClientIdentity = Depends(require_scope("rag:query")),
    _:        ClientIdentity = Depends(rate_limit()),
) -> QueryResponse:
    """
    Ponto de entrada principal.
    O try/finally garante que o signal é emitido independentemente
    do resultado do pipeline — incluindo falhas por CHROMA_API_KEY em falta.
    """
    started:    float              = time.monotonic()
    trace_id:   str                = request.state.trace_id
    controller: RAGControllerProtocol = request.app.state.rag_controller
    store:      SignalStore           = request.app.state.signal_store

    response:   QueryResponse | None = None
    error_type: str | None           = None

    try:
        response = await controller.query(
            request   = body,
            client_id = identity.client_id,
            trace_id  = trace_id,
        )
        request.state.tokens_used = getattr(response, "tokens_used", None)
        return response
    except Exception as exc:
        # Captura o tipo de erro para o signal antes de propagar
        error_type = type(exc).__name__
        raise
    finally:
        await store.emit(QuerySignal(
            trace_id       = trace_id,
            client_id      = identity.client_id,
            docs_retrieved = len(response.sources) if response else 0,
            retrieval_score= 0.0,
            latency_ms     = int((time.monotonic() - started) * 1000),
            success        = response is not None,
            error_type     = error_type,
            question_chars = len(body.question),
        ))


@router.post(
    "/feedback",
    response_model=FeedbackResponse,
    summary="Regista feedback sobre uma resposta",
)
async def feedback(
    body:     FeedbackRequest,
    request:  Request,
    identity: ClientIdentity = Depends(require_scope("rag:query")),
) -> FeedbackResponse:
    """
    Aceita feedback ligado a um trace_id e associa-o ao signal correspondente.
    """
    trace_id = request.state.trace_id
    store:  SignalStore = request.app.state.signal_store

    logger.info(
        "feedback_received",
        trace_id     = trace_id,
        client_id    = identity.client_id,
        rating       = body.rating,
        target_trace = body.trace_id,
        # reason não é logado — pode conter conteúdo sensível
    )

    await store.attach_feedback(body.trace_id, body.rating, body.reason)

    return FeedbackResponse(accepted=True, trace_id=trace_id)