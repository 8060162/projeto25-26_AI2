import structlog
from fastapi import APIRouter, Depends, Request

from rag_api.api.middleware.rate_limit import rate_limit
from rag_api.auth.middleware import require_scope
from rag_api.rag import RAGControllerProtocol
from rag_api.schemas.identity import ClientIdentity
from rag_api.schemas.requests import FeedbackRequest, QueryRequest
from rag_api.schemas.responses import FeedbackResponse, QueryResponse

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
    Auth e rate limit são resolvidos pelas dependências — o handler
    só conhece o contrato de negócio: recebe uma pergunta, devolve uma resposta.
    """
    trace_id:   str                 = request.state.trace_id
    controller: RAGControllerProtocol = request.app.state.rag_controller

    response = await controller.query(
        request   = body,
        client_id = identity.client_id,
        trace_id  = trace_id,
    )

    # Regista tokens usados para o access log (lido pelo middleware no fim)
    request.state.tokens_used = getattr(response, "tokens_used", None)

    return response


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
    Aceita feedback ligado a um trace_id.
    Alimenta a intelligence layer (módulo 6) — processamento assíncrono.
    """
    trace_id = request.state.trace_id

    logger.info(
        "feedback_received",
        trace_id       = trace_id,
        client_id      = identity.client_id,
        rating         = body.rating,
        target_trace   = body.trace_id,
        # reason não é logado — pode conter conteúdo sensível
    )

    # TODO(módulo 6): emitir evento para a intelligence layer
    return FeedbackResponse(accepted=True, trace_id=trace_id)