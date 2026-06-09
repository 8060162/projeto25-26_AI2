import time
import structlog
from fastapi import APIRouter, Depends, Request

from rag_api.api.middleware.rate_limit import rate_limit
from rag_api.auth.middleware import require_scope
from rag_api.memory.schemas import InteractionPayload, MemoryCreateRequest
from rag_api.rag import RAGControllerProtocol
from rag_api.schemas.identity import ClientIdentity
from rag_api.schemas.requests import FeedbackRequest, QueryRequest
from rag_api.schemas.responses import FeedbackResponse, QueryResponse
from rag_api.signals.models import QuerySignal
from rag_api.signals.store import SignalStore

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/v1")


async def _persist_memory(
    request:   Request,
    query_req: QueryRequest,
    response:  QueryResponse,
    client_id: str,
    trace_id:  str,
) -> None:
    """
    Persiste a interacção na memória longa — best-effort.
    Falha silenciosa com log: a query já foi respondida, a memória é secundária.

    Canal autenticado → user_token presente → memória pessoal do aluno.
    Canal anónimo     → user_token ausente  → memória da aplicação para métricas.

    topic e importance lidos do _signal_meta que o pipeline já calcula:
      topic      ← query_category  (mesmo conceito — refactoring futuro)
      importance ← retrieval_score normalizado para 0.0–10.0

    Fallbacks defensivos para quando o pipeline não preenche _signal_meta.
    """
    try:
        signal_meta = getattr(response, "_signal_meta", {})
        topic       = signal_meta.get("query_category", "uncategorized")
        # retrieval_score vem em 0.0–1.0 — normalizar para a escala 0.0–10.0 da memória
        raw_score   = signal_meta.get("retrieval_score", 0.5)
        importance  = round(min(max(float(raw_score) * 10, 0.0), 10.0), 2)

        memory_req = MemoryCreateRequest(
            user_token  = query_req.user_token,
            session_id  = query_req.session_id or trace_id,
            type        = "query_interaction",
            topic       = topic,
            importance  = importance,
            interaction = InteractionPayload(
                question = query_req.question,
                answer   = response.answer,
                sources  = response.sources,
                trace_id = trace_id,
            ),
        )

        from datetime import datetime, timedelta, timezone
        from rag_api.memory.keys import derive_user_key

        pepper     = request.app.state.memory_hmac_pepper
        ttl_days   = request.app.state.memory_ttl_days
        user_key   = derive_user_key(pepper, client_id, query_req.user_token)
        mode       = "authenticated" if query_req.user_token else "anonymous"
        expires_at = datetime.now(timezone.utc) + timedelta(days=ttl_days)

        await request.app.state.memory_repo.create(
            user_key   = user_key,
            client_id  = client_id,
            mode       = mode,
            req        = memory_req,
            expires_at = expires_at,
        )
    except Exception:
        # Não propaga — a resposta ao utilizador não é afectada por falhas de memória
        logger.exception(
            "memory_persist_failed",
            trace_id  = trace_id,
            client_id = client_id,
            mode      = "authenticated" if query_req.user_token else "anonymous",
        )


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
    Ponto de entrada principal — dois canais de memória:
      user_token presente → memória autenticada (aluno identificado)
      user_token ausente  → memória anónima (métricas da aplicação)

    Ordem de operações:
      1. Pipeline RAG → resposta
      2. Memória longa → best-effort após resposta bem sucedida
      3. QuerySignal  → sempre (finally)
    """
    started:    float                 = time.monotonic()
    trace_id:   str                   = request.state.trace_id
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

        # Memória longa — apenas após resposta bem sucedida
        await _persist_memory(request, body, response, identity.client_id, trace_id)

        return response
    except Exception as exc:
        error_type = type(exc).__name__
        raise
    finally:
        signal_meta = getattr(response, "_signal_meta", {}) if response else {}

        await store.emit(QuerySignal(
            trace_id          = trace_id,
            client_id         = identity.client_id,
            docs_retrieved    = signal_meta.get("docs_retrieved", len(response.sources) if response else 0),
            retrieval_score   = signal_meta.get("retrieval_score", 0.0),
            latency_ms        = int((time.monotonic() - started) * 1000),
            success           = response is not None,
            error_type        = error_type,
            question_chars    = len(body.question),
            tokens_input      = signal_meta.get("tokens_input", 0),
            tokens_output     = signal_meta.get("tokens_output", 0),
            query_category    = signal_meta.get("query_category", "uncategorized"),
            grounded          = signal_meta.get("grounded", False),
            evidence_strength = signal_meta.get("evidence_strength", "unknown"),
            grounding_status  = signal_meta.get("grounding_status", "unknown"),
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