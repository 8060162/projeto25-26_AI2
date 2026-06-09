import uuid
import asyncio
import structlog
from typing import Protocol

from rag_api.schemas.requests import QueryRequest
from rag_api.schemas.responses import QueryResponse

logger = structlog.get_logger(__name__)


class RAGControllerProtocol(Protocol):
    async def query(self, request: QueryRequest, client_id: str, trace_id: str) -> QueryResponse: ...


class StubRAGController:
    async def query(self, request: QueryRequest, client_id: str, trace_id: str) -> QueryResponse:
        return QueryResponse(
            answer=f"[stub] Resposta à pergunta: '{request.question}'",
            sources=["doc://stub/regulamento-1"],
            trace_id=trace_id,
            session_id=request.session_id or str(uuid.uuid4()),
        )


class RAGController:
    def __init__(self, retrieval_service) -> None:
        self._service = retrieval_service

    async def query(self, request: QueryRequest, client_id: str, trace_id: str) -> QueryResponse:
        loop   = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, self._service.answer_question, request.question
        )

        meta      = result.answer_metadata or {}
        quality   = meta.get("retrieval_quality", {})
        q_meta    = quality.get("metadata", {})
        evidence  = meta.get("route_metadata", {}).get("evidence_quality", {})
        grounding = meta.get("route_metadata", {}).get("grounding_verification") or {}
        usage     = meta.get("usage", {})

        # query_category derivado dos legal_intent_signals classificados pelo pipeline
        # Usa o primeiro intent detectado — "uncategorized" se nenhum disponível
        legal_intents = meta.get("query_metadata", {}).get("legal_intent_signals", [])
        query_category = legal_intents[0] if legal_intents else "uncategorized"

        response = QueryResponse(
            answer=result.answer_text or "",
            sources=[str(c) for c in q_meta.get("selected_chunk_ids", [])],
            trace_id=trace_id,
            session_id=request.session_id or str(uuid.uuid4()),
        )

        # Transporta métricas enriquecidas para consumo no finally da route
        object.__setattr__(response, "_signal_meta", {
            "docs_retrieved":   quality.get("context_chunk_count", 0),
            "retrieval_score":  float(q_meta.get("primary_anchor_score") or 0.0),
            "query_category":   query_category,
            "grounded":         meta.get("grounded", False),
            "evidence_strength": evidence.get("strength", "unknown"),
            "grounding_status": grounding.get("status", "unknown"),
            "tokens_input":     usage.get("prompt_tokens", 0),
            "tokens_output":    usage.get("completion_tokens", 0),
        })

        return response


def create_rag_controller():
    try:
        from retrieval.service import create_retrieval_service
        return RAGController(create_retrieval_service())
    except ImportError:
        logger.warning("retrieval_service_unavailable", reason="using stub")
        return StubRAGController()