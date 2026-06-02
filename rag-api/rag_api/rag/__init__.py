from typing import Protocol
import uuid
import asyncio

from rag_api.schemas.requests import QueryRequest
from rag_api.schemas.responses import QueryResponse


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
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self._service.answer_question, request.question)
        return QueryResponse(
            answer=result.answer_text or "",
            sources=[str(c) for c in (result.answer_metadata or {}).get("retrieved_chunk_ids", [])],
            trace_id=trace_id,
            session_id=request.session_id or str(uuid.uuid4()),
        )


def create_rag_controller():
    try:
        from retrieval.service import create_retrieval_service
        return RAGController(create_retrieval_service())
    except ImportError:
        import structlog
        structlog.get_logger(__name__).warning("retrieval_service_unavailable", reason="using stub")
        return StubRAGController()
