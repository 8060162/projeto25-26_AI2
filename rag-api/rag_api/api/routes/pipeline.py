import asyncio
import structlog
from fastapi import APIRouter, Depends, Request

from rag_api.auth.middleware import require_scope
from rag_api.pipeline.schemas import EmbedResponse
from rag_api.schemas.identity import ClientIdentity

logger = structlog.get_logger(__name__)

# Prefix actualizado para /v1/pipeline — consistente com o novo módulo pipeline.
# EmbedResponse movido para rag_api/pipeline/schemas.py (SSOT).
router = APIRouter(prefix="/v1/pipeline", tags=["pipeline"])


@router.post(
    "/embed",
    response_model=EmbedResponse,
    summary="Corre chunking + embedding e actualiza a Vector DB",
)
async def embed(
    request:  Request,
    identity: ClientIdentity = Depends(require_scope("rag:admin")),
) -> EmbedResponse:
    """
    Dispara o pipeline completo: chunking → embedding → indexação no ChromaDB.
    Apenas Admin.

    Este endpoint é síncrono do ponto de vista do pipeline — corre em executor
    para não bloquear o event loop do FastAPI.
    """
    trace_id = request.state.trace_id

    logger.info("embed_started", triggered_by=identity.client_id, trace_id=trace_id)

    loop = asyncio.get_event_loop()

    try:
        result = await loop.run_in_executor(None, _run_pipeline)
    except Exception as exc:
        logger.error("embed_failed", error=repr(exc), trace_id=trace_id)
        raise

    logger.info(
        "embed_completed",
        run_id               = result.run_id,
        embedded_record_count = result.embedded_record_count,
        trace_id             = trace_id,
    )

    return EmbedResponse(
        run_id                = result.run_id,
        input_record_count    = result.input_record_count,
        embedded_record_count = result.embedded_record_count,
        records_path          = result.records_path,
        manifest_path         = result.manifest_path,
    )


def _run_pipeline():
    """
    Corre o pipeline completo.
    Importação lazy — só falha se o pipeline não estiver no sys.path.
    """
    from embedding.indexer import run_embedding_indexer
    return run_embedding_indexer()