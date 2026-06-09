"""
Cliente ChromaDB para operações do módulo pipeline.

Responsabilidade única: delete cirúrgico por doc_id e contagem de chunks.
A indexação (write) é responsabilidade do indexer do colega —
este cliente só lida com as operações de gestão pós-indexação.
"""
from __future__ import annotations

import structlog

logger = structlog.get_logger(__name__)


class ChromaPipelineClient:
    """
    Wrapper fino sobre o cliente ChromaDB.

    Recebe a collection já instanciada — sem lógica de conexão aqui,
    a conexão é responsabilidade do caller (injecção de dependência).
    """

    def __init__(self, collection) -> None:
        # collection: chromadb.Collection — tipagem genérica para evitar
        # acoplamento ao chromadb no momento da importação.
        self._collection = collection

    async def delete_chunks(self, chroma_ids: list[str]) -> None:
        """
        Remove chunks pelo seu ID exacto.

        Best-effort — se a collection não estiver disponível (ex: CHROMA_API_KEY
        ausente no startup), regista um warning e continua sem falhar.
        Operação síncrona do chromadb envolta em executor para não
        bloquear o event loop — padrão consistente com pipeline.py existente.
        """
        if not chroma_ids:
            return

        if self._collection is None:
            logger.warning("chroma_delete_skipped", reason="collection_unavailable", count=len(chroma_ids) if isinstance(chroma_ids, list) else chroma_ids)
            return

        import asyncio
        loop = asyncio.get_event_loop()

        def _delete():
            self._collection.delete(ids=chroma_ids)

        await loop.run_in_executor(None, _delete)
        logger.info("chroma_chunks_deleted", count=len(chroma_ids))

    async def count_chunks_for_doc(self, doc_id: str) -> int:
        """Contagem de chunks activos para um doc_id — usado em reindex para validar."""
        import asyncio
        loop = asyncio.get_event_loop()

        def _count():
            result = self._collection.get(
                where={"doc_id": {"$eq": doc_id}},
                include=[],
            )
            return len(result.get("ids", []))

        return await loop.run_in_executor(None, _count)