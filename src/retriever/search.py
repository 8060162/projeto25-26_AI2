"""
search.py
---------
Responsabilidade única: orquestrar a pesquisa vetorial e devolver
uma lista de ArtigoContexto sem lógica de negócio embutida.
"""

import logging

from settings import N_RESULTS, QUERY_FETCH, CHUNK_HEADER_SEP
from retriever.db_client import get_collection
from retriever.json_store import get_artigo_completo
from retriever.models import ArtigoContexto
from shared.metadata_keys import MetaKey

logger = logging.getLogger(__name__)


# ── helpers privados ──────────────────────────────────────────────────────────

def _is_truncated(meta: dict) -> bool:
    """
    Interpreta o campo 'truncated' dos metadados do ChromaDB de forma
    robusta, independentemente de estar serializado como bool ou string.
    """
    val = meta.get(MetaKey.TRUNCATED, False)
    if isinstance(val, bool):
        return val
    return str(val).lower() == "true"


def _remover_cabecalho(chunk: str) -> str:
    """
    Remove o cabeçalho contextual inserido pelo chunker no momento de indexação.

    O separador é importado de settings.CHUNK_HEADER_SEP, garantindo que
    chunker.py (escrita) e esta função (leitura) estão sempre sincronizados.
    """
    if CHUNK_HEADER_SEP in chunk:
        return chunk.split(CHUNK_HEADER_SEP, 1)[-1]
    return chunk


def _expandir_se_truncado(doc: str, meta: dict) -> str:
    """
    Se o chunk estiver marcado como truncado, tenta obter o texto
    completo do artigo a partir do ficheiro JSON de origem.

    Devolve o chunk original como fallback se a expansão falhar,
    garantindo que nunca se devolve uma string vazia.
    """
    if not _is_truncated(meta):
        return doc

    completo = get_artigo_completo(meta[MetaKey.SOURCE], meta[MetaKey.ARTIGO_ID])
    if completo is not None:
        return completo

    logger.warning(
        "Não foi possível expandir artigo truncado '%s' de '%s'. "
        "A usar conteúdo do chunk.",
        meta.get(MetaKey.ARTIGO_ID),
        meta.get(MetaKey.SOURCE),
    )
    return doc


def _resolver_conteudo(doc: str, meta: dict) -> str:
    """
    Orquestra a resolução do conteúdo final de um chunk:
      1. Expande o artigo completo se truncado.
      2. Remove o cabeçalho técnico do chunk não truncado.
    """
    conteudo = _expandir_se_truncado(doc, meta)
    if _is_truncated(meta):
        return conteudo
    return _remover_cabecalho(conteudo)


# ── interface pública ─────────────────────────────────────────────────────────

def procurar_contexto(
    pergunta: str,
    n_resultados: int = N_RESULTS,
) -> list[ArtigoContexto]:
    """
    Executa uma pesquisa semântica e devolve até `n_resultados` artigos.

    A deduplicação é feita por (source, artigo_id) para evitar que
    múltiplos chunks do mesmo artigo apareçam no resultado.
    """
    collection = get_collection()

    results = collection.query(
        query_texts=[pergunta],
        n_results=min(QUERY_FETCH, collection.count()),
        include=["documents", "metadatas"],
    )

    artigos: list[ArtigoContexto] = []
    vistos: set[tuple[str, str]] = set()

    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        id_unico = (meta[MetaKey.SOURCE], meta[MetaKey.ARTIGO_ID])
        if id_unico in vistos:
            continue
        vistos.add(id_unico)

        artigos.append(ArtigoContexto(
            artigo_id=meta[MetaKey.ARTIGO_ID],
            conteudo=_resolver_conteudo(doc, meta),
            source=meta[MetaKey.SOURCE],
            capitulo_titulo=meta.get(MetaKey.CAPITULO)  or None,
            artigo_titulo=meta.get(MetaKey.ART_TITULO)  or None,
            pagina=meta.get(MetaKey.PAGINA)              or None,
        ))

        if len(artigos) >= n_resultados:
            break

    return artigos