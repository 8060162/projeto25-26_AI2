"""
search.py
---------
Responsabilidade única: orquestrar a pesquisa vetorial e devolver
uma lista de ArtigoContexto sem lógica de negócio embutida.
"""

import logging

from retriever.settings import N_RESULTS, QUERY_FETCH
from retriever.db_client import get_collection
from retriever.json_store import get_artigo_completo
from retriever.models import ArtigoContexto

logger = logging.getLogger(__name__)


# ── helpers privados ──────────────────────────────────────────────────────────

def _is_truncated(meta: dict) -> bool:
    """
    Interpreta o campo 'truncated' dos metadados do ChromaDB de forma
    robusta, independentemente de estar serializado como bool ou string.
    """
    val = meta.get("truncated", False)
    if isinstance(val, bool):
        return val
    return str(val).lower() == "true"


def _resolver_conteudo(doc: str, meta: dict) -> str:
    """
    Decide qual o conteúdo a devolver para um artigo:
      - Se truncado: tenta obter o texto completo do JSON de origem.
      - Caso contrário: remove o cabeçalho técnico do chunk.

    A lógica de fallback garante que nunca se devolve uma string vazia
    quando a fonte original não está disponível.
    """
    if _is_truncated(meta):
        completo = get_artigo_completo(meta["source"], meta["artigo_id"])
        if completo is not None:
            return completo
        logger.warning(
            "Não foi possível expandir artigo truncado '%s' de '%s'. "
            "A usar conteúdo do chunk.",
            meta.get("artigo_id"), meta.get("source"),
        )
        return doc

    # Remove cabeçalho técnico inserido no momento de indexação
    return doc.split("\n\n", 1)[-1] if "\n\n" in doc else doc


# ── interface pública ─────────────────────────────────────────────────────────

def procurar_contexto(
    pergunta: str,
    n_resultados: int = N_RESULTS,
) -> list[ArtigoContexto]:
    """
    Executa uma pesquisa semântica e devolve até `n_resultados` artigos.

    A deduplicação é feita por (source, artigo_id) para evitar que
    múltiplos chunks do mesmo artigo apareçam no resultado.

    Args:
        pergunta:     texto da questão do utilizador.
        n_resultados: número máximo de artigos a devolver.

    Returns:
        Lista de ArtigoContexto ordenada por relevância.
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
        id_unico = (meta["source"], meta["artigo_id"])
        if id_unico in vistos:
            continue
        vistos.add(id_unico)


        artigos.append(ArtigoContexto(
            artigo_id=meta["artigo_id"],
            conteudo=_resolver_conteudo(doc, meta),
            source=meta["source"],
            capitulo_titulo=meta.get("capitulo") or None,
            artigo_titulo=meta.get("doc_titulo") or None,
            pagina=meta.get("pagina") or None,
        ))


        if len(artigos) >= n_resultados:
            break

    return artigos