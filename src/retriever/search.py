"""
retriever.py
------------
Responsabilidade única: orquestrar a pesquisa e devolver artigos completos.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retriever.config     import N_RESULTS, QUERY_FETCH
from retriever.db_client  import get_collection
from retriever.json_store import get_artigo_completo
from retriever.models     import ArtigoContexto


def procurar_contexto(pergunta: str, n_resultados: int = N_RESULTS) -> list[ArtigoContexto]:
    collection = get_collection()

    results = collection.query(
        query_texts=[pergunta],
        n_results=min(QUERY_FETCH, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    vistos: set[tuple] = set()
    artigos: list[ArtigoContexto] = []

    for doc, meta, _ in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chave = (meta["source"], meta["artigo_id"])
        if chave in vistos:
            continue
        vistos.add(chave)

        conteudo = (
            get_artigo_completo(meta["source"], meta["artigo_id"]) or doc
            if meta.get("truncated") == "true"
            else doc
        )

        artigos.append(ArtigoContexto(
            artigo_id       = meta.get("artigo_id", ""),
            artigo_titulo   = meta.get("artigo_titulo", ""),
            capitulo_titulo = meta.get("capitulo_titulo", ""),
            doc_numero      = meta.get("doc_numero", ""),
            pagina          = meta.get("pagina", ""),
            source          = meta.get("source", ""),
            conteudo        = conteudo,
        ))

        if len(artigos) == n_resultados:
            break

    return artigos