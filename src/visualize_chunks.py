"""
visualize_chunks.py
-------------------
para vizualizar colocar no browser: http://localhost:7860
Exporta os chunks do ChromaDB para o Spotlight (Renumics),
permitindo visualizar a qualidade e dispersão dos embeddings no espaço vectorial.

Uso:
    python src/visualize_chunks.py

Requisito:
    pip install renumics-spotlight
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from settings import DB_PATH, COLLECTION_NAME
from datastore.embeddings import BGEM3EmbeddingFunction
from shared.device import resolve_device


def exportar_chunks() -> pd.DataFrame:
    """
    Lê todos os chunks e embeddings do ChromaDB e devolve um DataFrame
    pronto para o Spotlight.
    """
    import chromadb

    print(f"A conectar ao ChromaDB em: {DB_PATH}")
    client     = chromadb.PersistentClient(path=DB_PATH)
    embed_fn   = BGEM3EmbeddingFunction(device=resolve_device())
    collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
    )

    total = collection.count()
    print(f"Total de chunks na colecção '{COLLECTION_NAME}': {total}")

    if total == 0:
        print("Colecção vazia — executa primeiro o pipeline de ingestão.")
        sys.exit(1)

    # Lê todos os chunks, metadados e embeddings de uma vez
    resultado = collection.get(
        include=["documents", "metadatas", "embeddings"],
        limit=total,
    )

    docs       = resultado["documents"]
    metas      = resultado["metadatas"]
    embeddings = np.array(resultado["embeddings"])

    print(f"Embeddings carregados: shape={embeddings.shape}")

    # Constrói o DataFrame
    registos = []
    for i, (doc, meta) in enumerate(zip(docs, metas)):
        registos.append({
            # Conteúdo
            "chunk_texto":    doc,
            "chunk_tamanho":  len(doc),

            # Metadados de rastreabilidade
            "source":         meta.get("source",     ""),
            "artigo_id":      meta.get("artigo_id",  ""),
            "art_titulo":     meta.get("art_titulo",  ""),
            "capitulo":       meta.get("capitulo",   ""),
            "doc_titulo":     meta.get("doc_titulo", ""),
            "pagina":         meta.get("pagina",     ""),
            "truncated":      meta.get("truncated",  "false"),
            "part":           meta.get("part",       0),
        })

    df = pd.DataFrame(registos)

    # Adiciona embeddings como coluna do tipo Embedding (numpy array por linha)
    df["embedding"] = list(embeddings)

    print(f"DataFrame criado: {len(df)} linhas, {len(df.columns)} colunas")
    return df


def main():
    try:
        from renumics import spotlight
    except ImportError:
        print("Spotlight não instalado. Executa: pip install renumics-spotlight")
        sys.exit(1)

    df = exportar_chunks()

    print("\nA abrir o Spotlight no browser...")
    print("Usa o painel de visualização para explorar os embeddings em 2D/3D (UMAP/PCA).")
    print("Filtra por 'source', 'capitulo', ou 'truncated' para inspecionar grupos.")
    print("Carrega em Ctrl+C para terminar.\n")

    spotlight.show(
        df,
        dtype={"embedding": spotlight.Embedding},
        port=7860,
        wait=True,
    )


if __name__ == "__main__":
    main()