"""
analysis/_chromadb_loader.py
-----------------------------
Camada de acesso partilhada ao ChromaDB para todos os scripts de análise.

Responsabilidade única: abrir a colecção e devolver os dados brutos
(documentos, metadados, embeddings) num DataFrame normalizado.

NOTA IMPORTANTE — sem modelo de embeddings:
    Para leitura de embeddings já existentes, o ChromaDB não precisa de
    saber como gerá-los. Passar embedding_function a get_collection()
    forçaria o carregamento do BGE-M3 e uma chamada à HuggingFace — o
    que é desnecessário, lento, e requer ligação à internet.
    Usamos get_collection() sem embedding_function, que é o padrão
    correcto para operações de leitura pura.

Nenhum script de análise deve aceder directamente ao ChromaDB —
todos importam `carregar_dataframe()` deste módulo.
"""

import os
import sys
import warnings
import logging

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

# Garante que src/ está no path independentemente de onde o script é invocado
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from settings import DB_PATH, COLLECTION_NAME

logger = logging.getLogger(__name__)


def carregar_dataframe() -> pd.DataFrame:
    """
    Abre a colecção ChromaDB em modo de leitura e devolve um DataFrame com:
      - chunk_texto     : conteúdo textual do chunk
      - chunk_tamanho   : número de caracteres do chunk
      - embedding       : vector numpy (1024 dims para BGE-M3)
      - source          : nome do ficheiro JSON de origem
      - doc_titulo      : título do documento
      - capitulo        : título do capítulo
      - artigo_id       : identificador do artigo
      - art_titulo      : título do artigo
      - pagina          : página no documento original
      - truncated       : "true" se o artigo foi dividido em múltiplos chunks
      - part            : índice do chunk dentro do artigo (se truncated)

    Não instancia nenhum modelo de embeddings — lê os vectores
    directamente do ChromaDB sem geração de novos embeddings.

    Raises:
        SystemExit: se a colecção estiver vazia ou o ChromaDB inacessível.
    """
    import chromadb

    logger.info("A conectar ao ChromaDB em: %s", DB_PATH)
    client = chromadb.PersistentClient(path=DB_PATH)

    # Sem embedding_function — leitura pura, sem carregamento de modelo
    collection = client.get_collection(name=COLLECTION_NAME)

    total = collection.count()
    logger.info("Total de chunks na colecção '%s': %d", COLLECTION_NAME, total)

    if total == 0:
        logger.error("Colecção vazia — executa primeiro o pipeline de ingestão.")
        sys.exit(1)

    resultado = collection.get(
        include=["documents", "metadatas", "embeddings"],
        limit=total,
    )

    docs       = resultado["documents"]
    metas      = resultado["metadatas"]
    embeddings = np.array(resultado["embeddings"])

    logger.info("Embeddings carregados: shape=%s", embeddings.shape)

    registos = []
    for doc, meta in zip(docs, metas):
        registos.append({
            "chunk_texto":   doc,
            "chunk_tamanho": len(doc),
            "source":        meta.get("source",     ""),
            "doc_titulo":    meta.get("doc_titulo", ""),
            "capitulo":      meta.get("capitulo",   ""),
            "artigo_id":     meta.get("artigo_id",  ""),
            "art_titulo":    meta.get("art_titulo", ""),
            "pagina":        meta.get("pagina",     ""),
            "truncated":     meta.get("truncated",  "false"),
            "part":          meta.get("part",       0),
        })

    df = pd.DataFrame(registos)
    df["embedding"] = list(embeddings)

    logger.info("DataFrame pronto: %d linhas, %d colunas", len(df), len(df.columns))
    return df