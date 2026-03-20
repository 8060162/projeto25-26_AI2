"""
db_client.py
------------
Responsabilidade única: inicializar e devolver a colecção ChromaDB.

A colecção é inicializada uma única vez (singleton lazy) — o modelo
BGE-M3 ocupa vários GB de memória e demora segundos a carregar.
Chamadas repetidas a get_collection() reutilizam a instância existente.
"""

import warnings

import chromadb

warnings.filterwarnings("ignore", category=UserWarning)

from settings import COLLECTION_NAME, DB_PATH
from datastore.embeddings import BGEM3EmbeddingFunction
from shared.device import resolve_device

_collection: chromadb.Collection | None = None


def get_collection() -> chromadb.Collection:
    """
    Devolve a colecção ChromaDB configurada com a função de embeddings.

    Na primeira chamada, cria o cliente persistente e carrega o modelo
    de embeddings. Chamadas subsequentes devolvem a instância em cache,
    evitando o custo de reinicialização a cada pergunta.
    """
    global _collection
    if _collection is None:
        client      = chromadb.PersistentClient(path=DB_PATH)
        _collection = client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=BGEM3EmbeddingFunction(device=resolve_device()),
        )
    return _collection