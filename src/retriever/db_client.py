"""
db_client.py
------------
Responsabilidade única: inicializar e devolver a colecção ChromaDB.

A colecção é inicializada uma única vez (singleton lazy) — o modelo
BGE-M3 ocupa vários GB de memória e demora segundos a carregar.
Chamadas repetidas a get_collection() reutilizam a instância existente.

ALTERAÇÃO (refactor): a inicialização do singleton passou a ser protegida
por threading.Lock. Sem esta protecção, dois threads a chamar get_collection()
simultaneamente podiam inicializar _collection duas vezes (condição de corrida).
O padrão double-checked locking evita o custo do lock após a primeira
inicialização, mantendo correcta a semântica singleton em contextos
multi-thread (ex: servidor web com múltiplos workers).
"""

import threading
import warnings

import chromadb

warnings.filterwarnings("ignore", category=UserWarning)

from settings import COLLECTION_NAME, DB_PATH
from datastore.embeddings import BGEM3EmbeddingFunction
from shared.device import resolve_device

_collection: chromadb.Collection | None = None
_lock = threading.Lock()


def get_collection() -> chromadb.Collection:
    """
    Devolve a colecção ChromaDB configurada com a função de embeddings.

    Na primeira chamada, cria o cliente persistente e carrega o modelo
    de embeddings. Chamadas subsequentes devolvem a instância em cache,
    evitando o custo de reinicialização a cada pergunta.

    Thread-safe: usa double-checked locking para garantir inicialização
    única mesmo em contextos concorrentes.
    """
    global _collection

    # Fast path: sem lock se já inicializado (caso mais comum)
    if _collection is not None:
        return _collection

    with _lock:
        # Segunda verificação dentro do lock — outro thread pode ter
        # inicializado entre o primeiro if e a aquisição do lock.
        if _collection is None:
            client      = chromadb.PersistentClient(path=DB_PATH)
            _collection = client.get_collection(
                name=COLLECTION_NAME,
                embedding_function=BGEM3EmbeddingFunction(device=resolve_device()),
            )

    return _collection