"""
db_client.py
------------
Responsabilidade única: inicializar e devolver a colecção ChromaDB.
"""

import warnings

import chromadb
import torch

warnings.filterwarnings("ignore", category=UserWarning)

from retriever.settings import COLLECTION_NAME, DB_PATH
from datastore.embeddings import BGEM3EmbeddingFunction


def _resolve_device() -> str:
    """Devolve o dispositivo de inferência disponível."""
    return "mps" if torch.backends.mps.is_available() else "cpu"


def get_collection() -> chromadb.Collection:
    """
    Devolve a colecção ChromaDB configurada com a função de embeddings.

    O cliente é criado de forma persistente no caminho definido em settings.
    O dispositivo (MPS ou CPU) é resolvido automaticamente.
    """
    client = chromadb.PersistentClient(path=DB_PATH)
    return client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=BGEM3EmbeddingFunction(device=_resolve_device()),
    )