"""
db_client.py
------------
Responsabilidade única: inicializar e devolver a colecção ChromaDB.
"""

import os
import sys
import warnings
import chromadb
import torch

warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datastore.config     import DB_PATH, COLLECTION_NAME
from datastore.embeddings import BGEM3EmbeddingFunction


def get_collection() -> chromadb.Collection:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    client = chromadb.PersistentClient(path=DB_PATH)
    return client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=BGEM3EmbeddingFunction(device=device),
    )