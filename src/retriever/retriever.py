import os
import sys
import chromadb
import torch

# Adicionar src/ ao path para encontrar módulos irmãos
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datastore.config import BASE_DIR, DB_PATH, COLLECTION_NAME, N_RESULTS
from datastore.embeddings import BGEM3EmbeddingFunction


def get_retriever():
    """Inicializa e retorna a coleção do ChromaDB."""
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    embedding_fn = BGEM3EmbeddingFunction(device=device)
    client = chromadb.PersistentClient(path=DB_PATH)
    return client.get_collection(name=COLLECTION_NAME, embedding_function=embedding_fn)


def procurar_contexto(pergunta, n_resultados=N_RESULTS):
    """Função que será chamada pelo Generator."""
    collection = get_retriever()
    results = collection.query(query_texts=[pergunta], n_results=n_resultados)
    return results['documents'][0]


if __name__ == "__main__":
    p = input("Teste de busca: ")
    res = procurar_contexto(p)
    for r in res:
        print(f"\nEncontrado: {r[:1000]}...")