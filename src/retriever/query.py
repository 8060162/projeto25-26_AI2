import os
import chromadb
import torch
import warnings
from sentence_transformers import SentenceTransformer
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings

warnings.filterwarnings("ignore", category=UserWarning)

# --- CONFIGURAÇÃO ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_PATH = os.path.join(BASE_DIR, "data", "chromaDB")
COLLECTION_NAME = "artigos_legislativos"

class BGEM3EmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_name="BAAI/bge-m3", device="cpu"):
        self.model = SentenceTransformer(model_name, device=device)
    def __call__(self, input: Documents) -> Embeddings:
        return self.model.encode(list(input), normalize_embeddings=True).tolist()

def get_retriever():
    """Inicializa e retorna a coleção do ChromaDB."""
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    embedding_fn = BGEM3EmbeddingFunction(device=device)
    client = chromadb.PersistentClient(path=DB_PATH)
    return client.get_collection(name=COLLECTION_NAME, embedding_function=embedding_fn)

def procurar_contexto(pergunta, n_resultados=3):
    """Função que será chamada pelo Generator."""
    collection = get_retriever()
    results = collection.query(query_texts=[pergunta], n_results=n_resultados)
    return results['documents'][0]

if __name__ == "__main__":
    # Permite testar o query sozinho no terminal
    p = input("Teste de busca: ")
    res = procurar_contexto(p)
    for r in res:
        print(f"\nEncontrado: {r[:1000]}...")