import warnings
from sentence_transformers import SentenceTransformer
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings

warnings.filterwarnings("ignore", category=UserWarning)

# model_name = "stjiris/bert-large-portuguese-cased-legal-mlm-sts-v1.0"
# model_name="BAAI/bge-m3"
class BGEM3EmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_name="BAAI/bge-m3", device="cpu"):
        self.model = SentenceTransformer(model_name, device=device)

    def __call__(self, input: Documents) -> Embeddings:
        return self.model.encode(list(input), normalize_embeddings=True).tolist()