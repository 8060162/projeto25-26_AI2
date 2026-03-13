import os
import json
import chromadb
import torch
import warnings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings

# Silenciar avisos do Python 3.14/Pydantic
warnings.filterwarnings("ignore", category=UserWarning)

# --- CONFIGURAÇÃO DE CAMINHOS ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT_FOLDER = os.path.join(BASE_DIR, "data", "processed")
DB_PATH = os.path.join(BASE_DIR, "data", "chromaDB")
COLLECTION_NAME = "artigos_legislativos"

os.makedirs(DB_PATH, exist_ok=True)

# 1. Classe de Embedding BGE-M3 (Corrigida para evitar DeprecationWarning)
class BGEM3EmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_name="BAAI/bge-m3", device="cpu"):
        print(f"--- Inicializando Modelo BGE-M3 em {device.upper()} ---")
        self.model = SentenceTransformer(model_name, device=device)

    def __call__(self, input: Documents) -> Embeddings:
        # normalize_embeddings=True é crucial para o BGE-M3
        return self.model.encode(list(input), normalize_embeddings=True).tolist()

# Detetar hardware
device = "mps" if torch.backends.mps.is_available() else "cpu"
embedding_fn = BGEM3EmbeddingFunction(device=device)

# 2. Inicializar ChromaDB
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_fn
)

# 3. Definir o text_splitter (ESTA ERA A PEÇA QUE FALTA)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=150,
    separators=["\n\n", "\n", ". ", " ", ""]
)

def run_ingestion():
    if not os.path.exists(INPUT_FOLDER):
        print(f"ERRO: Pasta de entrada não encontrada: {INPUT_FOLDER}")
        return

    json_files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith('.json')]
    if not json_files:
        print(f"Nenhum ficheiro JSON encontrado em: {INPUT_FOLDER}")
        return

    print(f"A processar {len(json_files)} ficheiros...")

    for filename in json_files:
        file_path = os.path.join(INPUT_FOLDER, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except Exception as e:
                print(f"Erro ao ler {filename}: {e}")
                continue
            
            estrutura = data.get("estrutura", {})
            for cap_id, cap_data in estrutura.items():
                cap_titulo = cap_data.get("titulo", "Sem Título")
                artigos = cap_data.get("artigos", {})

                for art_id, art_data in artigos.items():
                    art_titulo = art_data.get("titulo", "")
                    conteudo = art_data.get("conteudo", "")
                    pagina = art_data.get("pagina", "N/A")

                    # Texto para indexar
                    texto_final = (
                        f"FICHEIRO: {filename}\n"
                        f"CAPÍTULO: {cap_titulo}\n"
                        f"ARTIGO: {art_id} - {art_titulo}\n"
                        f"CONTEÚDO: {conteudo}"
                    )

                    metadata = {
                        "source": filename,
                        "capitulo": cap_titulo,
                        "artigo_id": art_id,
                        "pagina": str(pagina)
                    }

                    doc_id = f"{filename}_{art_id}".replace(" ", "_")

                    # Uso do splitter definido acima
                    if len(conteudo) > 1200:
                        chunks = text_splitter.split_text(texto_final)
                        for i, chunk_txt in enumerate(chunks):
                            collection.upsert(
                                documents=[chunk_txt],
                                metadatas=[{**metadata, "part": i}],
                                ids=[f"{doc_id}_p{i}"]
                            )
                    else:
                        collection.upsert(
                            documents=[texto_final],
                            metadatas=[metadata],
                            ids=[doc_id]
                        )
        
        print(f"✓ {filename} indexado.")

if __name__ == "__main__":
    run_ingestion()
    print(f"\n--- Ingestão Concluída com Sucesso em {DB_PATH} ---")