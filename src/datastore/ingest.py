import os
import warnings
import chromadb
import torch

warnings.filterwarnings("ignore", category=UserWarning)

# --- CONFIGURAÇÃO DE CAMINHOS ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT_FOLDER = os.path.join(BASE_DIR, "data", "processed")
DB_PATH = os.path.join(BASE_DIR, "data", "chromaDB")
COLLECTION_NAME = "artigos_legislativos"

os.makedirs(DB_PATH, exist_ok=True)

from embeddings import BGEM3EmbeddingFunction
from document_parser import parse_ficheiro
from chunker import construir_texto, dividir_em_chunks


def run_ingestion():
    if not os.path.exists(INPUT_FOLDER):
        print(f"ERRO: Pasta de entrada não encontrada: {INPUT_FOLDER}")
        return

    json_files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith('.json')]
    if not json_files:
        print(f"Nenhum ficheiro JSON encontrado em: {INPUT_FOLDER}")
        return

    # Inicializar ChromaDB e modelo — uma única vez para todos os ficheiros
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    embedding_fn = BGEM3EmbeddingFunction(device=device)
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
    )

    print(f"A processar {len(json_files)} ficheiros...")

    for filename in json_files:
        filepath = os.path.join(INPUT_FOLDER, filename)
        try:
            artigos = parse_ficheiro(filepath, filename)
        except Exception as e:
            print(f"Erro ao ler {filename}: {e}")
            continue

        for artigo in artigos:
            texto = construir_texto(
                filename=artigo.filename,
                cap_titulo=artigo.cap_titulo,
                art_id=artigo.art_id,
                art_titulo=artigo.art_titulo,
                conteudo=artigo.conteudo,
            )
            chunks = dividir_em_chunks(texto, artigo.conteudo)
            doc_id = f"{filename}_{artigo.art_id}".replace(" ", "_")
            metadata = {
                "source": artigo.filename,
                "capitulo": artigo.cap_titulo,
                "artigo_id": artigo.art_id,
                "pagina": artigo.pagina,
            }

            for i, chunk in enumerate(chunks):
                collection.upsert(
                    documents=[chunk],
                    metadatas=[{**metadata, "part": i} if len(chunks) > 1 else metadata],
                    ids=[f"{doc_id}_p{i}" if len(chunks) > 1 else doc_id],
                )

        print(f"✓ {filename} indexado.")


if __name__ == "__main__":
    run_ingestion()
    print(f"\n--- Ingestão Concluída com Sucesso em {DB_PATH} ---")