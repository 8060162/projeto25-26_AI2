"""
ingest.py
---------
Pipeline de ingestão: lê os JSONs processados, divide em chunks e indexa
no ChromaDB com metadados completos de rastreabilidade.

Metadados por chunk
───────────────────
Campo               Descrição
─────────────────── ──────────────────────────────────────────────────────────
source              nome do ficheiro JSON de origem
doc_titulo          título oficial do documento
doc_numero          número/referência do despacho ou regulamento
doc_data            data de publicação
capitulo_id         chave interna do capítulo no JSON
capitulo_titulo     título legível do capítulo
artigo_id           identificador do artigo (ex.: "Artigo 5.º")
artigo_titulo       título do artigo
pagina              página no documento original
chunk_index         índice deste chunk dentro do artigo (0-based)
chunk_total         total de chunks do artigo
truncated           "true" / "false" — se True, o artigo não cabe num chunk;
                    para recuperar o texto completo, ler `source` + `artigo_id`
                    directamente do JSON.
"""

import os
import warnings
import chromadb
import torch

warnings.filterwarnings("ignore", category=UserWarning)

# ── Caminhos ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT_FOLDER = os.path.join(BASE_DIR, "data", "processed")
DB_PATH      = os.path.join(BASE_DIR, "data", "chromaDB")
COLLECTION_NAME = "artigos_legislativos"

os.makedirs(DB_PATH, exist_ok=True)

from embeddings       import BGEM3EmbeddingFunction
from document_parser  import parse_ficheiro
from chunker          import dividir_conteudo


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_doc_id(filename: str, art_id: str) -> str:
    """Gera um ID limpo para o ChromaDB (sem espaços nem caracteres especiais)."""
    base = f"{filename}__{art_id}"
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in base)


def _build_metadata(artigo, chunk_index: int, chunk_total: int, truncated: bool) -> dict:
    """Constrói o dicionário de metadados para um chunk."""
    return {
        # rastreabilidade do documento
        "source":          artigo.filename,
        "doc_titulo":      artigo.doc_titulo,
        "doc_numero":      artigo.doc_numero,
        "doc_data":        artigo.doc_data,
        # localização dentro do documento
        "capitulo_id":     artigo.cap_id,
        "capitulo_titulo": artigo.cap_titulo,
        "artigo_id":       artigo.art_id,
        "artigo_titulo":   artigo.art_titulo,
        "pagina":          artigo.pagina,
        # informação do chunk
        "chunk_index":     chunk_index,
        "chunk_total":     chunk_total,
        "truncated":       str(truncated).lower(),   # ChromaDB não suporta bool nativamente
    }


# ── Pipeline principal ────────────────────────────────────────────────────────

def run_ingestion():
    if not os.path.exists(INPUT_FOLDER):
        print(f"ERRO: Pasta de entrada não encontrada: {INPUT_FOLDER}")
        return

    json_files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(".json")]
    if not json_files:
        print(f"Nenhum ficheiro JSON encontrado em: {INPUT_FOLDER}")
        return

    # Inicializar modelo e base de dados uma única vez
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Dispositivo de embedding: {device}")

    embedding_fn = BGEM3EmbeddingFunction(device=device)
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
    )

    print(f"A processar {len(json_files)} ficheiro(s)...\n")

    total_chunks = 0
    total_truncated = 0

    for filename in json_files:
        filepath = os.path.join(INPUT_FOLDER, filename)
        try:
            artigos = parse_ficheiro(filepath, filename)
        except Exception as e:
            print(f"  ✗ Erro ao ler {filename}: {e}")
            continue

        file_chunks = 0
        file_truncated = 0

        for artigo in artigos:
            if not artigo.conteudo.strip():
                continue  # artigos sem conteúdo são ignorados

            chunks, truncated = dividir_conteudo(artigo.conteudo)
            chunk_total = len(chunks)
            doc_id_base = _safe_doc_id(filename, artigo.art_id)

            if truncated:
                file_truncated += 1

            documents, metadatas, ids = [], [], []

            for i, chunk_text in enumerate(chunks):
                documents.append(chunk_text)
                metadatas.append(_build_metadata(artigo, i, chunk_total, truncated))
                ids.append(f"{doc_id_base}_part{i + 1}" if chunk_total > 1 else doc_id_base)

            collection.upsert(documents=documents, metadatas=metadatas, ids=ids)
            file_chunks += chunk_total

        total_chunks    += file_chunks
        total_truncated += file_truncated

        trunc_info = f"  ({file_truncated} artigo(s) truncado(s))" if file_truncated else ""
        print(f"  ✓ {filename}  →  {len(artigos)} artigos  /  {file_chunks} chunks{trunc_info}")

    print(f"\n{'─'*60}")
    print(f"Ingestão concluída.")
    print(f"  Total de chunks indexados : {total_chunks}")
    print(f"  Artigos com truncated=true: {total_truncated}")
    print(f"  Base de dados em          : {DB_PATH}")


if __name__ == "__main__":
    run_ingestion()