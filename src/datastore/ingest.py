"""
ingest.py
---------
Responsabilidade única: orquestrar o pipeline de ingestão de documentos
JSON para ChromaDB.

Estrutura:
  - descobrir_ficheiros()    → I/O: lista os JSONs a processar
  - inicializar_colecao()    → infra: cria cliente ChromaDB + colecção
  - _safe_doc_id()           → utilitário: produz IDs seguros e estáveis
  - indexar_artigo()         → lógica: chunking + upsert de um artigo
  - run_ingestion()          → orquestrador: liga todas as peças
"""

import hashlib
import logging
import os
import warnings

import chromadb
import torch

warnings.filterwarnings("ignore", category=UserWarning)

from config import COLLECTION_NAME, DB_PATH, INPUT_FOLDER
from chunker import dividir_em_chunks
from document_parser import Artigo, parse_ficheiro
from embeddings import BGEM3EmbeddingFunction

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ── Utilitários ───────────────────────────────────────────────────────────────

def _safe_doc_id(filename: str, art_id: str) -> str:
    """
    Produz um ID seguro para o ChromaDB a partir do nome do ficheiro e do
    identificador do artigo.

    Regras:
      - Apenas caracteres alfanuméricos, '-', '_', '.' são mantidos.
      - Se o ID resultante exceder 512 caracteres, é truncado e sufixado
        com um hash MD5 (8 hex chars) para garantir unicidade.
    """
    base = f"{filename}__{art_id}"
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in base)

    if len(safe) > 512:
        digest = hashlib.md5(safe.encode()).hexdigest()[:8]
        safe   = f"{safe[:480]}_{digest}"

    return safe


# ── Inicialização de infra ────────────────────────────────────────────────────

def descobrir_ficheiros(folder: str) -> list[str]:
    """Devolve a lista de nomes de ficheiros JSON presentes em `folder`."""
    return [f for f in os.listdir(folder) if f.endswith(".json")]


def inicializar_colecao(
    db_path: str,
    collection_name: str,
    device: str,
) -> chromadb.Collection:
    """
    Cria (ou abre) o cliente ChromaDB persistente e devolve a colecção.

    Isola toda a lógica de inicialização de infra, tornando `run_ingestion`
    independente dos detalhes de configuração do ChromaDB.
    """
    os.makedirs(db_path, exist_ok=True)
    embedding_fn = BGEM3EmbeddingFunction(device=device)
    client       = chromadb.PersistentClient(path=db_path)
    return client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_fn,
    )


# ── Lógica de indexação ───────────────────────────────────────────────────────

def indexar_artigo(collection: chromadb.Collection, artigo: Artigo) -> None:
    """
    Gera os chunks de um artigo e faz upsert na colecção ChromaDB.

    Responsabilidade isolada: recebe um Artigo já parseado e uma colecção
    já inicializada — não sabe nada sobre ficheiros nem sobre o ChromaDB client.
    """
    if not artigo.conteudo.strip():
        return

    chunks       = dividir_em_chunks(**artigo.to_chunks_args())
    doc_id_base  = _safe_doc_id(artigo.filename, artigo.art_id)
    is_divided   = len(chunks) > 1

    for i, chunk_text in enumerate(chunks):
        metadata = artigo.to_metadata()

        if is_divided:
            # Sinaliza expansão no retriever e actualiza o flag truncated
            metadata["truncated"] = "true"
            metadata["part"]      = i

        collection.upsert(
            documents=[chunk_text],
            metadatas=[metadata],
            ids=[f"{doc_id_base}_p{i}" if is_divided else doc_id_base],
        )


# ── Orquestrador ──────────────────────────────────────────────────────────────

def run_ingestion() -> None:
    """
    Pipeline completo de ingestão:
      1. Valida a pasta de entrada.
      2. Inicializa o ChromaDB e o modelo de embeddings.
      3. Para cada ficheiro JSON: parse → indexação de cada artigo.
      4. Reporta ficheiros com erro no final.
    """
    if not os.path.exists(INPUT_FOLDER):
        logger.error("Pasta de entrada não encontrada: %s", INPUT_FOLDER)
        return

    json_files = descobrir_ficheiros(INPUT_FOLDER)
    if not json_files:
        logger.warning("Nenhum ficheiro JSON encontrado em: %s", INPUT_FOLDER)
        return

    device     = "mps" if torch.backends.mps.is_available() else "cpu"
    collection = inicializar_colecao(DB_PATH, COLLECTION_NAME, device)

    failed: list[str] = []

    for filename in json_files:
        filepath = os.path.join(INPUT_FOLDER, filename)
        try:
            artigos = parse_ficheiro(filepath, filename)
        except Exception:
            logger.error("Erro ao parsear '%s'", filename, exc_info=True)
            failed.append(filename)
            continue

        for artigo in artigos:
            indexar_artigo(collection, artigo)

        logger.info("✓ %s indexado com sucesso.", filename)

    # Relatório final de falhas
    if failed:
        logger.warning(
            "Ingestão concluída com %d erro(s). Ficheiros afectados: %s",
            len(failed),
            failed,
        )
    else:
        logger.info("Ingestão concluída sem erros.")


if __name__ == "__main__":
    run_ingestion()