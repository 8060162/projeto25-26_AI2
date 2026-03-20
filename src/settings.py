"""
settings.py
-----------
Fonte única de verdade para toda a configuração do projecto RAG.

Cobre quatro domínios: LLM, DATASTORE, RETRIEVAL e CHUNKING.
Todos os módulos importam exclusivamente deste ficheiro.
As variáveis de ambiente têm precedência sobre os valores por omissão,
permitindo sobrepor configuração em CI/staging/produção sem alterar código.

Nota: credenciais sensíveis (API keys) são lidas via dotenv — ver .env.example.
"""

import os
from dotenv import load_dotenv

load_dotenv()


def _require_env(key: str) -> str:
    """Lê uma variável de ambiente obrigatória ou lança erro claro."""
    value = os.getenv(key)
    if not value:
        raise EnvironmentError(
            f"Variável de ambiente obrigatória não definida: '{key}'. "
            f"Verifica o ficheiro .env ou o ambiente de execução."
        )
    return value


# ── LLM ──────────────────────────────────────────────────────────────────────

# Valores válidos: "openai" | "ollama"
LLM_BACKEND: str = os.getenv("LLM_BACKEND", "openai")

IAEDU_ENDPOINT:   str = _require_env("IAEDU_ENDPOINT")   if LLM_BACKEND == "openai" else ""
IAEDU_API_KEY:    str = _require_env("IAEDU_API_KEY")    if LLM_BACKEND == "openai" else ""
IAEDU_CHANNEL_ID: str = _require_env("IAEDU_CHANNEL_ID") if LLM_BACKEND == "openai" else ""
IAEDU_THREAD_ID:  str = _require_env("IAEDU_THREAD_ID")  if LLM_BACKEND == "openai" else ""

OLLAMA_MODEL: str       = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
MODEL_DISPLAY_NAME: str = os.getenv("MODEL_DISPLAY_NAME", LLM_BACKEND)

# ── DATASTORE ─────────────────────────────────────────────────────────────────

DB_PATH: str         = os.getenv("DATASTORE_DB_PATH",         "data/chromaDB")
COLLECTION_NAME: str = os.getenv("DATASTORE_COLLECTION_NAME", "artigos_legislativos")
JSON_FOLDER: str     = os.getenv("DATASTORE_JSON_FOLDER",     "data/processed")

# Alias explícito usado por ingest.py (mesmo caminho que JSON_FOLDER)
INPUT_FOLDER: str = JSON_FOLDER

# ── RETRIEVAL ────────────────────────────────────────────────────────────────

# Número de artigos devolvidos ao generator
N_RESULTS: int   = int(os.getenv("RETRIEVER_N_RESULTS",    5))

# Número de candidatos pedidos ao ChromaDB antes de deduplicar
QUERY_FETCH: int = int(os.getenv("RETRIEVER_QUERY_FETCH", 15))

# ── CHUNKING ─────────────────────────────────────────────────────────────────

CHUNK_TARGET: int  = int(os.getenv("CHUNK_TARGET",  550))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP",  80))
HEADER_BUDGET: int = int(os.getenv("CHUNK_HEADER_BUDGET", 120))

# Limiar de divisão (~670 chars) — calculado, nunca hardcoded
CHUNK_MAX: int = CHUNK_TARGET + HEADER_BUDGET

# Separador entre cabeçalho contextual e corpo do chunk.
# Partilhado entre chunker.py (escrita) e search.py (leitura) —
# alterar aqui propaga-se automaticamente a ambos os lados.
CHUNK_HEADER_SEP: str = "\n\n"