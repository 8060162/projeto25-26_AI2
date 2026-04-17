"""
settings.py
-----------
Fonte única de verdade para toda a configuração do projecto RAG.

Cobre quatro domínios: LLM, DATASTORE, RETRIEVAL e CHUNKING.
Todos os módulos importam exclusivamente deste ficheiro.
As variáveis de ambiente têm precedência sobre os valores por omissão,
permitindo sobrepor configuração em CI/staging/produção sem alterar código.

Nota: credenciais sensíveis (API keys) são lidas via dotenv — ver .env.example.

ALTERAÇÃO (refactor): as credenciais do backend OpenAI deixaram de ser
validadas em tempo de importação. A validação ocorre em openai_client.py,
no momento em que o cliente é efectivamente instanciado. Desta forma,
importar settings em testes ou com LLM_BACKEND=ollama nunca lança
EnvironmentError desnecessário.
"""

import os
from dotenv import load_dotenv

load_dotenv()


# ── LLM ──────────────────────────────────────────────────────────────────────

# Valores válidos: "openai" | "ollama"
LLM_BACKEND: str = os.getenv("LLM_BACKEND", "openai")

# Credenciais lidas sem validação — a validação é feita em openai_client.py
IAEDU_ENDPOINT:   str = os.getenv("IAEDU_ENDPOINT",   "")
IAEDU_API_KEY:    str = os.getenv("IAEDU_API_KEY",    "")
IAEDU_CHANNEL_ID: str = os.getenv("IAEDU_CHANNEL_ID", "")
IAEDU_THREAD_ID:  str = os.getenv("IAEDU_THREAD_ID",  "")

OLLAMA_MODEL:       str = os.getenv("OLLAMA_MODEL",        "qwen2.5:7b")
MODEL_DISPLAY_NAME: str = os.getenv("MODEL_DISPLAY_NAME",  LLM_BACKEND)

# ── DATASTORE ─────────────────────────────────────────────────────────────────

DB_PATH:         str = os.getenv("DATASTORE_DB_PATH",         "data/chromaDB")
COLLECTION_NAME: str = os.getenv("DATASTORE_COLLECTION_NAME", "artigos_legislativos")
JSON_FOLDER:     str = os.getenv("DATASTORE_JSON_FOLDER",     "data/processed")

# Alias explícito usado por ingest.py (mesmo caminho que JSON_FOLDER)
INPUT_FOLDER: str = JSON_FOLDER

# ── RETRIEVAL ────────────────────────────────────────────────────────────────

# Número de artigos devolvidos ao generator
N_RESULTS:   int = int(os.getenv("RETRIEVER_N_RESULTS",   3))

# Número de candidatos pedidos ao ChromaDB antes de deduplicar
QUERY_FETCH: int = int(os.getenv("RETRIEVER_QUERY_FETCH", 15))

# ── CHUNKING ─────────────────────────────────────────────────────────────────

CHUNK_TARGET:  int = int(os.getenv("CHUNK_TARGET",        550))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP",        80))
HEADER_BUDGET: int = int(os.getenv("CHUNK_HEADER_BUDGET", 120))

# Limiar de divisão (~670 chars) — calculado, nunca hardcoded
CHUNK_MAX: int = CHUNK_TARGET + HEADER_BUDGET

# Separador entre cabeçalho contextual e corpo do chunk.
# Partilhado entre chunker.py (escrita) e search.py (leitura) —
# alterar aqui propaga-se automaticamente a ambos os lados.
CHUNK_HEADER_SEP: str = "\n\n"