"""
settings.py
-----------
Ponto único de configuração do sistema.
Todos os módulos (retriever e datastore) importam daqui.

As variáveis de ambiente têm precedência sobre os valores por omissão,
o que permite sobrepor configuração em CI/staging/produção sem alterar código.
"""

import os

# ── Retrieval ────────────────────────────────────────────────────────────────

# Número de artigos devolvidos ao generator
N_RESULTS: int = int(os.getenv("RETRIEVER_N_RESULTS", 5))

# Número de candidatos pedidos ao ChromaDB antes de deduplicar
QUERY_FETCH: int = int(os.getenv("RETRIEVER_QUERY_FETCH", 15))

# ── Datastore ────────────────────────────────────────────────────────────────

# Directório raiz da base de dados ChromaDB
DB_PATH: str = os.getenv("DATASTORE_DB_PATH", "data/chromaDB")

# Nome da colecção ChromaDB
COLLECTION_NAME: str = os.getenv("DATASTORE_COLLECTION_NAME", "artigos_legislativos")

# Directório com os ficheiros JSON de origem
JSON_FOLDER: str = os.getenv("DATASTORE_JSON_FOLDER", "data/processed")