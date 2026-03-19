"""
config.py
---------
Única fonte de verdade para todas as constantes e caminhos do projecto.

REGRA: nenhum outro módulo deve definir estas constantes localmente.
       Qualquer ajuste deve ser feito exclusivamente aqui.
"""

import os

# ── Caminhos ──────────────────────────────────────────────────────────────────

BASE_DIR        = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_PATH         = os.path.join(BASE_DIR, "data", "chromaDB")
JSON_FOLDER     = os.path.join(BASE_DIR, "data", "processed")
COLLECTION_NAME = "artigos_legislativos"

# Alias explícito usado por ingest.py (mesmo caminho que JSON_FOLDER)
INPUT_FOLDER = JSON_FOLDER

# ── Chunking — alinhado com a estratégia BGE-M3 ───────────────────────────────

CHUNK_TARGET  = 550   # Alvo de caracteres de conteúdo puro
CHUNK_OVERLAP = 80    # Overlap para o fallback RecursiveCharacterTextSplitter
HEADER_BUDGET = 120   # Reserva para o cabeçalho contextual (DOC + CAP + ART)
CHUNK_MAX     = CHUNK_TARGET + HEADER_BUDGET  # Limiar de divisão (~670 chars)
                                               # Calculado, nunca hardcoded