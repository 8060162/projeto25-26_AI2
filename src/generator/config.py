"""
config.py
---------
Fonte única de verdade para toda a configuração do sistema RAG.
Carrega variáveis de ambiente com fallback explícito para desenvolvimento.
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


# --- Backend LLM activo ---
# Valores válidos: "openai" | "ollama"
LLM_BACKEND: str = os.getenv("LLM_BACKEND", "openai")

# --- Configuração iaedu (openai_client) ---
IAEDU_ENDPOINT:   str = _require_env("IAEDU_ENDPOINT")   if LLM_BACKEND == "openai" else ""
IAEDU_API_KEY:    str = _require_env("IAEDU_API_KEY")    if LLM_BACKEND == "openai" else ""
IAEDU_CHANNEL_ID: str = _require_env("IAEDU_CHANNEL_ID") if LLM_BACKEND == "openai" else ""
IAEDU_THREAD_ID:  str = _require_env("IAEDU_THREAD_ID")  if LLM_BACKEND == "openai" else ""

# --- Configuração Ollama ---
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")

# --- Nome do modelo para apresentação ---
MODEL_DISPLAY_NAME: str = os.getenv("MODEL_DISPLAY_NAME", LLM_BACKEND)