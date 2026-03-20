"""
generator.py
------------
Responsabilidade única: orquestrar o pipeline RAG completo.

Fluxo:
    1. Retrieval  — obter_contexto()   → list[ArtigoContexto]
    2. Prompt     — construir_prompt() → (sistema, utilizador)
    3. Generation — chamar_modelo()    → resposta final

O cliente LLM activo é seleccionado em runtime via variável de ambiente
LLM_BACKEND, sem necessidade de editar este ficheiro.
"""

import os
import sys
import warnings
from typing import Callable

warnings.filterwarnings("ignore", category=UserWarning)

# Adiciona src/ ao path para que todos os pacotes sejam encontrados
# independentemente do directório de trabalho no momento de execução.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))   # src/generator/
_SRC_DIR  = os.path.dirname(_THIS_DIR)                   # src/
for _p in (_SRC_DIR, _THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from retriever.query import obter_contexto
from prompt_builder  import construir_prompt
from settings        import LLM_BACKEND, MODEL_DISPLAY_NAME


def _get_cliente() -> Callable[[str, str], str]:
    """Devolve a função chamar_modelo do cliente configurado em LLM_BACKEND."""
    if LLM_BACKEND == "ollama":
        from ollama_client import chamar_modelo
    elif LLM_BACKEND == "openai":
        from openai_client import chamar_modelo
    else:
        raise ValueError(
            f"LLM_BACKEND inválido: '{LLM_BACKEND}'. Valores válidos: 'openai', 'ollama'."
        )
    return chamar_modelo


def gerar_resposta(pergunta: str) -> str:
    """
    Pipeline RAG completo: retrieval → prompt → generation.

    Args:
        pergunta: Questão do utilizador.

    Returns:
        Resposta fundamentada nos regulamentos, ou mensagem de erro controlada.
    """
    artigos = obter_contexto(pergunta)

    if not artigos:
        return "Não encontrei informações nos documentos para responder a essa questão."

    try:
        prompt_sistema, prompt_utilizador = construir_prompt(pergunta, artigos)
    except FileNotFoundError as e:
        return (
            f"Erro de configuração: o template do prompt não foi encontrado. "
            f"Detalhes: {e}"
        )

    try:
        chamar_modelo = _get_cliente()
        return chamar_modelo(prompt_sistema, prompt_utilizador)
    except (RuntimeError, ValueError) as e:
        return f"Erro ao gerar resposta: {e}"


if __name__ == "__main__":
    print("\n" + "═" * 60)
    print(f"  RAG ACTIVO  |  backend: {LLM_BACKEND}  |  modelo: {MODEL_DISPLAY_NAME}")
    print("═" * 60)

    while True:
        pergunta = input("\nQuestão (ou 'sair'): ").strip()
        if pergunta.lower() in ["sair", "q"]:
            break
        if not pergunta:
            continue

        print("\n[A processar...]")
        resposta = gerar_resposta(pergunta)
        print("\n" + "─" * 20 + " RESPOSTA " + "─" * 20)
        print(resposta)
        print("─" * 50)