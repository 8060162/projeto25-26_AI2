"""
generator.py
------------
Responsabilidade única: orquestrar o pipeline RAG completo.

Fluxo:
    1. Retrieval  — obter_contexto()   → list[ArtigoContexto]
    2. Prompt     — construir_prompt() → (sistema, utilizador)
    3. Generation — chamar_modelo()    → resposta final
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retriever.query  import obter_contexto
from prompt_builder   import construir_prompt

# Trocar o import para mudar de modelo — o resto do código não muda
# from ollama_client import chamar_modelo, MODEL_NAME
from openai_client import chamar_modelo, MODEL_NAME


def gerar_resposta(pergunta: str) -> str:
    """
    Pipeline RAG completo: retrieval → prompt → generation.

    Args:
        pergunta: questão do utilizador.

    Returns:
        Resposta fundamentada nos regulamentos.
    """
    artigos = obter_contexto(pergunta)

    if not artigos:
        return "Não encontrei informações nos documentos para responder a essa questão."

    prompt_sistema, prompt_utilizador = construir_prompt(pergunta, artigos)

    return chamar_modelo(prompt_sistema, prompt_utilizador)


if __name__ == "__main__":
    print("\n" + "═" * 60)
    print(f"  RAG ATIVO ({MODEL_NAME})")
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