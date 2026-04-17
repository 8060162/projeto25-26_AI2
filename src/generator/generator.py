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

ALTERAÇÕES (refactor):
  1. Introduzida GeneratorError — excepção própria que permite ao chamador
     distinguir entre uma resposta legítima e uma falha do pipeline.
     Os erros deixam de ser devolvidos como strings, o que impossibilitava
     retry lógico e logging estruturado.

  2. Registry de backends (_BACKENDS) substituiu o if/elif em _get_cliente().
     Adicionar um novo backend requer apenas registar o módulo no dicionário,
     sem tocar na lógica de orquestração (princípio Open/Closed).

  3. gerar_resposta() propaga GeneratorError — o ponto de entrada (CLI)
     trata o erro de forma adequada ao contexto em vez de receber uma
     string de erro disfarçada de resposta.
"""

import importlib
import os
import sys
import warnings
from typing import Callable

warnings.filterwarnings("ignore", category=UserWarning)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR  = os.path.dirname(_THIS_DIR)
for _p in (_SRC_DIR, _THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from retriever.query import obter_contexto
from prompt_builder  import construir_prompt
from settings        import LLM_BACKEND, MODEL_DISPLAY_NAME


# ── Excepção própria ──────────────────────────────────────────────────────────

class GeneratorError(Exception):
    """
    Excepção levantada quando o pipeline RAG falha a gerar uma resposta.

    Distingue falhas do pipeline de respostas legítimas do modelo,
    permitindo ao chamador aplicar retry, logging estruturado ou
    tratamento diferenciado conforme o contexto (CLI, API, teste).
    """


# ── Registry de backends ──────────────────────────────────────────────────────

# Mapeamento backend → nome do módulo cliente.
# Para adicionar um novo backend: registar aqui sem tocar em _get_cliente().
_BACKENDS: dict[str, str] = {
    "openai": "openai_client",
    "ollama": "ollama_client",
}


def _get_cliente() -> Callable[[str, str], str]:
    """
    Devolve a função chamar_modelo do cliente configurado em LLM_BACKEND.

    Raises:
        GeneratorError: se LLM_BACKEND não corresponder a nenhum backend registado.
    """
    module_name = _BACKENDS.get(LLM_BACKEND)
    if module_name is None:
        raise GeneratorError(
            f"LLM_BACKEND inválido: '{LLM_BACKEND}'. "
            f"Valores válidos: {', '.join(repr(k) for k in _BACKENDS)}."
        )
    module = importlib.import_module(module_name)
    return module.chamar_modelo


# ── Pipeline RAG ──────────────────────────────────────────────────────────────

def gerar_resposta(pergunta: str) -> str:
    """
    Pipeline RAG completo: retrieval → prompt → generation.

    Args:
        pergunta: Questão do utilizador.

    Returns:
        Resposta fundamentada nos regulamentos.

    Raises:
        GeneratorError: se qualquer etapa do pipeline falhar.
                        O chamador decide como tratar o erro (log, retry, UI).
    """
    artigos = obter_contexto(pergunta)

    if not artigos:
        return "Não encontrei informações nos documentos para responder a essa questão."

    try:
        prompt_sistema, prompt_utilizador = construir_prompt(pergunta, artigos)
    except FileNotFoundError as e:
        raise GeneratorError(
            f"Erro de configuração: o template do prompt não foi encontrado. Detalhes: {e}"
        ) from e

    try:
        chamar_modelo = _get_cliente()
        return chamar_modelo(prompt_sistema, prompt_utilizador)
    except GeneratorError:
        raise
    except (RuntimeError, ValueError, EnvironmentError) as e:
        raise GeneratorError(f"Erro ao gerar resposta: {e}") from e


# ── CLI interactivo ───────────────────────────────────────────────────────────

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
        try:
            resposta = gerar_resposta(pergunta)
            print("\n" + "─" * 20 + " RESPOSTA " + "─" * 20)
            print(resposta)
        except GeneratorError as e:
            print(f"\n[ERRO] {e}")
        print("─" * 50)