"""
prompt_builder.py
-----------------
Responsabilidade única: construir o prompt a enviar ao modelo.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retriever.models import ArtigoContexto


def formatar_contexto(artigos: list[ArtigoContexto]) -> str:
    """Formata os artigos recuperados numa string de contexto para o prompt."""
    blocos = []
    for art in artigos:
        bloco = (
            f"[FONTE: {art.source} | "
            f"{art.artigo_id} — {art.artigo_titulo} | "
            f"Pág. {art.pagina}]\n"
            f"{art.conteudo}"
        )
        blocos.append(bloco)
    return "\n\n---\n\n".join(blocos)


def construir_prompt(pergunta: str, artigos: list[ArtigoContexto]) -> tuple[str, str]:
    """
    Devolve o par (prompt_sistema, prompt_utilizador) prontos a enviar ao modelo.
    """
    contexto = formatar_contexto(artigos)

    prompt_sistema = """
    És um assistente virtual universitário. Responde sempre em Português de Portugal.
    REGRAS:
    1. Responde APENAS com base no CONTEXTO fornecido.
    2. Cita SEMPRE a fonte no formato: (Artigo X — Título, pág. N, Ficheiro).
    3. No final da resposta inclui uma secção "Fontes consultadas:" com a lista
       de todos os artigos utilizados.
    4. Se não souberes, diz que o regulamento não refere o assunto.
    """

    prompt_utilizador = f"CONTEXTO:\n{contexto}\n\nPERGUNTA: {pergunta}"

    return prompt_sistema, prompt_utilizador