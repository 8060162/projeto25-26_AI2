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
            f"[{art.artigo_id} — {art.artigo_titulo} | "
            f"Pág. {art.pagina} | {art.source}]\n"
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
    És um assistente universitário. Responde sempre em Português de Portugal.
    REGRAS:
    1. Responde de forma DIRECTA e CONCISA — máximo 3 frases por ponto.
    2. Não repitas informação que já disseste.
    3. Cita a fonte apenas uma vez no final, no formato curto:
       → Fonte: Artigo X, pág. N, Ficheiro
    4. Se a resposta for simples, responde numa só frase com a fonte no final.
    5. Responde APENAS com base no CONTEXTO. Se não souberes, diz que o regulamento não refere o assunto.
    """

    prompt_utilizador = f"CONTEXTO:\n{contexto}\n\nPERGUNTA: {pergunta}"

    return prompt_sistema, prompt_utilizador