"""
query.py
--------
Ponto de entrada público do módulo retriever.
É o único ficheiro que o generator importa.

Não contém lógica — apenas expõe uma interface estável
que isola o generator de mudanças internas ao retriever.
"""

import sys
import os
# Garante que src/ está no path quando o script é executado directamente
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retriever.search import procurar_contexto
from retriever.models import ArtigoContexto


def obter_contexto(
    pergunta: str,
    n_resultados: int | None = None,
) -> list[ArtigoContexto]:
    """
    Devolve artigos relevantes para a `pergunta`.

    Args:
        pergunta:     questão do utilizador.
        n_resultados: se None, usa o valor por omissão definido em settings.

    Returns:
        Lista de ArtigoContexto ordenada por relevância.
    """
    if n_resultados is not None:
        return procurar_contexto(pergunta, n_resultados)
    return procurar_contexto(pergunta)


if __name__ == "__main__":
    pergunta = input("Pergunta: ")
    resultados = obter_contexto(pergunta)

    print(f"\n{len(resultados)} artigo(s) encontrado(s) para: '{pergunta}'\n")

    for i, art in enumerate(resultados, 1):
        print(f"{'='*60}")
        print(f"{i}. {art.artigo_id} — {art.artigo_titulo or '(sem título)'}")
        print(f"   Fonte     : {art.source}")
        print(f"   Capítulo  : {art.capitulo_titulo or '—'}")
        print(f"   Página    : {art.pagina or '—'}")
        print(f"\n{art.conteudo}")
        print()