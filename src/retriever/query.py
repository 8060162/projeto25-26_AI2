"""
query.py
--------
Ponto de entrada público do módulo retriever.
É o único ficheiro que o generator importa.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retriever.search import procurar_contexto
from retriever.models    import ArtigoContexto


def obter_contexto(pergunta: str, n_resultados: int = None) -> list[ArtigoContexto]:
    if n_resultados is not None:
        return procurar_contexto(pergunta, n_resultados)
    return procurar_contexto(pergunta)


if __name__ == "__main__":
    pergunta = input("Pergunta: ")
    resultados = obter_contexto(pergunta)

    print(f"\n{len(resultados)} artigo(s) encontrado(s) para: '{pergunta}'\n")

    for i, art in enumerate(resultados, 1):
        print(f"{'='*60}")
        print(f"{i}. {art.artigo_id} — {art.artigo_titulo}")
        print(f"   Fonte     : {art.source}")
        print(f"   Capítulo  : {art.capitulo_titulo}")
        print(f"   Página    : {art.pagina}")
        print(f"\n{art.conteudo}")
        print()