"""
report_formatter.py
-------------------
Responsabilidade única: formatar e imprimir os resultados das métricas
calculadas em metrics.py.

SEPARAÇÃO (refactor de evaluate.py):
  Este módulo contém exclusivamente lógica de apresentação — recebe DTOs
  de metrics.py e produz output legível. Não calcula nada, não faz I/O
  de ficheiros.

  Cada função de impressão recebe valores já calculados (DTOs ou scalars),
  não DataFrames raw. Isto torna possível:
    - Obter valores calculados sem executar a impressão (ex: logging JSON)
    - Testar a formatação de forma isolada com dados sintéticos
    - Redirigir o output (ex: para ficheiro) sem alterar a lógica
"""

import math
from typing import Optional

import numpy as np

from analysis.metrics import (
    ResumoCorpus,
    CoesaoRegulamento,
    DistanciaRegulamento,
    CoesaoArtigo,
    DistanciaArtigo,
)

# ── Constantes de layout ──────────────────────────────────────────────────────

_LARGURA_NOME  = 52
_LARGURA_TOTAL = 72


# ── Utilitários de formatação ─────────────────────────────────────────────────

def _truncar(texto: str, max_len: int = _LARGURA_NOME) -> str:
    return texto if len(texto) <= max_len else texto[:max_len - 1] + "…"


def _barra_visual(valor: float, minimo: float, maximo: float, largura: int = 20) -> str:
    if maximo == minimo:
        preenchimento = largura
    else:
        preenchimento = int((valor - minimo) / (maximo - minimo) * largura)
    return "█" * preenchimento + "░" * (largura - preenchimento)


def _titulo_secao(texto: str) -> None:
    print(f"\n{'═' * _LARGURA_TOTAL}")
    print(f"  {texto}")
    print(f"{'═' * _LARGURA_TOTAL}")


def _subtitulo(texto: str) -> None:
    print(f"\n  ┌─ {texto}")
    print(f"  └{'─' * (_LARGURA_TOTAL - 4)}")


# ── Resumo do corpus ──────────────────────────────────────────────────────────

def imprimir_resumo_corpus(resumo: ResumoCorpus) -> None:
    print(f"\n  {'Documentos indexados':<35} {resumo.n_documentos:>5}")
    print(f"  {'Artigos totais':<35} {resumo.n_artigos:>5}")
    print(f"  {'Artigos com 1 chunk (não divididos)':<35} {resumo.artigos_simples:>5}")
    print(f"  {'Artigos com 2+ chunks (divididos)':<35} {resumo.artigos_divididos:>5}")
    print(f"  {'Total de chunks':<35} {resumo.n_chunks:>5}")
    if resumo.n_artigos > 0:
        print(f"  {'Média de chunks por artigo':<35} {resumo.media_chunks:>5.2f}")


# ── NÍVEL 1 ───────────────────────────────────────────────────────────────────

def imprimir_silhouette_regulamento(score: float) -> None:
    barra = _barra_visual(max(score, 0), 0, 1, largura=20)

    if math.isnan(score):
        interpretacao = "indeterminado — dados insuficientes"
    elif score >= 0.15:
        interpretacao = "boa separação entre regulamentos ✓"
    elif score >= 0.05:
        interpretacao = "sobreposição esperada — regulamentos complementares do P.PORTO"
    else:
        interpretacao = "sobreposição elevada — esperado em legislação do mesmo domínio"

    print(f"\n  {'Silhouette Score (por regulamento)':<34} {score:>6.4f}  [{barra}]  escala: -1 → +1")
    print(f"  {'Interpretação':<34} {interpretacao}")
    print(f"\n  O que significa:")
    print(f"    +1.0  →  artigos do mesmo regulamento muito próximos e bem separados dos restantes")
    print(f"     0.0  →  sobreposição entre regulamentos (normal em legislação académica)")
    print(f"    -1.0  →  artigos mais próximos de outro regulamento do que do seu")
    print(f"\n  Nota: cada artigo é representado pelo centroide dos seus chunks.")


def imprimir_coesao_regulamento(resultados: list[CoesaoRegulamento]) -> None:
    if not resultados:
        print("\n  Sem dados suficientes.")
        return

    validos = [r.coesao_media for r in resultados if not math.isnan(r.coesao_media)]
    if not validos:
        print("\n  Sem dados suficientes.")
        return

    minimo  = min(validos)
    maximo  = max(validos)
    media   = sum(validos) / len(validos)
    mediana = sorted(validos)[len(validos) // 2]

    print(f"\n  Regulamentos avaliados : {len(resultados)}")
    print(f"  Média de coesão        : {media:.4f}")
    print(f"  Mediana                : {mediana:.4f}")
    print()

    col = _LARGURA_NOME
    print(f"  {'Regulamento':<{col}}  {'Artigos':>7}  {'Coesão':>6}  Barra")
    print(f"  {'─'*col}  {'───────':>7}  {'──────':>6}  {'─'*20}")

    for r in resultados:
        doc = _truncar(r.doc_titulo, col)
        if math.isnan(r.coesao_media):
            print(f"  {doc:<{col}}  {r.n_artigos:>7}     n/a")
            continue
        barra = _barra_visual(r.coesao_media, minimo, maximo)
        print(f"  {doc:<{col}}  {r.n_artigos:>7}  {r.coesao_media:>6.4f}  {barra}")

    print(f"\n  Leitura: barra mais longa = artigos do regulamento mais homogéneos entre si.")
    print(f"  Coesão = similaridade coseno média entre centroides de artigos do mesmo regulamento.")
    print(f"  Valores > 0.5 são normais em regulamentos académicos do mesmo instituto.")


def imprimir_distancia_regulamento(resultados: list[DistanciaRegulamento]) -> None:
    if not resultados:
        print("\n  Sem dados suficientes.")
        return

    valores = [r.distancia_coseno for r in resultados]
    minimo  = min(valores)
    maximo  = max(valores)

    col = 34
    print(f"\n  {'Regulamento A':<{col}}  {'Regulamento B':<{col}}  {'Dist.':>5}  Barra")
    print(f"  {'─'*col}  {'─'*col}  {'─────':>5}  {'─'*20}")

    for r in resultados:
        a     = _truncar(r.doc_a, col)
        b     = _truncar(r.doc_b, col)
        barra = _barra_visual(r.distancia_coseno, minimo, maximo)
        print(f"  {a:<{col}}  {b:<{col}}  {r.distancia_coseno:>5.4f}  {barra}")

    print(f"\n  Leitura: barra mais longa = regulamentos semanticamente mais afastados.")
    print(f"  Centroide do regulamento = média dos centroides dos seus artigos.")
    print(f"  Distância coseno = 1 − similaridade. Próximo de 0 = regulamentos muito semelhantes.")


# ── NÍVEL 2 ───────────────────────────────────────────────────────────────────

def imprimir_silhouette_artigo(score: float) -> None:
    barra = _barra_visual(max(score, 0), 0, 1, largura=20)

    if math.isnan(score):
        interpretacao = "indeterminado — artigos com chunk único não avaliáveis"
    elif score >= 0.5:
        interpretacao = "chunks muito coesos dentro de cada artigo ✓"
    elif score >= 0.2:
        interpretacao = "coesão razoável — alguma sobreposição entre artigos próximos"
    elif score >= 0.0:
        interpretacao = "sobreposição moderada — esperado em regulamentos do mesmo domínio"
    else:
        interpretacao = "chunks semanticamente mais próximos de outros artigos"

    print(f"\n  {'Silhouette Score (por artigo)':<34} {score:>6.4f}  [{barra}]  escala: -1 → +1")
    print(f"  {'Interpretação':<34} {interpretacao}")
    print(f"\n  O que significa:")
    print(f"    +1.0  →  chunks do mesmo artigo muito coesos e separados de outros artigos")
    print(f"     0.0  →  chunks na fronteira semântica entre artigos")
    print(f"    -1.0  →  chunks mais próximos de outro artigo do que do seu próprio")
    print(f"\n  Nota: avalia apenas artigos com 2+ chunks (truncated=true).")


def imprimir_coesao_artigo(resultados: list[CoesaoArtigo]) -> None:
    if not resultados:
        print("\n  Nenhum artigo dividido em múltiplos chunks — métrica não aplicável.")
        return

    valores = [r.coesao_media for r in resultados]
    minimo  = min(valores)
    maximo  = max(valores)
    media   = sum(valores) / len(valores)
    mediana = sorted(valores)[len(valores) // 2]

    print(f"\n  Artigos avaliados : {len(resultados)}")
    print(f"  Média de coesão   : {media:.4f}")
    print(f"  Mediana           : {mediana:.4f}")
    print(f"  Mínimo / Máximo   : {minimo:.4f} / {maximo:.4f}")
    print()

    col_doc = 28
    col_art = 20
    print(f"  {'Regulamento':<{col_doc}}  {'Artigo':<{col_art}}  {'Chunks':>6}  {'Coesão':>6}  Barra")
    print(f"  {'─'*col_doc}  {'─'*col_art}  {'──────':>6}  {'──────':>6}  {'─'*20}")

    for r in resultados:
        doc   = _truncar(r.doc_titulo, col_doc)
        art   = _truncar(r.artigo_id,  col_art)
        barra = _barra_visual(r.coesao_media, minimo, maximo)
        print(f"  {doc:<{col_doc}}  {art:<{col_art}}  {r.n_chunks:>6}  {r.coesao_media:>6.4f}  {barra}")

    print(f"\n  Leitura: barra mais longa = chunks do artigo mais coesos entre si.")
    print(f"  Coesão = similaridade coseno média entre chunks do mesmo artigo.")
    print(f"  Valores > 0.7 indicam que os chunks representam bem o mesmo conteúdo.")


def imprimir_distancia_artigo(resultados: list[DistanciaArtigo], top_n: int = 15) -> None:
    if not resultados:
        print("\n  Sem dados suficientes.")
        return

    top     = resultados[:top_n]
    valores = [r.distancia_coseno for r in resultados]
    minimo  = min(valores)
    maximo  = max(valores)

    col = 22
    print(f"\n  (a mostrar os {len(top)} pares de artigos mais distantes por regulamento)\n")
    print(f"  {'Regulamento':<{col}}  {'Artigo A':<{col}}  {'Artigo B':<{col}}  {'Dist.':>5}  Barra")
    print(f"  {'─'*col}  {'─'*col}  {'─'*col}  {'─────':>5}  {'─'*20}")

    for r in top:
        doc   = _truncar(r.doc_titulo, col)
        art_a = _truncar(r.artigo_a,   col)
        art_b = _truncar(r.artigo_b,   col)
        barra = _barra_visual(r.distancia_coseno, minimo, maximo)
        print(f"  {doc:<{col}}  {art_a:<{col}}  {art_b:<{col}}  {r.distancia_coseno:>5.4f}  {barra}")

    print(f"\n  Leitura: barra mais longa = artigos semanticamente mais afastados.")
    print(f"  Baseado em centroides de artigos — inclui artigos com chunk único.")
    print(f"  Distância coseno = 1 − similaridade. Próximo de 0 = artigos muito semelhantes.")