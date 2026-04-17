"""
metrics.py
----------
Responsabilidade única: calcular métricas quantitativas de qualidade
dos embeddings. Funções puras — sem I/O, sem formatação, sem impressão.

SEPARAÇÃO (refactor de evaluate.py):
  Este módulo contém exclusivamente lógica de cálculo.
  A formatação e impressão estão em report_formatter.py.
  A orquestração (I/O + sequência) está em evaluate.py.

Esta separação torna as métricas testáveis de forma completamente isolada:
    from analysis.metrics import calcular_silhouette_regulamento
    score = calcular_silhouette_regulamento(df)
    assert score > 0.05

ARQUITECTURA DE AVALIAÇÃO
─────────────────────────
  Corpus
    └── Regulamento A  ← cluster de nível 1
    │     └── Artigo 1  → centroide = média dos embeddings dos seus chunks
    │     └── Artigo 2  → centroide = média dos embeddings dos seus chunks
    └── Regulamento B  ← cluster de nível 1

PRINCÍPIO FUNDAMENTAL:
  A unidade atómica de conteúdo é o ARTIGO, não o chunk.
  Toda a avaliação ao nível do regulamento opera sobre centroides
  de artigos — nunca sobre chunks directos.
"""

import logging
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd

from shared.metadata_keys import is_truncated

logger = logging.getLogger(__name__)


# ── DTOs de resultado ─────────────────────────────────────────────────────────

@dataclass
class CoesaoRegulamento:
    doc_titulo:   str
    n_artigos:    int
    coesao_media: float  # nan se n_artigos < 2


@dataclass
class DistanciaRegulamento:
    doc_a:            str
    doc_b:            str
    distancia_coseno: float


@dataclass
class CoesaoArtigo:
    artigo_key:   str
    doc_titulo:   str
    artigo_id:    str
    n_chunks:     int
    coesao_media: float


@dataclass
class DistanciaArtigo:
    doc_titulo:       str
    artigo_a:         str
    artigo_b:         str
    distancia_coseno: float


@dataclass
class ResumoCorpus:
    n_documentos:      int
    n_artigos:         int
    artigos_simples:   int
    artigos_divididos: int
    n_chunks:          int
    media_chunks:      float


# ── Preparação do DataFrame ───────────────────────────────────────────────────

def adicionar_chave_artigo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adiciona a coluna 'artigo_key' = doc_titulo + ' § ' + artigo_id.
    Esta chave identifica univocamente um artigo no corpus inteiro.
    Necessária porque artigo_id (ex: 'ART_1') repete-se entre documentos.
    """
    df = df.copy()
    df["artigo_key"] = df["doc_titulo"] + " § " + df["artigo_id"]
    return df


def calcular_centroides_artigos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Single Source of Truth para centroides de artigos.

    Devolve um DataFrame com UMA linha por artigo:
        artigo_key | doc_titulo | artigo_id | n_chunks | centroide

    O centroide é a média dos embeddings de todos os chunks do artigo.
    Artigos com chunk único têm centroide igual ao embedding desse chunk.

    Todas as métricas de nível 1 e a distância inter-artigo do nível 2
    devem usar esta função — nunca recalcular centroides localmente.
    """
    resultados = []
    for artigo_key, subset in df.groupby("artigo_key"):
        embeddings = np.stack(subset["embedding"].values)
        resultados.append({
            "artigo_key": artigo_key,
            "doc_titulo": subset["doc_titulo"].iloc[0],
            "artigo_id":  subset["artigo_id"].iloc[0],
            "n_chunks":   len(embeddings),
            "centroide":  np.mean(embeddings, axis=0),
        })
    return pd.DataFrame(resultados).reset_index(drop=True)


# ── Resumo do corpus ──────────────────────────────────────────────────────────

def resumo_corpus(df: pd.DataFrame) -> ResumoCorpus:
    """Calcula estatísticas gerais do corpus. Devolve DTO, não imprime."""
    n_chunks          = len(df)
    n_artigos         = df["artigo_key"].nunique()
    n_docs            = df["doc_titulo"].nunique()
    # Usa is_truncated centralizado — robusto a bool e string
    artigos_divididos = df[df["truncated"].apply(
        lambda v: is_truncated({"truncated": v})
    )]["artigo_key"].nunique()
    artigos_simples   = n_artigos - artigos_divididos
    media             = n_chunks / n_artigos if n_artigos > 0 else 0.0

    return ResumoCorpus(
        n_documentos=n_docs,
        n_artigos=n_artigos,
        artigos_simples=artigos_simples,
        artigos_divididos=artigos_divididos,
        n_chunks=n_chunks,
        media_chunks=round(media, 2),
    )


# ── NÍVEL 1 — Métricas por Regulamento ───────────────────────────────────────

def calcular_silhouette_regulamento(df: pd.DataFrame) -> float:
    """
    Silhouette Score ao nível do regulamento.
    Unidade de análise: centroide de cada artigo (1 vector por artigo).
    Intervalo [-1, 1]. Devolve nan se dados insuficientes.
    """
    from sklearn.metrics import silhouette_score

    centroides_df = calcular_centroides_artigos(df)
    n_docs = centroides_df["doc_titulo"].nunique()

    if n_docs < 2:
        logger.warning("Silhouette (regulamento) requer 2+ regulamentos. Encontrados: %d.", n_docs)
        return float("nan")

    contagens    = centroides_df.groupby("doc_titulo").size()
    docs_validos = contagens[contagens >= 2].index
    centroides_df = centroides_df[centroides_df["doc_titulo"].isin(docs_validos)]

    if centroides_df["doc_titulo"].nunique() < 2:
        logger.warning("Silhouette (regulamento): regulamentos insuficientes com 2+ artigos.")
        return float("nan")

    embeddings = np.stack(centroides_df["centroide"].values)
    labels     = centroides_df["doc_titulo"].values
    return round(float(silhouette_score(embeddings, labels, metric="cosine")), 4)


def calcular_coesao_intra_regulamento(df: pd.DataFrame) -> list[CoesaoRegulamento]:
    """
    Coesão intra-regulamento.
    Para cada regulamento, calcula a similaridade coseno média entre
    todos os pares de centroides de artigos do mesmo regulamento.
    Devolve lista de DTOs ordenada por coesao_media descendente.
    """
    from sklearn.metrics.pairwise import cosine_similarity

    centroides_df = calcular_centroides_artigos(df)
    resultados: list[CoesaoRegulamento] = []

    for doc, subset in centroides_df.groupby("doc_titulo"):
        embeddings = np.stack(subset["centroide"].values)
        n = len(embeddings)

        if n < 2:
            resultados.append(CoesaoRegulamento(
                doc_titulo=str(doc), n_artigos=n, coesao_media=float("nan")
            ))
            continue

        matriz  = cosine_similarity(embeddings)
        mascara = ~np.eye(n, dtype=bool)
        coesao  = round(float(matriz[mascara].mean()), 4)
        resultados.append(CoesaoRegulamento(
            doc_titulo=str(doc), n_artigos=n, coesao_media=coesao
        ))

    return sorted(
        resultados,
        key=lambda r: r.coesao_media if not np.isnan(r.coesao_media) else -1,
        reverse=True,
    )


def calcular_distancia_inter_regulamento(df: pd.DataFrame) -> list[DistanciaRegulamento]:
    """
    Distância coseno entre todos os pares de centroides de regulamentos.
    Devolve lista de DTOs ordenada por distancia_coseno descendente.
    """
    from sklearn.metrics.pairwise import cosine_similarity

    centroides_df  = calcular_centroides_artigos(df)
    centroides_reg = (
        centroides_df.groupby("doc_titulo")["centroide"]
        .apply(lambda vecs: np.mean(np.stack(vecs.values), axis=0))
    )

    resultados: list[DistanciaRegulamento] = []
    for (doc_a, c_a), (doc_b, c_b) in combinations(centroides_reg.items(), 2):
        sim = cosine_similarity([c_a], [c_b])[0][0]
        resultados.append(DistanciaRegulamento(
            doc_a=str(doc_a),
            doc_b=str(doc_b),
            distancia_coseno=round(float(1 - sim), 4),
        ))

    return sorted(resultados, key=lambda r: r.distancia_coseno, reverse=True)


# ── NÍVEL 2 — Métricas por Artigo ────────────────────────────────────────────

def calcular_silhouette_artigo(df: pd.DataFrame) -> float:
    """
    Silhouette Score ao nível do artigo.
    Unidade de análise: chunks individuais de artigos com 2+ chunks.
    Devolve nan se dados insuficientes.
    """
    from sklearn.metrics import silhouette_score

    artigos_divididos = df.groupby("artigo_key").filter(lambda x: len(x) >= 2)
    n_artigos = artigos_divididos["artigo_key"].nunique()

    if n_artigos < 2:
        logger.warning(
            "Silhouette (artigo) requer 2+ artigos com 2+ chunks. Encontrados: %d.", n_artigos
        )
        return float("nan")

    embeddings = np.stack(artigos_divididos["embedding"].values)
    labels     = artigos_divididos["artigo_key"].values
    return round(float(silhouette_score(embeddings, labels, metric="cosine")), 4)


def calcular_coesao_intra_artigo(df: pd.DataFrame) -> list[CoesaoArtigo]:
    """
    Coesão intra-artigo para artigos com 2+ chunks.
    Devolve lista de DTOs ordenada por coesao_media descendente.
    """
    from sklearn.metrics.pairwise import cosine_similarity

    artigos_divididos = df[df["truncated"].apply(
        lambda v: is_truncated({"truncated": v})
    )]

    if artigos_divididos.empty:
        logger.warning("Nenhum artigo com 2+ chunks (truncated=true). Métrica não aplicável.")
        return []

    resultados: list[CoesaoArtigo] = []
    for artigo_key, subset in artigos_divididos.groupby("artigo_key"):
        embeddings = np.stack(subset["embedding"].values)
        if len(embeddings) < 2:
            continue

        matriz  = cosine_similarity(embeddings)
        mascara = ~np.eye(len(matriz), dtype=bool)
        coesao  = round(float(matriz[mascara].mean()), 4)
        resultados.append(CoesaoArtigo(
            artigo_key=str(artigo_key),
            doc_titulo=subset["doc_titulo"].iloc[0],
            artigo_id=subset["artigo_id"].iloc[0],
            n_chunks=len(embeddings),
            coesao_media=coesao,
        ))

    return sorted(resultados, key=lambda r: r.coesao_media, reverse=True)


def calcular_distancia_inter_artigo(df: pd.DataFrame) -> list[DistanciaArtigo]:
    """
    Distância inter-artigo dentro do mesmo regulamento.
    Unidade: centroide de cada artigo. Inclui todos os artigos.
    Devolve lista de DTOs ordenada por distancia_coseno descendente.
    """
    from sklearn.metrics.pairwise import cosine_similarity

    centroides_df = calcular_centroides_artigos(df)
    resultados: list[DistanciaArtigo] = []

    for (_, row_a), (_, row_b) in combinations(centroides_df.iterrows(), 2):
        if row_a["doc_titulo"] != row_b["doc_titulo"]:
            continue

        sim = cosine_similarity([row_a["centroide"]], [row_b["centroide"]])[0][0]
        resultados.append(DistanciaArtigo(
            doc_titulo=row_a["doc_titulo"],
            artigo_a=row_a["artigo_id"],
            artigo_b=row_b["artigo_id"],
            distancia_coseno=round(float(1 - sim), 4),
        ))

    return sorted(resultados, key=lambda r: r.distancia_coseno, reverse=True)