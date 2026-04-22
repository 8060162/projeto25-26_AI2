"""
benchmark/benchmark_metrics.py
--------------------------------
Responsabilidade única: calcular métricas de retrieval a partir de
listas de QueryResult. Funções puras — sem I/O, sem estado, sem
dependências externas ao subsistema de benchmark.

MÉTRICAS IMPLEMENTADAS:

  Hit@K  (por query)
    1 se pelo menos um gold_id está nos K retrieved, 0 caso contrário.
    É a métrica mais directamente ligada à utilidade do sistema para
    o utilizador final: "encontrou ou não encontrou a resposta?"

  MRR — Mean Reciprocal Rank  (agregado)
    Média de 1/rank do primeiro gold_id encontrado.
    Penaliza resultados correctos em posições baixas.
    0 se nenhum gold_id foi encontrado.

  Recall@K  (por query)
    Fracção dos gold_ids recuperados nos K resultados.
    |retrieved ∩ gold_ids| / |gold_ids|
    Relevante para queries multi-artigo onde todos os artigos importam.

NOTA SOBRE GRANULARIDADE:
  As métricas operam sobre GoldId = (source, artigo_id).
  Não sobre chunks — alinhado com o princípio fundamental do projecto
  de que a unidade atómica é o artigo.
  Um artigo truncado em 3 chunks que aparece em rank 2 conta como
  rank 2, não como 3 ocorrências separadas.
"""

from __future__ import annotations

from benchmark_models import (
    GoldId,
    MetricasAgregadas,
    MetricasPorTipo,
    QueryResult,
)


# ── Métricas por query ────────────────────────────────────────────────────────

def calcular_hit(result: QueryResult) -> bool:
    """
    Hit@K: True se pelo menos um gold_id está nos retrieved.

    Complexidade: O(K × |gold_ids|) — K e gold_ids são pequenos em prática.
    """
    gold_set = set(result.gold_ids)
    return any(item.to_gold_id() in gold_set for item in result.retrieved)


def calcular_reciprocal_rank(result: QueryResult) -> float:
    """
    Reciprocal Rank: 1/rank do primeiro gold_id encontrado.

    Devolve 0.0 se nenhum gold_id foi recuperado.
    rank é 1-based (o primeiro resultado tem rank=1).
    """
    gold_set = set(result.gold_ids)
    for item in result.retrieved:
        if item.to_gold_id() in gold_set:
            return 1.0 / item.rank
    return 0.0


def calcular_recall(result: QueryResult) -> float:
    """
    Recall@K: fracção dos gold_ids recuperados.

    |retrieved ∩ gold_ids| / |gold_ids|

    Devolve 0.0 se gold_ids estiver vazio (caso defensivo).
    Para queries directas com um único gold_id, é equivalente a Hit@K.
    Para queries multi-artigo, pode ser parcial (ex: 0.5 se recuperou 1/2).
    """
    if not result.gold_ids:
        return 0.0
    gold_set    = set(result.gold_ids)
    retrieved_set = {item.to_gold_id() for item in result.retrieved}
    return len(retrieved_set & gold_set) / len(gold_set)


def preencher_metricas(result: QueryResult) -> QueryResult:
    """
    Calcula e preenche hit, reciprocal_rank e recall num QueryResult.

    Não modifica o objecto recebido — devolve uma cópia com as métricas
    preenchidas. Resultados com erro mantêm os valores por omissão (0/False).
    """
    if result.erro is not None:
        return result

    from dataclasses import replace
    return replace(
        result,
        hit=calcular_hit(result),
        reciprocal_rank=calcular_reciprocal_rank(result),
        recall=calcular_recall(result),
    )


# ── Métricas agregadas ────────────────────────────────────────────────────────

def agregar_por_k(results: list[QueryResult], k: int) -> MetricasAgregadas:
    """
    Agrega Hit@K, MRR e Recall@K para um conjunto de QueryResult com o mesmo K.

    Queries com erro são contadas mas excluídas do cálculo das médias —
    n_erros permite ao utilizador avaliar se o número de falhas afecta
    a validade dos resultados.

    Args:
        results: lista de QueryResult já com métricas preenchidas por preencher_metricas().
        k:       valor de K (informativo — não filtra results).

    Returns:
        MetricasAgregadas com as médias calculadas.
    """
    validos = [r for r in results if r.erro is None]
    n_erros = len(results) - len(validos)

    if not validos:
        return MetricasAgregadas(
            k=k, n_queries=len(results),
            hit_rate=0.0, mrr=0.0, recall_medio=0.0,
            n_erros=n_erros,
        )

    hit_rate     = sum(1 for r in validos if r.hit)     / len(validos)
    mrr          = sum(r.reciprocal_rank for r in validos) / len(validos)
    recall_medio = sum(r.recall for r in validos)          / len(validos)

    return MetricasAgregadas(
        k=k,
        n_queries=len(results),
        hit_rate=round(hit_rate,     4),
        mrr=round(mrr,              4),
        recall_medio=round(recall_medio, 4),
        n_erros=n_erros,
    )


def agregar_por_tipo(
    results: list[QueryResult],
    entradas: dict[str, str],
) -> list[MetricasPorTipo]:
    """
    Segmenta os resultados por tipo de query e calcula métricas por segmento.

    Args:
        results:  lista de QueryResult com métricas preenchidas.
        entradas: dict {query_id → tipo} — evita acoplar este módulo ao dataset.
                  Construído pelo runner a partir de BenchmarkDataset.

    Returns:
        Lista de MetricasPorTipo, uma por tipo presente nos dados.
    """
    from collections import defaultdict

    por_tipo: dict[str, list[QueryResult]] = defaultdict(list)
    for r in results:
        tipo = entradas.get(r.query_id, "desconhecido")
        por_tipo[tipo].append(r)

    metricas: list[MetricasPorTipo] = []
    for tipo, grupo in sorted(por_tipo.items()):
        validos = [r for r in grupo if r.erro is None]
        if not validos:
            continue
        metricas.append(MetricasPorTipo(
            tipo=tipo,
            n_queries=len(validos),
            hit_rate=round(sum(1 for r in validos if r.hit)         / len(validos), 4),
            mrr=round(sum(r.reciprocal_rank for r in validos)       / len(validos), 4),
            recall_medio=round(sum(r.recall for r in validos)       / len(validos), 4),
        ))

    return metricas