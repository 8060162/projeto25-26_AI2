"""
benchmark/benchmark_io.py
--------------------------
Responsabilidade única: serializar e deserializar os tipos de dados
do benchmark para/de JSON em disco.

SEPARAÇÃO: este módulo é o único ponto do subsistema que toca em ficheiros.
           Todos os outros módulos recebem e devolvem objectos Python puros.

CONTRATOS:
  - carregar_dataset(path)  → BenchmarkDataset
  - guardar_dataset(ds, path)
  - guardar_report(report, path)
  - carregar_report(path)   → BenchmarkReport

FORMATO DO FICHEIRO benchmark.json:
  {
    "versao": "1.0.0",
    "descricao": "...",
    "queries": [
      {
        "id": "q001",
        "pergunta": "...",
        "gold_ids": [{"source": "...", "artigo_id": "..."}],
        "tipo": "directa",
        "dificuldade": "facil",
        "notas": ""
      }
    ]
  }
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from benchmark_models import (
    BenchmarkDataset,
    BenchmarkReport,
    GoldId,
    MetricasAgregadas,
    MetricasPorTipo,
    QueryEntry,
    QueryResult,
    RetrievedItem,
)

logger = logging.getLogger(__name__)


# ── Dataset ───────────────────────────────────────────────────────────────────

def carregar_dataset(path: str | Path) -> BenchmarkDataset:
    """
    Lê o ficheiro benchmark.json e devolve um BenchmarkDataset.

    Raises:
        FileNotFoundError: se o ficheiro não existir.
        ValueError:        se o JSON estiver malformado ou campos obrigatórios
                           estiverem ausentes.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Dataset não encontrado: '{p}'. "
            f"Corre generate_dataset.py para o criar."
        )

    try:
        raw: dict = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON inválido em '{p}': {e}") from e

    queries: list[QueryEntry] = []
    for item in raw.get("queries", []):
        try:
            gold_ids = [
                GoldId(source=g["source"], artigo_id=g["artigo_id"])
                for g in item.get("gold_ids", [])
            ]
            queries.append(QueryEntry(
                id=item["id"],
                pergunta=item["pergunta"],
                gold_ids=gold_ids,
                tipo=item.get("tipo", "directa"),
                dificuldade=item.get("dificuldade", "facil"),
                notas=item.get("notas", ""),
            ))
        except KeyError as e:
            raise ValueError(
                f"Campo obrigatório ausente na query '{item.get('id', '?')}': {e}"
            ) from e

    return BenchmarkDataset(
        versao=raw.get("versao", "0.0.0"),
        descricao=raw.get("descricao", ""),
        queries=queries,
    )


def guardar_dataset(dataset: BenchmarkDataset, path: str | Path) -> None:
    """
    Serializa um BenchmarkDataset para JSON e guarda em disco.

    Cria directorias intermédias se não existirem.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    raw: dict[str, Any] = {
        "versao":    dataset.versao,
        "descricao": dataset.descricao,
        "queries": [
            {
                "id":          q.id,
                "pergunta":    q.pergunta,
                "gold_ids":    [{"source": g.source, "artigo_id": g.artigo_id} for g in q.gold_ids],
                "tipo":        q.tipo,
                "dificuldade": q.dificuldade,
                "notas":       q.notas,
            }
            for q in dataset.queries
        ],
    }
    p.write_text(json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Dataset guardado: %s (%d queries)", p, len(dataset.queries))


# ── Relatório ─────────────────────────────────────────────────────────────────

def guardar_report(report: BenchmarkReport, path: str | Path) -> None:
    """
    Serializa um BenchmarkReport para JSON e guarda em disco.

    O relatório é separado do dataset — é um artefacto de execução,
    não versionado com o código.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    raw: dict[str, Any] = {
        "timestamp":          report.timestamp,
        "dataset_versao":     report.dataset_versao,
        "n_results_testados": report.n_results_testados,
        "configuracao":       report.configuracao,
        "resultados_por_k": [
            {
                "k":             m.k,
                "n_queries":     m.n_queries,
                "hit_rate":      m.hit_rate,
                "mrr":           m.mrr,
                "recall_medio":  m.recall_medio,
                "n_erros":       m.n_erros,
            }
            for m in report.resultados_por_k
        ],
        "por_tipo": [
            {
                "tipo":         t.tipo,
                "n_queries":    t.n_queries,
                "hit_rate":     t.hit_rate,
                "mrr":          t.mrr,
                "recall_medio": t.recall_medio,
            }
            for t in report.por_tipo
        ],
        "query_results": [
            {
                "query_id":        r.query_id,
                "pergunta":        r.pergunta,
                "k":               r.k,
                "hit":             r.hit,
                "reciprocal_rank": r.reciprocal_rank,
                "recall":          r.recall,
                "gold_ids":        [{"source": g.source, "artigo_id": g.artigo_id} for g in r.gold_ids],
                "retrieved":       [
                    {"rank": i.rank, "source": i.source, "artigo_id": i.artigo_id}
                    for i in r.retrieved
                ],
                "erro": r.erro,
            }
            for r in report.query_results
        ],
    }
    p.write_text(json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Relatório guardado: %s", p)


def carregar_report(path: str | Path) -> BenchmarkReport:
    """
    Lê um relatório JSON previamente guardado.

    Útil para comparar execuções históricas sem re-correr o benchmark.

    Raises:
        FileNotFoundError: se o ficheiro não existir.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Relatório não encontrado: '{p}'.")

    raw = json.loads(p.read_text(encoding="utf-8"))

    resultados_por_k = [
        MetricasAgregadas(**m) for m in raw.get("resultados_por_k", [])
    ]
    por_tipo = [
        MetricasPorTipo(**t) for t in raw.get("por_tipo", [])
    ]
    query_results = [
        QueryResult(
            query_id=r["query_id"],
            pergunta=r["pergunta"],
            k=r["k"],
            hit=r["hit"],
            reciprocal_rank=r["reciprocal_rank"],
            recall=r["recall"],
            gold_ids=[GoldId(**g) for g in r["gold_ids"]],
            retrieved=[RetrievedItem(**i) for i in r["retrieved"]],
            erro=r.get("erro"),
        )
        for r in raw.get("query_results", [])
    ]

    return BenchmarkReport(
        timestamp=raw["timestamp"],
        dataset_versao=raw["dataset_versao"],
        n_results_testados=raw["n_results_testados"],
        configuracao=raw["configuracao"],
        resultados_por_k=resultados_por_k,
        por_tipo=por_tipo,
        query_results=query_results,
    )