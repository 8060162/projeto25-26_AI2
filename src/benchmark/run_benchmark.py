"""
benchmark/run_benchmark.py
---------------------------
Responsabilidade única: executar o pipeline de retrieval sobre o dataset
de benchmark e produzir um BenchmarkReport.

ARQUITECTURA — acoplamento mínimo e deliberado:
  Este script tem UM ponto de contacto com o pipeline principal:
      from retriever.query import obter_contexto
  Nada mais é importado do pipeline. Toda a lógica de métricas,
  I/O e formatação está nos módulos deste subsistema.

FLUXO:
  1. Carregar dataset (benchmark_io)
  2. Para cada K em K_VALUES:
       Para cada query em paralelo (até MAX_WORKERS threads):
         a. Chamar obter_contexto(pergunta, n_resultados=K)
         b. Construir QueryResult com os retrieved_ids
         c. Calcular métricas individuais (benchmark_metrics)
  3. Agregar métricas por K e por tipo (benchmark_metrics)
  4. Construir e guardar BenchmarkReport (benchmark_io)
  5. Imprimir relatório no terminal (benchmark_report_printer)

NOTAS DE ESCALABILIDADE:
  - O retriever é chamado com ThreadPoolExecutor para paralelizar
    as queries sem conflito com o GIL (ChromaDB é thread-safe em leitura).
  - MAX_WORKERS controla a concorrência. Para 100+ queries, aumentar
    para 8-16 reduz o tempo de execução de forma linear.
  - A colecção ChromaDB é inicializada uma única vez antes do pool
    para evitar o custo de carregamento do modelo de embeddings
    por cada worker.

USO:
  python run_benchmark.py
  python run_benchmark.py --dataset benchmark/benchmark.json
  python run_benchmark.py --k 1 3 5 --workers 8
  python run_benchmark.py --output resultados/run_2025.json
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import logging
import os
import sys
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
_BENCHMARK_DIR = Path(__file__).resolve().parent
_SRC_DIR       = _BENCHMARK_DIR.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
# ─────────────────────────────────────────────────────────────────────────────

from benchmark_models import (
    BenchmarkDataset,
    BenchmarkReport,
    GoldId,
    QueryEntry,
    QueryResult,
    RetrievedItem,
)
from benchmark_io      import carregar_dataset, guardar_report
from benchmark_metrics import agregar_por_k, agregar_por_tipo, preencher_metricas
from benchmark_printer import imprimir_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Defaults ──────────────────────────────────────────────────────────────────

_K_VALUES_DEFAULT   = [1, 3, 5]
_MAX_WORKERS_DEFAULT = 4


# ── Execução de uma query ─────────────────────────────────────────────────────

def _executar_query(entry: QueryEntry, k: int) -> QueryResult:
    """
    Executa o retriever para uma query e devolve um QueryResult não avaliado.

    Isolado numa função para ser chamado via ThreadPoolExecutor.
    Captura qualquer excepção do retriever sem propagar — o campo
    `erro` do QueryResult regista a falha para diagnóstico.

    ÚNICO ponto de acoplamento ao pipeline principal.
    """
    # Import local: garante que o modelo de embeddings é carregado
    # uma vez por processo (singleton em db_client.py), não por thread.
    from retriever.query import obter_contexto

    try:
        artigos = obter_contexto(entry.pergunta, n_resultados=k)
        retrieved = [
            RetrievedItem(rank=i + 1, source=art.source, artigo_id=art.artigo_id)
            for i, art in enumerate(artigos)
        ]
        return QueryResult(
            query_id=entry.id,
            pergunta=entry.pergunta,
            gold_ids=entry.gold_ids,
            retrieved=retrieved,
            k=k,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Erro ao executar query '%s' (k=%d): %s", entry.id, k, exc)
        return QueryResult(
            query_id=entry.id,
            pergunta=entry.pergunta,
            gold_ids=entry.gold_ids,
            retrieved=[],
            k=k,
            erro=str(exc),
        )


# ── Runner para um valor de K ─────────────────────────────────────────────────

def _correr_para_k(
    dataset:     BenchmarkDataset,
    k:           int,
    max_workers: int,
) -> list[QueryResult]:
    """
    Executa todas as queries do dataset para um dado K.

    Usa ThreadPoolExecutor para paralelizar chamadas ao retriever.
    O progresso é logado a cada 10 queries para datasets grandes.

    Returns:
        Lista de QueryResult com métricas individuais preenchidas.
    """
    n        = len(dataset.queries)
    results: list[QueryResult] = [None] * n  # type: ignore[list-item]

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_executar_query, entry, k): i
            for i, entry in enumerate(dataset.queries)
        }

        concluidos = 0
        for future in concurrent.futures.as_completed(futures):
            idx           = futures[future]
            results[idx]  = preencher_metricas(future.result())
            concluidos   += 1
            if concluidos % 10 == 0 or concluidos == n:
                logger.info("  k=%d: %d/%d queries concluídas", k, concluidos, n)

    return results


# ── Snapshot de configuração ──────────────────────────────────────────────────

def _snapshot_configuracao() -> dict:
    """
    Captura um snapshot das settings relevantes para o relatório.

    Permite saber exactamente com que configuração cada execução foi feita,
    essencial para comparar resultados históricos.
    """
    try:
        from settings import (
            N_RESULTS, QUERY_FETCH, CHUNK_TARGET, CHUNK_MAX,
            COLLECTION_NAME, LLM_BACKEND,
        )
        return {
            "N_RESULTS":       N_RESULTS,
            "QUERY_FETCH":     QUERY_FETCH,
            "CHUNK_TARGET":    CHUNK_TARGET,
            "CHUNK_MAX":       CHUNK_MAX,
            "COLLECTION_NAME": COLLECTION_NAME,
            "LLM_BACKEND":     LLM_BACKEND,
        }
    except ImportError:
        return {}


# ── Orquestrador principal ────────────────────────────────────────────────────

def correr_benchmark(
    dataset_path: str,
    output_path:  str,
    k_values:     list[int],
    max_workers:  int,
) -> BenchmarkReport:
    """
    Executa o benchmark completo e devolve o BenchmarkReport.

    Fluxo:
      1. Carregar dataset
      2. Para cada K: executar queries → calcular métricas → agregar
      3. Construir relatório com metadados de rastreabilidade
      4. Guardar e devolver

    Args:
        dataset_path: caminho para benchmark.json
        output_path:  onde guardar benchmark_report.json
        k_values:     lista de K a testar (ex: [1, 3, 5])
        max_workers:  número de threads para paralelização

    Returns:
        BenchmarkReport completo.
    """
    dataset = carregar_dataset(dataset_path)
    logger.info(
        "Dataset carregado: %d queries (versão %s)",
        len(dataset.queries), dataset.versao,
    )

    # Mapa query_id → tipo para segmentação por tipo
    tipo_por_id = {q.id: q.tipo for q in dataset.queries}

    # Acumula todos os QueryResult do K principal (o maior K testado)
    # para incluir no relatório completo por query
    k_principal     = max(k_values)
    results_completos: list[QueryResult] = []

    resultados_por_k = []
    for k in sorted(k_values):
        logger.info("A correr retrieval com k=%d...", k)
        results = _correr_para_k(dataset, k, max_workers)
        resultados_por_k.append(agregar_por_k(results, k))

        if k == k_principal:
            results_completos = results

    por_tipo = agregar_por_tipo(results_completos, tipo_por_id)

    report = BenchmarkReport(
        timestamp=datetime.datetime.now().isoformat(timespec="seconds"),
        dataset_versao=dataset.versao,
        n_results_testados=sorted(k_values),
        configuracao=_snapshot_configuracao(),
        resultados_por_k=resultados_por_k,
        por_tipo=por_tipo,
        query_results=results_completos,
    )

    guardar_report(report, output_path)
    return report


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Corre o benchmark de retrieval e produz um relatório."
    )
    parser.add_argument(
        "--dataset", default="benchmark/benchmark.json",
        help="Caminho para o dataset de benchmark (default: benchmark/benchmark.json)",
    )
    parser.add_argument(
        "--output", default="benchmark/benchmark_report.json",
        help="Caminho de saída do relatório (default: benchmark/benchmark_report.json)",
    )
    parser.add_argument(
        "--k", nargs="+", type=int, default=_K_VALUES_DEFAULT,
        metavar="K",
        help=f"Valores de K a testar (default: {_K_VALUES_DEFAULT})",
    )
    parser.add_argument(
        "--workers", type=int, default=_MAX_WORKERS_DEFAULT,
        help=f"Threads de retrieval em paralelo (default: {_MAX_WORKERS_DEFAULT})",
    )
    return parser.parse_args()


def main() -> None:
    args   = _parse_args()
    report = correr_benchmark(
        dataset_path=args.dataset,
        output_path=args.output,
        k_values=args.k,
        max_workers=args.workers,
    )
    imprimir_report(report)


if __name__ == "__main__":
    main()