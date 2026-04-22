# benchmark/__init__.py
#
# Subsistema de benchmark de retrieval para o projecto RAG P.PORTO.
#
# Módulos:
#   benchmark_models.py   — DTOs: GoldId, QueryEntry, BenchmarkDataset, BenchmarkReport, …
#   benchmark_io.py       — serialização/deserialização JSON (único ponto de I/O)
#   benchmark_metrics.py  — cálculo puro de métricas (Hit@K, MRR, Recall@K)
#   benchmark_printer.py  — formatação e impressão do relatório
#   generate_dataset.py   — geração semi-automática do dataset via LLM (script)
#   run_benchmark.py      — runner principal (script)
#
# Scripts de entrada:
#   python benchmark/generate_dataset.py [--por-tipo N] [--dry-run]
#   python benchmark/run_benchmark.py    [--k 1 3 5] [--workers N]
#
# Acoplamento ao pipeline principal: ZERO excepto em run_benchmark.py,
# que importa exclusivamente `from retriever.query import obter_contexto`.