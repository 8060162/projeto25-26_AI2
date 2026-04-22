"""
benchmark/benchmark_printer.py
--------------------------------
Responsabilidade única: formatar e imprimir um BenchmarkReport no terminal.

SEPARAÇÃO: este módulo recebe DTOs já calculados e produz output legível.
           Não calcula métricas, não lê ficheiros, não chama o retriever.
           Pode ser usado independentemente do runner para re-imprimir
           um relatório histórico carregado com benchmark_io.carregar_report().

USO ISOLADO:
    from benchmark_io import carregar_report
    from benchmark_printer import imprimir_report
    report = carregar_report("benchmark/benchmark_report.json")
    imprimir_report(report)
"""

from __future__ import annotations

from benchmark_models import BenchmarkReport, MetricasAgregadas, QueryResult

_LARGURA = 68


def _linha(char: str = "─") -> None:
    print(char * _LARGURA)


def _titulo(texto: str, char: str = "═") -> None:
    print(f"\n{char * _LARGURA}")
    print(f"  {texto}")
    print(char * _LARGURA)


def _barra(valor: float, largura: int = 24) -> str:
    """Barra visual proporcional a valor ∈ [0, 1]."""
    preenchimento = round(valor * largura)
    return "█" * preenchimento + "░" * (largura - preenchimento)


def _fmt_metrica(nome: str, valor: float) -> str:
    barra = _barra(valor)
    return f"  {nome:<18} {valor:>6.4f}  [{barra}]"


# ── Secções ───────────────────────────────────────────────────────────────────

def _imprimir_cabecalho(report: BenchmarkReport) -> None:
    _titulo("BENCHMARK DE RETRIEVAL — P.PORTO RAG")
    print(f"  Execução      : {report.timestamp}")
    print(f"  Dataset       : versão {report.dataset_versao}")
    print(f"  K testados    : {report.n_results_testados}")
    cfg = report.configuracao
    if cfg:
        print(f"  N_RESULTS     : {cfg.get('N_RESULTS', '?')}")
        print(f"  QUERY_FETCH   : {cfg.get('QUERY_FETCH', '?')}")
        print(f"  CHUNK_TARGET  : {cfg.get('CHUNK_TARGET', '?')}")
        print(f"  Colecção      : {cfg.get('COLLECTION_NAME', '?')}")


def _imprimir_metricas_por_k(metricas: list[MetricasAgregadas]) -> None:
    _titulo("MÉTRICAS AGREGADAS POR K")

    col_k   = 4
    col_n   = 8
    col_hit = 10
    col_mrr = 10
    col_rec = 10
    col_err = 7

    header = (
        f"  {'K':>{col_k}}  {'Queries':>{col_n}}  "
        f"{'Hit@K':>{col_hit}}  {'MRR':>{col_mrr}}  "
        f"{'Recall@K':>{col_rec}}  {'Erros':>{col_err}}"
    )
    print(header)
    _linha()

    for m in sorted(metricas, key=lambda x: x.k):
        erros_str = f"{m.n_erros}" if m.n_erros == 0 else f"⚠ {m.n_erros}"
        print(
            f"  {m.k:>{col_k}}  {m.n_queries:>{col_n}}  "
            f"{m.hit_rate:>{col_hit}.4f}  {m.mrr:>{col_mrr}.4f}  "
            f"{m.recall_medio:>{col_rec}.4f}  {erros_str:>{col_err}}"
        )

    print()
    # Barra visual para o K principal
    k_principal = max(metricas, key=lambda x: x.k)
    print(f"  Visualização (K={k_principal.k}):")
    print(_fmt_metrica("Hit@K", k_principal.hit_rate))
    print(_fmt_metrica("MRR", k_principal.mrr))
    print(_fmt_metrica("Recall@K", k_principal.recall_medio))

    print()
    print("  Legenda: Hit@K = fracção com ≥1 artigo correcto recuperado")
    print("           MRR   = posição média do 1º artigo correcto (1/rank)")
    print("           Recall= fracção dos gold_ids recuperados (multi-artigo)")


def _imprimir_por_tipo(report: BenchmarkReport) -> None:
    if not report.por_tipo:
        return

    _titulo("MÉTRICAS POR TIPO DE QUERY")
    col_tipo = 12
    col_n    = 8
    col_hit  = 10
    col_mrr  = 10
    col_rec  = 10

    header = (
        f"  {'Tipo':<{col_tipo}}  {'Queries':>{col_n}}  "
        f"{'Hit@K':>{col_hit}}  {'MRR':>{col_mrr}}  {'Recall@K':>{col_rec}}"
    )
    print(header)
    _linha()

    for t in report.por_tipo:
        print(
            f"  {t.tipo:<{col_tipo}}  {t.n_queries:>{col_n}}  "
            f"{t.hit_rate:>{col_hit}.4f}  {t.mrr:>{col_mrr}.4f}  "
            f"{t.recall_medio:>{col_rec}.4f}"
        )

    print()
    print("  Tipos: directa=lexical simples · tematica=semântica · multi=multi-artigo · negativa=fora de corpus")


def _imprimir_falhas(results: list[QueryResult], top_n: int = 10) -> None:
    """Imprime as queries onde o retriever falhou (hit=False), por ordem de RR."""
    falhas = [r for r in results if not r.hit and r.erro is None]
    if not falhas:
        _titulo("QUERIES COM FALHA")
        print("  Nenhuma falha de retrieval.")
        return

    _titulo(f"QUERIES COM FALHA (top {min(top_n, len(falhas))} de {len(falhas)})")
    falhas_sorted = sorted(falhas, key=lambda r: r.reciprocal_rank)

    for r in falhas_sorted[:top_n]:
        gold_str = ", ".join(f"{g.artigo_id}@{g.source}" for g in r.gold_ids) or "(negativa)"
        ret_str  = ", ".join(f"{i.artigo_id}@{i.source}" for i in r.retrieved[:3]) or "—"
        _linha("·")
        print(f"  [{r.query_id}] {r.pergunta}")
        print(f"  Gold    : {gold_str}")
        print(f"  Top-3   : {ret_str}")

    if len(falhas) > top_n:
        print(f"\n  ... e mais {len(falhas) - top_n} falhas. Ver benchmark_report.json para detalhes completos.")


def _imprimir_erros(results: list[QueryResult]) -> None:
    """Imprime queries que falharam com erro de execução."""
    erros = [r for r in results if r.erro is not None]
    if not erros:
        return
    _titulo(f"ERROS DE EXECUÇÃO ({len(erros)})")
    for r in erros:
        print(f"  [{r.query_id}] {r.erro}")


# ── Interface pública ─────────────────────────────────────────────────────────

def imprimir_report(report: BenchmarkReport) -> None:
    """
    Imprime o relatório completo no terminal.

    Ordem:
      1. Cabeçalho com metadados de rastreabilidade
      2. Métricas agregadas por K (tabela + barras visuais)
      3. Métricas por tipo de query
      4. Diagnóstico de falhas (queries onde o retriever não encontrou gold)
      5. Erros de execução (se existirem)
    """
    _imprimir_cabecalho(report)
    _imprimir_metricas_por_k(report.resultados_por_k)
    _imprimir_por_tipo(report)
    _imprimir_falhas(report.query_results)
    _imprimir_erros(report.query_results)

    _titulo("FIM DO RELATÓRIO", "═")
    print(f"  Relatório completo guardado em benchmark_report.json\n")