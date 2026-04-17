"""
evaluate.py
-----------
Responsabilidade única: orquestrar o relatório de avaliação de embeddings.

SEPARAÇÃO (refactor):
  Este módulo é agora um orquestrador puro — sem cálculos, sem formatação.
  A sequência é:
    1. Carregar dados (_chromadb_loader)
    2. Calcular métricas (metrics.py)
    3. Formatar e imprimir (report_formatter.py)
    4. Guardar relatório em ficheiro (I/O local)

  A separação em três módulos torna cada camada testável de forma isolada:
    - metrics.py        → testável com pytest, sem I/O, sem formatação
    - report_formatter  → testável com dados sintéticos, sem ChromaDB
    - evaluate.py       → ponto de entrada CLI, integra as outras duas camadas

Uso:
    python -m src.analysis.evaluate

Requisito:
    pip install scikit-learn
"""

import datetime
import io
import logging
import os
import sys
from contextlib import redirect_stdout

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

_LARGURA_TOTAL = 72
_REPORT_PATH   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluate_report.txt")


# ── Utilitário de I/O ─────────────────────────────────────────────────────────

class _Tee:
    """
    Stream que escreve simultaneamente para dois destinos (terminal + buffer).
    Usado com contextlib.redirect_stdout para capturar output sem suprimir
    a impressão no terminal.
    """
    def __init__(self, stream_a, stream_b):
        self._a = stream_a
        self._b = stream_b

    def write(self, data: str) -> int:
        self._a.write(data)
        self._b.write(data)
        return len(data)

    def flush(self) -> None:
        self._a.flush()
        self._b.flush()


# ── Cabeçalhos de secção ──────────────────────────────────────────────────────

def _titulo_secao(texto: str) -> None:
    print(f"\n{'═' * _LARGURA_TOTAL}")
    print(f"  {texto}")
    print(f"{'═' * _LARGURA_TOTAL}")


def _subtitulo(texto: str) -> None:
    print(f"\n  ┌─ {texto}")
    print(f"  └{'─' * (_LARGURA_TOTAL - 4)}")


# ── Relatório ─────────────────────────────────────────────────────────────────

def imprimir_relatorio(df) -> None:
    """
    Calcula todas as métricas e imprime o relatório completo.
    O output é escrito simultaneamente no terminal e em evaluate_report.txt.

    Delega cálculos a metrics.py e formatação a report_formatter.py.
    """
    from analysis.metrics import (
        adicionar_chave_artigo,
        resumo_corpus,
        calcular_silhouette_regulamento,
        calcular_coesao_intra_regulamento,
        calcular_distancia_inter_regulamento,
        calcular_silhouette_artigo,
        calcular_coesao_intra_artigo,
        calcular_distancia_inter_artigo,
    )
    from analysis.report_formatter import (
        imprimir_resumo_corpus,
        imprimir_silhouette_regulamento,
        imprimir_coesao_regulamento,
        imprimir_distancia_regulamento,
        imprimir_silhouette_artigo,
        imprimir_coesao_artigo,
        imprimir_distancia_artigo,
    )

    df     = adicionar_chave_artigo(df)
    buffer = io.StringIO()

    with redirect_stdout(_Tee(sys.stdout, buffer)):

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"  Gerado em: {timestamp}")

        # ── Resumo ────────────────────────────────────────────────────────────
        _titulo_secao("RESUMO DO CORPUS")
        _subtitulo("Estatísticas gerais")
        imprimir_resumo_corpus(resumo_corpus(df))

        # ── NÍVEL 1 ───────────────────────────────────────────────────────────
        _titulo_secao("NÍVEL 1 — AVALIAÇÃO POR REGULAMENTO")
        print(f"  Unidade de análise: centroide de cada artigo")
        print(f"  Inclui todos os artigos (com ou sem divisão em chunks)")

        _subtitulo("Silhouette Score  (separação entre regulamentos)")
        imprimir_silhouette_regulamento(calcular_silhouette_regulamento(df))

        _subtitulo("Coesão intra-regulamento  (homogeneidade semântica dos artigos)")
        imprimir_coesao_regulamento(calcular_coesao_intra_regulamento(df))

        _subtitulo("Distância inter-regulamento  (separação semântica entre pares)")
        imprimir_distancia_regulamento(calcular_distancia_inter_regulamento(df))

        # ── NÍVEL 2 ───────────────────────────────────────────────────────────
        _titulo_secao("NÍVEL 2 — VERIFICAÇÃO DE CHUNKING (por Artigo)")
        print(f"  Unidade de análise: chunks individuais")
        print(f"  Inclui apenas artigos com 2+ chunks (truncated=true)")

        _subtitulo("Silhouette Score  (coesão intra-artigo vs separação inter-artigo)")
        imprimir_silhouette_artigo(calcular_silhouette_artigo(df))

        _subtitulo("Coesão intra-artigo  (chunks do mesmo artigo)")
        imprimir_coesao_artigo(calcular_coesao_intra_artigo(df))

        _subtitulo("Distância inter-artigo  (pares mais distantes por regulamento)")
        imprimir_distancia_artigo(calcular_distancia_inter_artigo(df))

        print(f"\n{'═' * _LARGURA_TOTAL}\n")

    with open(_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(buffer.getvalue())

    logger.info("Relatório gravado em: %s", _REPORT_PATH)


# ── Ponto de entrada ──────────────────────────────────────────────────────────

def main() -> None:
    try:
        import sklearn  # noqa: F401
    except ImportError:
        logger.error("scikit-learn não instalado. Executa: pip install scikit-learn")
        sys.exit(1)

    from _chromadb_loader import carregar_dataframe

    df = carregar_dataframe()
    imprimir_relatorio(df)


if __name__ == "__main__":
    main()