"""
analysis/evaluate.py
---------------------
Métricas quantitativas de qualidade dos embeddings para um sistema RAG
sobre legislação académica do P.PORTO.

ARQUITECTURA DE AVALIAÇÃO
─────────────────────────
Existem dois níveis de cluster, com unidades de análise distintas:

  Corpus
    └── Regulamento A  ← cluster de nível 1
    │     └── Artigo 1  → centroide = média dos embeddings dos seus chunks
    │     └── Artigo 2  → centroide = média dos embeddings dos seus chunks
    └── Regulamento B  ← cluster de nível 1
          └── ...

PRINCÍPIO FUNDAMENTAL:
  A unidade atómica de conteúdo é o ARTIGO, não o chunk.
  Os chunks são um artefacto de arquitectura de indexação.
  Toda a avaliação ao nível do regulamento opera sobre centroides
  de artigos — nunca sobre chunks directos.

NÍVEL 1 — REGULAMENTO (unidade: centroide de artigo)
  Responde a: os artigos do mesmo regulamento são coesos entre si
  e os regulamentos estão separados entre si?
  - Silhouette Score    : cada artigo (centroide) vs outros artigos
  - Coesão intra        : similaridade média entre centroides do mesmo regulamento
  - Distância inter     : distância entre centroides de regulamentos

NÍVEL 2 — ARTIGO / verificação de chunking (unidade: chunk)
  Responde a: a divisão em chunks preservou a coerência semântica?
  - Silhouette Score    : cada chunk vs outros chunks (só artigos com 2+)
  - Coesão intra        : similaridade média entre chunks do mesmo artigo
  - Distância inter     : distância entre centroides de artigos (mesmo regulamento)

Uso:
    python -m src.analysis.evaluate

Requisito:
    pip install scikit-learn
"""

import logging
import os
import sys
from itertools import combinations

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ── Preparação do DataFrame ───────────────────────────────────────────────────

def _adicionar_chave_artigo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adiciona a coluna 'artigo_key' = doc_titulo + ' § ' + artigo_id.
    Esta chave identifica univocamente um artigo no corpus inteiro.
    Necessária porque artigo_id (ex: 'ART_1') repete-se entre documentos.
    """
    df = df.copy()
    df["artigo_key"] = df["doc_titulo"] + " § " + df["artigo_id"]
    return df


def _calcular_centroides_artigos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Single Source of Truth para centroides de artigos.

    Devolve um DataFrame com UMA linha por artigo:
        artigo_key | doc_titulo | artigo_id | n_chunks | centroide

    O centroide é a média dos embeddings de todos os chunks do artigo,
    independentemente de truncated. Artigos com chunk único têm centroide
    igual ao embedding desse chunk — o que é correcto e esperado.

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


# ── NÍVEL 1 — Métricas por Regulamento ───────────────────────────────────────

def calcular_silhouette_regulamento(df: pd.DataFrame) -> float:
    """
    Silhouette Score ao nível do regulamento.

    Unidade de análise: centroide de cada artigo (1 vector por artigo).
    Label: doc_titulo do artigo.

    Cada artigo é avaliado pela sua proximidade aos outros artigos do
    mesmo regulamento vs artigos de outros regulamentos.
    Inclui TODOS os artigos (independentemente de truncated).

    Intervalo [-1, 1]:
      +1.0 → artigos do mesmo regulamento muito próximos e bem separados
       0.0 → sobreposição entre regulamentos (normal neste domínio)
      -1.0 → artigos mais próximos de outro regulamento do que do seu
    """
    from sklearn.metrics import silhouette_score

    centroides_df = _calcular_centroides_artigos(df)

    n_docs = centroides_df["doc_titulo"].nunique()
    if n_docs < 2:
        logger.warning("Silhouette (regulamento) requer 2+ regulamentos. Encontrados: %d.", n_docs)
        return float("nan")

    # Verifica que todos os regulamentos têm 2+ artigos (requisito do Silhouette)
    contagens = centroides_df.groupby("doc_titulo").size()
    docs_validos = contagens[contagens >= 2].index
    centroides_df = centroides_df[centroides_df["doc_titulo"].isin(docs_validos)]

    if centroides_df["doc_titulo"].nunique() < 2:
        logger.warning("Silhouette (regulamento): regulamentos insuficientes com 2+ artigos.")
        return float("nan")

    embeddings = np.stack(centroides_df["centroide"].values)
    labels     = centroides_df["doc_titulo"].values

    return round(float(silhouette_score(embeddings, labels, metric="cosine")), 4)


def calcular_coesao_intra_regulamento(df: pd.DataFrame) -> pd.DataFrame:
    """
    Coesão intra-regulamento.

    Unidade de análise: centroide de cada artigo.
    Para cada regulamento, calcula a similaridade coseno média entre
    todos os pares de centroides de artigos do mesmo regulamento.
    Inclui TODOS os artigos (independentemente de truncated).

    Responde a: os artigos de um regulamento são semanticamente homogéneos?

    Retorna DataFrame ordenado por coesao_media descendente:
        doc_titulo | n_artigos | coesao_media
    """
    from sklearn.metrics.pairwise import cosine_similarity

    centroides_df = _calcular_centroides_artigos(df)
    resultados = []

    for doc, subset in centroides_df.groupby("doc_titulo"):
        embeddings = np.stack(subset["centroide"].values)
        n = len(embeddings)

        if n < 2:
            resultados.append({
                "doc_titulo":   doc,
                "n_artigos":    n,
                "coesao_media": float("nan"),
            })
            continue

        matriz  = cosine_similarity(embeddings)
        mascara = ~np.eye(n, dtype=bool)
        coesao  = matriz[mascara].mean()

        resultados.append({
            "doc_titulo":   doc,
            "n_artigos":    n,
            "coesao_media": round(float(coesao), 4),
        })

    return (
        pd.DataFrame(resultados)
        .sort_values("coesao_media", ascending=False)
        .reset_index(drop=True)
    )


def calcular_distancia_inter_regulamento(df: pd.DataFrame) -> pd.DataFrame:
    """
    Distância inter-regulamento.

    O centroide de um regulamento é a média dos centroides dos seus artigos
    (não a média directa dos chunks — respeita o princípio fundamental).

    Calcula a distância coseno entre todos os pares de centroides de
    regulamentos e devolve DataFrame ordenado por distancia_coseno descendente:
        doc_a | doc_b | distancia_coseno
    """
    from sklearn.metrics.pairwise import cosine_similarity

    centroides_df = _calcular_centroides_artigos(df)

    # Centroide do regulamento = média dos centroides dos seus artigos
    centroides_reg = (
        centroides_df.groupby("doc_titulo")["centroide"]
        .apply(lambda vecs: np.mean(np.stack(vecs.values), axis=0))
    )

    resultados = []
    for (doc_a, c_a), (doc_b, c_b) in combinations(centroides_reg.items(), 2):
        sim = cosine_similarity([c_a], [c_b])[0][0]
        resultados.append({
            "doc_a":            doc_a,
            "doc_b":            doc_b,
            "distancia_coseno": round(float(1 - sim), 4),
        })

    if not resultados:
        return pd.DataFrame(columns=["doc_a", "doc_b", "distancia_coseno"])

    return (
        pd.DataFrame(resultados)
        .sort_values("distancia_coseno", ascending=False)
        .reset_index(drop=True)
    )


# ── NÍVEL 2 — Métricas por Artigo (verificação de chunking) ──────────────────

def calcular_silhouette_artigo(df: pd.DataFrame) -> float:
    """
    Silhouette Score ao nível do artigo — verificação de chunking.

    Unidade de análise: chunks individuais.
    Label: artigo_key de cada chunk.
    Inclui APENAS artigos com 2+ chunks (truncated=true).

    Cada chunk é avaliado pela sua proximidade aos outros chunks do mesmo
    artigo vs chunks de outros artigos.

    Responde a: a divisão em chunks preservou a coerência semântica
    do artigo? Os chunks do mesmo artigo estão próximos entre si?

    Intervalo [-1, 1]:
      +1.0 → chunks do mesmo artigo muito coesos e separados de outros
       0.0 → chunks na fronteira semântica entre artigos
      -1.0 → chunks mais próximos de outro artigo do que do seu
    """
    from sklearn.metrics import silhouette_score

    # Apenas artigos divididos em múltiplos chunks
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


def calcular_coesao_intra_artigo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Coesão intra-artigo — verificação de chunking.

    Unidade de análise: chunks individuais.
    Inclui APENAS artigos com 2+ chunks (truncated=true).
    Artigos com chunk único são excluídos — não há par para comparar,
    e a sua ausência não indica problema de qualidade.

    Responde a: os chunks gerados a partir do mesmo artigo são
    semanticamente semelhantes entre si?

    Retorna DataFrame ordenado por coesao_media descendente:
        artigo_key | doc_titulo | artigo_id | n_chunks | coesao_media
    """
    from sklearn.metrics.pairwise import cosine_similarity

    artigos_divididos = df[df["truncated"] == "true"]

    if artigos_divididos.empty:
        logger.warning(
            "Nenhum artigo com 2+ chunks (truncated=true). "
            "Coesão intra-artigo não aplicável."
        )
        return pd.DataFrame(columns=["artigo_key", "doc_titulo", "artigo_id",
                                     "n_chunks", "coesao_media"])

    resultados = []
    for artigo_key, subset in artigos_divididos.groupby("artigo_key"):
        embeddings = np.stack(subset["embedding"].values)

        if len(embeddings) < 2:
            continue  # salvaguarda

        matriz  = cosine_similarity(embeddings)
        mascara = ~np.eye(len(matriz), dtype=bool)
        coesao  = matriz[mascara].mean()

        resultados.append({
            "artigo_key":   artigo_key,
            "doc_titulo":   subset["doc_titulo"].iloc[0],
            "artigo_id":    subset["artigo_id"].iloc[0],
            "n_chunks":     len(embeddings),
            "coesao_media": round(float(coesao), 4),
        })

    return (
        pd.DataFrame(resultados)
        .sort_values("coesao_media", ascending=False)
        .reset_index(drop=True)
    )


def calcular_distancia_inter_artigo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Distância inter-artigo dentro do mesmo regulamento.

    Unidade de análise: centroide de cada artigo (via _calcular_centroides_artigos).
    Inclui TODOS os artigos (independentemente de truncated).
    Compara apenas artigos do mesmo regulamento.
    Devolve os top 15 pares mais distantes.

    Responde a: dentro de um regulamento, quais os artigos semanticamente
    mais afastados entre si?

    Retorna DataFrame:
        doc_titulo | artigo_a | artigo_b | distancia_coseno
    """
    from sklearn.metrics.pairwise import cosine_similarity

    centroides_df = _calcular_centroides_artigos(df)

    resultados = []
    for (_, row_a), (_, row_b) in combinations(centroides_df.iterrows(), 2):
        if row_a["doc_titulo"] != row_b["doc_titulo"]:
            continue

        sim = cosine_similarity([row_a["centroide"]], [row_b["centroide"]])[0][0]
        resultados.append({
            "doc_titulo":       row_a["doc_titulo"],
            "artigo_a":         row_a["artigo_id"],
            "artigo_b":         row_b["artigo_id"],
            "distancia_coseno": round(float(1 - sim), 4),
        })

    if not resultados:
        return pd.DataFrame(columns=["doc_titulo", "artigo_a", "artigo_b",
                                     "distancia_coseno"])

    return (
        pd.DataFrame(resultados)
        .sort_values("distancia_coseno", ascending=False)
        .reset_index(drop=True)
    )


# ── Estatísticas do corpus ────────────────────────────────────────────────────

def resumo_corpus(df: pd.DataFrame) -> None:
    """Imprime estatísticas gerais do corpus."""
    n_chunks          = len(df)
    n_artigos         = df["artigo_key"].nunique()
    n_docs            = df["doc_titulo"].nunique()
    artigos_divididos = df[df["truncated"] == "true"]["artigo_key"].nunique()
    artigos_simples   = n_artigos - artigos_divididos

    print(f"\n  {'Documentos indexados':<35} {n_docs:>5}")
    print(f"  {'Artigos totais':<35} {n_artigos:>5}")
    print(f"  {'Artigos com 1 chunk (não divididos)':<35} {artigos_simples:>5}")
    print(f"  {'Artigos com 2+ chunks (divididos)':<35} {artigos_divididos:>5}")
    print(f"  {'Total de chunks':<35} {n_chunks:>5}")
    if n_artigos > 0:
        print(f"  {'Média de chunks por artigo':<35} {n_chunks/n_artigos:>5.2f}")


# ── Formatação ────────────────────────────────────────────────────────────────

_LARGURA_NOME  = 52
_LARGURA_TOTAL = 72


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


# ── Impressão — Nível 1 ───────────────────────────────────────────────────────

def _imprimir_silhouette_regulamento(score: float) -> None:
    barra = _barra_visual(max(score, 0), 0, 1, largura=20)

    if np.isnan(score):
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


def _imprimir_coesao_regulamento(df_coesao: pd.DataFrame) -> None:
    if df_coesao.empty:
        print("\n  Sem dados suficientes.")
        return

    validos = df_coesao["coesao_media"].dropna()
    minimo  = validos.min()
    maximo  = validos.max()
    media   = validos.mean()
    mediana = validos.median()

    print(f"\n  Regulamentos avaliados : {len(df_coesao)}")
    print(f"  Média de coesão        : {media:.4f}")
    print(f"  Mediana                : {mediana:.4f}")
    print()

    col = _LARGURA_NOME
    print(f"  {'Regulamento':<{col}}  {'Artigos':>7}  {'Coesão':>6}  Barra")
    print(f"  {'─'*col}  {'───────':>7}  {'──────':>6}  {'─'*20}")

    for _, row in df_coesao.iterrows():
        doc   = _truncar(str(row["doc_titulo"]), col)
        nart  = int(row["n_artigos"])
        valor = row["coesao_media"]

        if np.isnan(valor):
            print(f"  {doc:<{col}}  {nart:>7}     n/a")
            continue

        barra = _barra_visual(valor, minimo, maximo)
        print(f"  {doc:<{col}}  {nart:>7}  {valor:>6.4f}  {barra}")

    print(f"\n  Leitura: barra mais longa = artigos do regulamento mais homogéneos entre si.")
    print(f"  Coesão = similaridade coseno média entre centroides de artigos do mesmo regulamento.")
    print(f"  Valores > 0.5 são normais em regulamentos académicos do mesmo instituto.")


def _imprimir_distancia_regulamento(df_dist: pd.DataFrame) -> None:
    if df_dist.empty:
        print("\n  Sem dados suficientes.")
        return

    minimo = df_dist["distancia_coseno"].min()
    maximo = df_dist["distancia_coseno"].max()

    col = 34
    print(f"\n  {'Regulamento A':<{col}}  {'Regulamento B':<{col}}  {'Dist.':>5}  Barra")
    print(f"  {'─'*col}  {'─'*col}  {'─────':>5}  {'─'*20}")

    for _, row in df_dist.iterrows():
        a     = _truncar(str(row["doc_a"]), col)
        b     = _truncar(str(row["doc_b"]), col)
        valor = row["distancia_coseno"]
        barra = _barra_visual(valor, minimo, maximo)
        print(f"  {a:<{col}}  {b:<{col}}  {valor:>5.4f}  {barra}")

    print(f"\n  Leitura: barra mais longa = regulamentos semanticamente mais afastados.")
    print(f"  Centroide do regulamento = média dos centroides dos seus artigos.")
    print(f"  Distância coseno = 1 − similaridade. Próximo de 0 = regulamentos muito semelhantes.")


# ── Impressão — Nível 2 ───────────────────────────────────────────────────────

def _imprimir_silhouette_artigo(score: float) -> None:
    barra = _barra_visual(max(score, 0), 0, 1, largura=20)

    if np.isnan(score):
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


def _imprimir_coesao_artigo(df_coesao: pd.DataFrame) -> None:
    if df_coesao.empty:
        print("\n  Nenhum artigo dividido em múltiplos chunks — métrica não aplicável.")
        return

    minimo  = df_coesao["coesao_media"].min()
    maximo  = df_coesao["coesao_media"].max()
    media   = df_coesao["coesao_media"].mean()
    mediana = df_coesao["coesao_media"].median()
    minval  = df_coesao["coesao_media"].min()
    maxval  = df_coesao["coesao_media"].max()

    print(f"\n  Artigos avaliados : {len(df_coesao)}")
    print(f"  Média de coesão   : {media:.4f}")
    print(f"  Mediana           : {mediana:.4f}")
    print(f"  Mínimo / Máximo   : {minval:.4f} / {maxval:.4f}")
    print()

    col_doc = 28
    col_art = 20
    print(f"  {'Regulamento':<{col_doc}}  {'Artigo':<{col_art}}  {'Chunks':>6}  {'Coesão':>6}  Barra")
    print(f"  {'─'*col_doc}  {'─'*col_art}  {'──────':>6}  {'──────':>6}  {'─'*20}")

    for _, row in df_coesao.iterrows():
        doc   = _truncar(str(row["doc_titulo"]), col_doc)
        art   = _truncar(str(row["artigo_id"]),  col_art)
        n     = int(row["n_chunks"])
        valor = row["coesao_media"]
        barra = _barra_visual(valor, minimo, maximo)
        print(f"  {doc:<{col_doc}}  {art:<{col_art}}  {n:>6}  {valor:>6.4f}  {barra}")

    print(f"\n  Leitura: barra mais longa = chunks do artigo mais coesos entre si.")
    print(f"  Coesão = similaridade coseno média entre chunks do mesmo artigo.")
    print(f"  Valores > 0.7 indicam que os chunks representam bem o mesmo conteúdo.")


def _imprimir_distancia_artigo(df_dist: pd.DataFrame, top_n: int = 15) -> None:
    if df_dist.empty:
        print("\n  Sem dados suficientes.")
        return

    minimo  = df_dist["distancia_coseno"].min()
    maximo  = df_dist["distancia_coseno"].max()
    df_top  = df_dist.head(top_n)

    col = 22
    print(f"\n  (a mostrar os {len(df_top)} pares de artigos mais distantes por regulamento)\n")
    print(f"  {'Regulamento':<{col}}  {'Artigo A':<{col}}  {'Artigo B':<{col}}  {'Dist.':>5}  Barra")
    print(f"  {'─'*col}  {'─'*col}  {'─'*col}  {'─────':>5}  {'─'*20}")

    for _, row in df_top.iterrows():
        doc   = _truncar(str(row["doc_titulo"]), col)
        art_a = _truncar(str(row["artigo_a"]),   col)
        art_b = _truncar(str(row["artigo_b"]),   col)
        valor = row["distancia_coseno"]
        barra = _barra_visual(valor, minimo, maximo)
        print(f"  {doc:<{col}}  {art_a:<{col}}  {art_b:<{col}}  {valor:>5.4f}  {barra}")

    print(f"\n  Leitura: barra mais longa = artigos semanticamente mais afastados.")
    print(f"  Baseado em centroides de artigos — inclui artigos com chunk único.")
    print(f"  Distância coseno = 1 − similaridade. Próximo de 0 = artigos muito semelhantes.")


# ── Relatório ─────────────────────────────────────────────────────────────────

_REPORT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluate_report.txt")


class _Tee:
    """
    Stream que escreve simultaneamente para dois destinos (terminal + buffer).
    Usado com contextlib.redirect_stdout — não toca no builtins.print,
    evitando qualquer risco de recursão.
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


def imprimir_relatorio(df: pd.DataFrame) -> None:
    """
    Calcula e imprime o relatório completo em dois níveis.
    O output é escrito simultaneamente no terminal e em evaluate_report.txt,
    substituindo o ficheiro existente a cada execução.
    """
    import io
    import datetime
    from contextlib import redirect_stdout

    df     = _adicionar_chave_artigo(df)
    buffer = io.StringIO()

    with redirect_stdout(_Tee(sys.stdout, buffer)):

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"  Gerado em: {timestamp}")

        # ── Resumo ────────────────────────────────────────────────────────────
        _titulo_secao("RESUMO DO CORPUS")
        _subtitulo("Estatísticas gerais")
        resumo_corpus(df)

        # ════════════════════════════════════════════════════════════════════
        # NÍVEL 1 — REGULAMENTO
        # Unidade de análise: centroide de artigo (1 vector por artigo)
        # Inclui TODOS os artigos independentemente de truncated
        # ════════════════════════════════════════════════════════════════════
        _titulo_secao("NÍVEL 1 — AVALIAÇÃO POR REGULAMENTO")
        print(f"  Unidade de análise: centroide de cada artigo")
        print(f"  Inclui todos os artigos (com ou sem divisão em chunks)")

        _subtitulo("Silhouette Score  (separação entre regulamentos)")
        _imprimir_silhouette_regulamento(calcular_silhouette_regulamento(df))

        _subtitulo("Coesão intra-regulamento  (homogeneidade semântica dos artigos)")
        _imprimir_coesao_regulamento(calcular_coesao_intra_regulamento(df))

        _subtitulo("Distância inter-regulamento  (separação semântica entre pares)")
        _imprimir_distancia_regulamento(calcular_distancia_inter_regulamento(df))

        # ════════════════════════════════════════════════════════════════════
        # NÍVEL 2 — ARTIGO / verificação de chunking
        # Unidade de análise: chunks individuais
        # Inclui APENAS artigos com 2+ chunks (truncated=true)
        # ════════════════════════════════════════════════════════════════════
        _titulo_secao("NÍVEL 2 — VERIFICAÇÃO DE CHUNKING (por Artigo)")
        print(f"  Unidade de análise: chunks individuais")
        print(f"  Inclui apenas artigos com 2+ chunks (truncated=true)")

        _subtitulo("Silhouette Score  (coesão intra-artigo vs separação inter-artigo)")
        _imprimir_silhouette_artigo(calcular_silhouette_artigo(df))

        _subtitulo("Coesão intra-artigo  (chunks do mesmo artigo)")
        _imprimir_coesao_artigo(calcular_coesao_intra_artigo(df))

        _subtitulo("Distância inter-artigo  (pares mais distantes por regulamento)")
        _imprimir_distancia_artigo(calcular_distancia_inter_artigo(df))

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