"""
benchmark/benchmark_models.py
------------------------------
Estruturas de dados do subsistema de benchmark.

RESPONSABILIDADE ÚNICA: definir os contratos de dados.
Sem dependências internas — pode ser importado por qualquer módulo
do subsistema sem risco de import circular ou acoplamento transitivo.

REGRA: nenhum outro módulo deste pacote define estruturas de dados.
       Qualquer adição ou alteração aos contratos é feita aqui.

Hierarquia de tipos:
  GoldId           — par (source, artigo_id) que identifica um artigo de forma
                     inequívoca no corpus. Usado como ground truth.
  QueryEntry       — uma pergunta de benchmark com os seus gold_ids e metadados.
  BenchmarkDataset — colecção de QueryEntry, com metadados do dataset.
  RetrievedItem    — um artigo devolvido pelo retriever para uma query.
  QueryResult      — resultado completo de uma query: retrieved + métricas.
  BenchmarkReport  — relatório agregado de uma execução completa.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ── Ground truth ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GoldId:
    """
    Identifica inequivocamente um artigo no corpus.

    Frozen para ser usável como elemento de set e chave de dict —
    a comparação por valor é intencional e necessária para as métricas.

    Campos:
        source:    nome do ficheiro JSON (ex: "regulamento_avaliacao_2024.json")
        artigo_id: identificador do artigo (ex: "ART_15")

    Nota: artigo_id isolado não é único no corpus — o mesmo "ART_1"
    pode existir em dois regulamentos diferentes. O par (source, artigo_id)
    é a chave canónica, alinhada com MetaKey.SOURCE + MetaKey.ARTIGO_ID.
    """
    source:    str
    artigo_id: str


# ── Dataset ───────────────────────────────────────────────────────────────────

@dataclass
class QueryEntry:
    """
    Uma pergunta de benchmark com o seu ground truth e metadados.

    Campos obrigatórios:
        id:       identificador único da query (ex: "q001")
        pergunta: texto da pergunta tal como será enviada ao retriever
        gold_ids: conjunto de GoldId que constituem a resposta correcta

    Campos opcionais (para análise segmentada dos resultados):
        tipo:        categoria da query — ver QueryTipo
        dificuldade: nível de dificuldade estimado — ver QueryDificuldade
        notas:       observações livres para uso humano (não processadas)
    """
    id:          str
    pergunta:    str
    gold_ids:    list[GoldId]
    tipo:        str  = "directa"      # ver QueryTipo
    dificuldade: str  = "facil"        # ver QueryDificuldade
    notas:       str  = ""


class QueryTipo:
    """Valores válidos para QueryEntry.tipo."""
    DIRECTA    = "directa"     # resposta num único artigo, referência explícita
    TEMATICA   = "tematica"    # conceito espalhado por vários artigos
    MULTI      = "multi"       # requer combinar dois ou mais artigos
    NEGATIVA   = "negativa"    # a resposta não existe no corpus


class QueryDificuldade:
    """Valores válidos para QueryEntry.dificuldade."""
    FACIL   = "facil"    # correspondência lexical directa entre query e artigo
    MEDIO   = "medio"    # requer inferência semântica moderada
    DIFICIL = "dificil"  # paráfrase significativa ou raciocínio sobre múltiplos artigos


@dataclass
class BenchmarkDataset:
    """
    Colecção de QueryEntry com metadados do dataset.

    Campos:
        versao:    string de versão semântica (ex: "1.0.0")
        descricao: descrição livre do dataset
        queries:   lista de QueryEntry

    A versão é usada no relatório para rastrear qual dataset
    produziu quais resultados — essencial quando o dataset evolui.
    """
    versao:    str
    descricao: str
    queries:   list[QueryEntry] = field(default_factory=list)


# ── Resultados ────────────────────────────────────────────────────────────────

@dataclass
class RetrievedItem:
    """
    Um artigo devolvido pelo retriever para uma query específica.

    Campos:
        rank:      posição na lista de resultados (1-based)
        source:    nome do ficheiro JSON de origem
        artigo_id: identificador do artigo
    """
    rank:      int
    source:    str
    artigo_id: str

    def to_gold_id(self) -> GoldId:
        """Converte para GoldId para comparação com o ground truth."""
        return GoldId(source=self.source, artigo_id=self.artigo_id)


@dataclass
class QueryResult:
    """
    Resultado completo de uma query de benchmark.

    Campos:
        query_id:   identificador da QueryEntry correspondente
        pergunta:   texto da query (desnormalizado para leitura do relatório)
        gold_ids:   ground truth desta query
        retrieved:  lista de RetrievedItem ordenada por rank
        k:          valor de K usado nesta execução

    Métricas calculadas (preenchidas por BenchmarkMetrics):
        hit:        True se pelo menos um gold_id está nos retrieved
        reciprocal_rank: 1/posição do primeiro gold_id (0 se não encontrado)
        recall:     fracção dos gold_ids recuperados

    Campos de diagnóstico:
        erro: mensagem de erro se o retriever falhou (None se OK)
    """
    query_id:        str
    pergunta:        str
    gold_ids:        list[GoldId]
    retrieved:       list[RetrievedItem]
    k:               int
    hit:             bool          = False
    reciprocal_rank: float         = 0.0
    recall:          float         = 0.0
    erro:            Optional[str] = None


@dataclass
class MetricasAgregadas:
    """
    Métricas agregadas sobre um conjunto de QueryResult.

    Campos:
        k:          valor de K a que estas métricas se referem
        n_queries:  número de queries avaliadas
        hit_rate:   Hit@K = fracção de queries com pelo menos um gold_id recuperado
        mrr:        Mean Reciprocal Rank
        recall_medio: Recall@K médio sobre todas as queries
        n_erros:    número de queries que falharam com erro
    """
    k:             int
    n_queries:     int
    hit_rate:      float
    mrr:           float
    recall_medio:  float
    n_erros:       int = 0


@dataclass
class MetricasPorTipo:
    """
    Métricas agregadas segmentadas por tipo de query.

    Permite identificar se o sistema falha selectivamente
    em queries temáticas, multi-artigo, ou negativas.
    """
    tipo:         str
    n_queries:    int
    hit_rate:     float
    mrr:          float
    recall_medio: float


@dataclass
class BenchmarkReport:
    """
    Relatório completo de uma execução de benchmark.

    Campos de rastreabilidade:
        timestamp:         data/hora da execução (ISO 8601)
        dataset_versao:    versão do dataset usado
        n_results_testados: lista de K testados
        configuracao:      snapshot das settings relevantes (N_RESULTS, QUERY_FETCH, etc.)

    Resultados:
        resultados_por_k:  MetricasAgregadas para cada K testado
        por_tipo:          MetricasPorTipo para cada tipo de query, para o K principal
        query_results:     lista completa de QueryResult (diagnóstico por query)
    """
    timestamp:          str
    dataset_versao:     str
    n_results_testados: list[int]
    configuracao:       dict

    resultados_por_k:   list[MetricasAgregadas]  = field(default_factory=list)
    por_tipo:           list[MetricasPorTipo]     = field(default_factory=list)
    query_results:      list[QueryResult]         = field(default_factory=list)