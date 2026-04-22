"""
benchmark/generate_dataset.py
------------------------------
Responsabilidade única: gerar o ficheiro benchmark.json a partir dos
JSON processados em data/processed/, usando o modelo Ollama local
para formular perguntas.

ARQUITECTURA — zero acoplamento ao pipeline principal:
  Este script acede aos JSONs de data/processed/ directamente,
  sem passar pelo ChromaDB nem pelo retriever. Só precisa de saber
  onde estão os ficheiros e como está estruturado um artigo.
  O único ponto de contacto com o projecto é a leitura de
  settings.OLLAMA_MODEL e settings.JSON_FOLDER.

POR QUE SÍNCRONO (sem asyncio):
  O Ollama corre localmente num processo único. Chamadas paralelas
  não aumentam a velocidade — competem pelo mesmo hardware (GPU/CPU)
  e podem ser mais lentas por troca de contexto. O loop sequencial
  simples é o padrão correcto para um LLM local.

ESTIMATIVA DE TEMPO:
  qwen2.5:7b gera ~20 tok/s. Cada pergunta requer ~80 tokens de output.
  60 queries (15 × 4 tipos) × ~4s cada ≈ 4 minutos no total.
  O progresso é impresso a cada query para acompanhar a execução.

FLUXO:
  1. Descobrir todos os artigos em data/processed/*.json
  2. Amostrar artigos por tipo de query (estratificado por regulamento)
  3. Para cada artigo, chamar Ollama e obter uma pergunta
  4. Construir e guardar o BenchmarkDataset

MODO DE EXECUÇÃO:
  # Gerar dataset completo (15 queries por tipo)
  python generate_dataset.py

  # Gerar apenas N perguntas por tipo
  python generate_dataset.py --por-tipo 10

  # Usar um modelo diferente do definido em settings
  python generate_dataset.py --modelo llama3.2:3b

  # Forçar regeneração mesmo que benchmark.json já exista
  python generate_dataset.py --forcar

  # Pré-visualizar artigos amostrados sem chamar o LLM
  python generate_dataset.py --dry-run

REVISÃO HUMANA:
  O dataset gerado deve ser revisto manualmente antes de ser usado.
  Perguntas ambíguas, incorrectas ou que testem o mesmo conceito
  devem ser editadas directamente no benchmark.json.
  A versão do dataset deve ser incrementada após cada revisão.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
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
    GoldId,
    QueryDificuldade,
    QueryEntry,
    QueryTipo,
)
from benchmark_io import guardar_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Constantes ────────────────────────────────────────────────────────────────

# Comprimento mínimo do conteúdo para um artigo ser elegível
_MIN_CONTEUDO_CHARS = 80

# Versão inicial do dataset gerado
_VERSAO_INICIAL = "1.0.0"

# Máximo de caracteres do conteúdo enviado ao LLM por artigo.
# Evita exceder o context window de modelos mais pequenos.
_MAX_CONTEUDO_CHARS = 1200


# ── Estrutura interna de trabalho ─────────────────────────────────────────────

class _ArtigoElegivel:
    """Artigo candidato a geração de query. Apenas os campos necessários."""
    __slots__ = ("source", "artigo_id", "art_titulo", "cap_titulo", "conteudo", "doc_titulo")

    def __init__(self, source, artigo_id, art_titulo, cap_titulo, conteudo, doc_titulo):
        self.source     = source
        self.artigo_id  = artigo_id
        self.art_titulo = art_titulo
        self.cap_titulo = cap_titulo
        self.conteudo   = conteudo
        self.doc_titulo = doc_titulo


# ── Leitura dos artigos ───────────────────────────────────────────────────────

def _carregar_artigos(json_folder: str) -> list[_ArtigoElegivel]:
    """
    Lê todos os JSONs processados e devolve a lista de artigos elegíveis.

    Artigos com conteúdo vazio ou abaixo do mínimo são ignorados.
    Não usa document_parser — leitura directa para manter zero acoplamento.
    """
    artigos: list[_ArtigoElegivel] = []
    pasta = Path(json_folder)

    for json_path in sorted(pasta.glob("*.json")):
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Ignorado '%s': %s", json_path.name, e)
            continue

        info       = data.get("document_info", {})
        doc_titulo = info.get("titulo", info.get("title", info.get("nome", json_path.stem)))
        estrutura  = data.get("estrutura", {})

        for cap_data in estrutura.values():
            cap_titulo = cap_data.get("titulo", "")
            for art_id, art_data in cap_data.get("artigos", {}).items():
                conteudo = (art_data.get("conteudo") or "").strip()
                if len(conteudo) < _MIN_CONTEUDO_CHARS:
                    continue
                artigos.append(_ArtigoElegivel(
                    source=json_path.name,
                    artigo_id=art_id,
                    art_titulo=art_data.get("titulo", ""),
                    cap_titulo=cap_titulo,
                    conteudo=conteudo,
                    doc_titulo=doc_titulo,
                ))

    logger.info("Artigos elegíveis encontrados: %d", len(artigos))
    return artigos


# ── Amostragem ────────────────────────────────────────────────────────────────

def _amostrar_estratificado(
    artigos:    list[_ArtigoElegivel],
    n_por_tipo: int,
    seed:       int = 42,
) -> dict[str, list[_ArtigoElegivel]]:
    """
    Amostra artigos estratificados por regulamento para evitar viés.

    Documentos com mais artigos não dominam o dataset — cada regulamento
    contribui no máximo n_por_tipo // n_regulamentos artigos.

    Returns:
        dict {tipo → lista de artigos amostrados}
    """
    rng = random.Random(seed)

    por_reg: dict[str, list[_ArtigoElegivel]] = {}
    for art in artigos:
        por_reg.setdefault(art.source, []).append(art)

    n_regs    = len(por_reg)
    por_reg_n = max(1, n_por_tipo // n_regs) if n_regs > 0 else n_por_tipo

    amostras_base: list[_ArtigoElegivel] = []
    for reg_artigos in por_reg.values():
        k = min(por_reg_n, len(reg_artigos))
        amostras_base.extend(rng.sample(reg_artigos, k))

    # Completar até n_por_tipo se a estratificação ficar curta
    if len(amostras_base) < n_por_tipo:
        em_uso    = set(id(a) for a in amostras_base)
        restantes = [a for a in artigos if id(a) not in em_uso]
        faltam    = n_por_tipo - len(amostras_base)
        amostras_base.extend(rng.sample(restantes, min(faltam, len(restantes))))

    # Queries multi-artigo: pares do mesmo regulamento
    pares: list[_ArtigoElegivel] = []
    for reg_artigos in por_reg.values():
        if len(reg_artigos) >= 2:
            k       = min(por_reg_n, len(reg_artigos) // 2)
            amostra = rng.sample(reg_artigos, k * 2)
            pares.extend(amostra)
        if len(pares) >= n_por_tipo * 2:
            break

    return {
        QueryTipo.DIRECTA:  amostras_base[:n_por_tipo],
        QueryTipo.TEMATICA: rng.sample(amostras_base, min(n_por_tipo, len(amostras_base))),
        QueryTipo.MULTI:    pares[:n_por_tipo * 2],  # processado em pares
        QueryTipo.NEGATIVA: [],                       # não requer artigo base
    }


# ── Prompts ───────────────────────────────────────────────────────────────────

_PROMPT_DIRECTA = """\
Tens acesso ao seguinte artigo de um regulamento académico do P.PORTO:

Documento: {doc_titulo}
Capítulo: {cap_titulo}
{artigo_id} — {art_titulo}

{conteudo}

Formula UMA pergunta directa e específica que:
- Só pode ser respondida consultando este artigo
- Um estudante real poderia fazer
- Não menciona o número do artigo nem o título do regulamento
- Está escrita em português de Portugal

Responde APENAS com a pergunta, sem introdução nem explicação."""

_PROMPT_TEMATICA = """\
Tens acesso ao seguinte artigo de um regulamento académico do P.PORTO:

Documento: {doc_titulo}
Capítulo: {cap_titulo}
{artigo_id} — {art_titulo}

{conteudo}

Formula UMA pergunta temática que:
- Aborda o tema geral do artigo sem citar detalhes específicos
- Poderia ser respondida por este artigo ou por artigos relacionados
- Um estudante que não conhece o regulamento poderia fazer
- Não menciona artigos, capítulos, nem regulamentos por nome

Responde APENAS com a pergunta, sem introdução nem explicação."""

_PROMPT_MULTI = """\
Tens acesso a dois artigos de um regulamento académico do P.PORTO:

--- ARTIGO A ---
Documento: {doc_titulo_a}
{artigo_id_a} — {art_titulo_a}
{conteudo_a}

--- ARTIGO B ---
Documento: {doc_titulo_b}
{artigo_id_b} — {art_titulo_b}
{conteudo_b}

Formula UMA pergunta que:
- Só pode ser respondida completamente consultando AMBOS os artigos
- Um estudante real poderia fazer
- Não menciona números de artigos nem títulos de regulamentos
- Está escrita em português de Portugal

Responde APENAS com a pergunta, sem introdução nem explicação."""

_PROMPT_NEGATIVA = """\
És um estudante do P.PORTO. Formula UMA pergunta sobre a vida académica que:
- NÃO possa ser respondida por nenhum regulamento (ex: questões pessoais,
  opiniões, previsões, questões fora do âmbito académico)
- Pareça legítima mas seja impossível de responder com base em regulamentos
- Esteja escrita em português de Portugal
- Não seja obviamente absurda

Exemplos do que NÃO queres: "Qual é o melhor restaurante perto do campus?"
Exemplos do que QUERES: "Qual será a tendência de notas nos próximos 5 anos?"

Responde APENAS com a pergunta, sem introdução nem explicação."""


# ── Chamada ao Ollama ─────────────────────────────────────────────────────────

def _chamar_ollama(prompt: str, modelo: str) -> str | None:
    """
    Chama o Ollama localmente e devolve o texto gerado.

    Usa o cliente ollama directamente — sem HTTP, sem asyncio.
    O Ollama é síncrono por natureza: uma chamada de cada vez é o
    padrão correcto para um modelo local.

    Temperature 0.7 para geração de perguntas — mais variação
    criativa do que o RAG (0.1), que prioriza precisão factual.

    Devolve None em caso de erro para não interromper o batch.
    """
    try:
        import ollama as _ollama
        response = _ollama.chat(
            model=modelo,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.7, "num_ctx": 4096},
        )
        texto = response["message"]["content"].strip()
        # Remover aspas que alguns modelos adicionam à pergunta gerada
        return texto.strip('"').strip("'").strip()
    except Exception as e:
        logger.warning("Erro na chamada Ollama: %s", e)
        return None


# ── Geração por tipo ──────────────────────────────────────────────────────────

def _gerar_directas(
    artigos:   list[_ArtigoElegivel],
    modelo:    str,
    offset_id: int,
) -> list[QueryEntry]:
    """Gera queries directas — uma por artigo, sequencialmente."""
    entries: list[QueryEntry] = []
    total = len(artigos)

    for i, art in enumerate(artigos, 1):
        logger.info("  directa %d/%d — %s", i, total, art.artigo_id)
        prompt = _PROMPT_DIRECTA.format(
            doc_titulo=art.doc_titulo,
            cap_titulo=art.cap_titulo,
            artigo_id=art.artigo_id,
            art_titulo=art.art_titulo,
            conteudo=art.conteudo[:_MAX_CONTEUDO_CHARS],
        )
        pergunta = _chamar_ollama(prompt, modelo)
        if not pergunta:
            logger.warning("    ignorada (Ollama não respondeu)")
            continue
        entries.append(QueryEntry(
            id=f"q{offset_id + len(entries) + 1:04d}",
            pergunta=pergunta,
            gold_ids=[GoldId(source=art.source, artigo_id=art.artigo_id)],
            tipo=QueryTipo.DIRECTA,
            dificuldade=QueryDificuldade.FACIL,
        ))

    return entries


def _gerar_tematicas(
    artigos:   list[_ArtigoElegivel],
    modelo:    str,
    offset_id: int,
) -> list[QueryEntry]:
    """Gera queries temáticas — uma por artigo, dificuldade média."""
    entries: list[QueryEntry] = []
    total = len(artigos)

    for i, art in enumerate(artigos, 1):
        logger.info("  temática %d/%d — %s", i, total, art.artigo_id)
        prompt = _PROMPT_TEMATICA.format(
            doc_titulo=art.doc_titulo,
            cap_titulo=art.cap_titulo,
            artigo_id=art.artigo_id,
            art_titulo=art.art_titulo,
            conteudo=art.conteudo[:_MAX_CONTEUDO_CHARS],
        )
        pergunta = _chamar_ollama(prompt, modelo)
        if not pergunta:
            logger.warning("    ignorada (Ollama não respondeu)")
            continue
        entries.append(QueryEntry(
            id=f"q{offset_id + len(entries) + 1:04d}",
            pergunta=pergunta,
            gold_ids=[GoldId(source=art.source, artigo_id=art.artigo_id)],
            tipo=QueryTipo.TEMATICA,
            dificuldade=QueryDificuldade.MEDIO,
        ))

    return entries


def _gerar_multi(
    artigos:   list[_ArtigoElegivel],
    modelo:    str,
    offset_id: int,
) -> list[QueryEntry]:
    """
    Gera queries multi-artigo — uma por par de artigos adjacentes.
    artigos deve ter comprimento par: [art_a, art_b, art_c, art_d, ...]
    """
    entries: list[QueryEntry] = []
    pares   = [(artigos[i], artigos[i + 1]) for i in range(0, len(artigos) - 1, 2)]
    total   = len(pares)

    for i, (a, b) in enumerate(pares, 1):
        logger.info("  multi %d/%d — %s + %s", i, total, a.artigo_id, b.artigo_id)
        max_chars = _MAX_CONTEUDO_CHARS // 2
        prompt = _PROMPT_MULTI.format(
            doc_titulo_a=a.doc_titulo, artigo_id_a=a.artigo_id,
            art_titulo_a=a.art_titulo, conteudo_a=a.conteudo[:max_chars],
            doc_titulo_b=b.doc_titulo, artigo_id_b=b.artigo_id,
            art_titulo_b=b.art_titulo, conteudo_b=b.conteudo[:max_chars],
        )
        pergunta = _chamar_ollama(prompt, modelo)
        if not pergunta:
            logger.warning("    ignorada (Ollama não respondeu)")
            continue
        entries.append(QueryEntry(
            id=f"q{offset_id + len(entries) + 1:04d}",
            pergunta=pergunta,
            gold_ids=[
                GoldId(source=a.source, artigo_id=a.artigo_id),
                GoldId(source=b.source, artigo_id=b.artigo_id),
            ],
            tipo=QueryTipo.MULTI,
            dificuldade=QueryDificuldade.DIFICIL,
        ))

    return entries


def _gerar_negativas(
    n:         int,
    modelo:    str,
    offset_id: int,
) -> list[QueryEntry]:
    """Gera n queries negativas — sem gold_ids."""
    entries: list[QueryEntry] = []

    for i in range(1, n + 1):
        logger.info("  negativa %d/%d", i, n)
        pergunta = _chamar_ollama(_PROMPT_NEGATIVA, modelo)
        if not pergunta:
            logger.warning("    ignorada (Ollama não respondeu)")
            continue
        entries.append(QueryEntry(
            id=f"q{offset_id + len(entries) + 1:04d}",
            pergunta=pergunta,
            gold_ids=[],
            tipo=QueryTipo.NEGATIVA,
            dificuldade=QueryDificuldade.FACIL,
            notas="Verificar manualmente: a resposta não deve existir no corpus.",
        ))

    return entries


# ── Orquestrador ──────────────────────────────────────────────────────────────

def gerar_dataset(
    json_folder: str,
    n_por_tipo:  int,
    output_path: str,
    modelo:      str,
    dry_run:     bool = False,
) -> BenchmarkDataset:
    """
    Orquestra a geração completa do dataset de forma sequencial.

    Gera os quatro tipos de query um a um, com logging de progresso
    a cada chamada ao LLM. O dataset é guardado assim que concluído.

    Args:
        json_folder: pasta com os JSONs processados
        n_por_tipo:  número de queries a gerar por tipo
        output_path: caminho de saída do benchmark.json
        modelo:      nome do modelo Ollama (ex: "qwen2.5:7b")
        dry_run:     se True, mostra amostras sem chamar o LLM

    Returns:
        BenchmarkDataset gerado (ou vazio em dry_run).
    """
    artigos  = _carregar_artigos(json_folder)
    amostras = _amostrar_estratificado(artigos, n_por_tipo)

    if dry_run:
        logger.info("=== DRY RUN — nenhuma chamada ao Ollama ===")
        for tipo, lista in amostras.items():
            logger.info("  %s: %d artigos amostrados", tipo, len(lista))
        return BenchmarkDataset(versao=_VERSAO_INICIAL, descricao="dry-run", queries=[])

    n_estimado = (
        len(amostras[QueryTipo.DIRECTA])
        + len(amostras[QueryTipo.TEMATICA])
        + len(amostras[QueryTipo.MULTI]) // 2
        + n_por_tipo
    )
    logger.info(
        "A gerar ~%d queries com modelo '%s' (sequencial)...",
        n_estimado, modelo,
    )

    directas  = _gerar_directas( amostras[QueryTipo.DIRECTA],  modelo, offset_id=0)
    tematicas = _gerar_tematicas(amostras[QueryTipo.TEMATICA], modelo, offset_id=len(directas))
    multi     = _gerar_multi(    amostras[QueryTipo.MULTI],    modelo, offset_id=len(directas) + len(tematicas))
    negativas = _gerar_negativas(n_por_tipo, modelo,           offset_id=len(directas) + len(tematicas) + len(multi))

    todas = directas + tematicas + multi + negativas

    # Re-numerar IDs de forma contínua após possíveis falhas do LLM
    for i, q in enumerate(todas):
        q.id = f"q{i + 1:04d}"

    dataset = BenchmarkDataset(
        versao=_VERSAO_INICIAL,
        descricao=(
            f"Dataset gerado com '{modelo}' a partir de '{Path(json_folder).name}'. "
            f"REQUER REVISÃO HUMANA antes de ser usado como benchmark oficial."
        ),
        queries=todas,
    )

    guardar_dataset(dataset, output_path)

    logger.info(
        "Concluído: %d queries (%d directas, %d temáticas, %d multi, %d negativas)",
        len(todas), len(directas), len(tematicas), len(multi), len(negativas),
    )
    return dataset


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Gera o dataset de benchmark usando o Ollama local. "
            "O modelo e a pasta de JSONs são lidos de settings.py por omissão."
        )
    )
    parser.add_argument(
        "--json-folder", default=None,
        help="Pasta com os JSONs processados (default: settings.JSON_FOLDER)",
    )
    parser.add_argument(
        "--output", default="benchmark/benchmark.json",
        help="Caminho de saída do dataset (default: benchmark/benchmark.json)",
    )
    parser.add_argument(
        "--por-tipo", type=int, default=15,
        help="Número de queries a gerar por tipo (default: 15)",
    )
    parser.add_argument(
        "--modelo", default=None,
        help="Modelo Ollama a usar (default: settings.OLLAMA_MODEL)",
    )
    parser.add_argument(
        "--forcar", action="store_true",
        help="Regenerar mesmo que benchmark.json já exista",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Mostrar artigos amostrados sem chamar o Ollama",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # Resolver json_folder e modelo: argumento CLI > settings > erro
    try:
        from settings import JSON_FOLDER, OLLAMA_MODEL
        json_folder = args.json_folder or JSON_FOLDER
        modelo      = args.modelo      or OLLAMA_MODEL
    except ImportError:
        if not args.json_folder or not args.modelo:
            logger.error(
                "Não foi possível importar settings.py. "
                "Usa --json-folder e --modelo para definir os valores manualmente."
            )
            sys.exit(1)
        json_folder = args.json_folder
        modelo      = args.modelo

    output_path = Path(args.output)
    if output_path.exists() and not args.forcar and not args.dry_run:
        logger.warning(
            "Dataset já existe em '%s'. Usa --forcar para regenerar.", output_path
        )
        sys.exit(0)

    gerar_dataset(
        json_folder=json_folder,
        n_por_tipo=args.por_tipo,
        output_path=str(output_path),
        modelo=modelo,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()