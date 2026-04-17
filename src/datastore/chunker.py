"""
chunker.py  [Abordagem B — Conteúdo Puro]
------------------------------------------
Responsabilidade única: dividir o conteúdo de um artigo em chunks
com conteúdo semântico puro, sem cabeçalho contextual.

Toda a rastreabilidade (ficheiro, capítulo, artigo, página) é da
responsabilidade dos metadados no ChromaDB — ver Artigo.to_metadata()
em document_parser.py.

Constantes de chunking importadas de settings.py — não redefinir aqui.

ALTERAÇÃO (refactor): a assinatura de dividir_em_chunks foi simplificada
para receber apenas `conteudo: str`. Os parâmetros filename, cap_titulo,
art_id e art_titulo não eram utilizados internamente — a sua presença
criava acoplamento desnecessário com a estrutura do dataclass Artigo e
tornava a função desonesta sobre as suas dependências reais.
O chamador (ingest.py) passa agora apenas artigo.conteudo.
"""

import re
from langchain_text_splitters import RecursiveCharacterTextSplitter

from settings import CHUNK_TARGET, CHUNK_OVERLAP, CHUNK_MAX

# ── Fallback genérico para artigos sem estrutura numerada clara ───────────────

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_TARGET,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)

# ── Regex para fronteiras semânticas na legislação portuguesa ─────────────────

_PATTERN_SEMANTICO = re.compile(
    r'(?=\n(?:'
    r'\d+\.\d+\.\s'     # Ex: "2.1. "
    r'|\d+\.\s'         # Ex: "1. "
    r'|[a-z]\)\s'       # Ex: "a) "
    r'|[ivxlIVXL]+\)\s' # Ex: "i) "
    r'))'
)


def dividir_em_chunks(conteudo: str) -> list[str]:
    """
    Divide o conteúdo de um artigo em chunks semânticos seguindo a
    estratégia de 3 zonas:

      Zona 1+2 — artigo completo cabe no limite → devolve um único chunk.
      Zona 3   — divisão estrutural por padrão semântico (alíneas, números).
                 Se uma alínea isolada exceder CHUNK_TARGET, aplica fallback
                 com RecursiveCharacterTextSplitter.

    Args:
        conteudo: texto integral do artigo, já strip().

    Returns:
        Lista de strings com o conteúdo dividido. Nunca vazia se conteudo
        não for vazio.
    """
    conteudo = conteudo.strip()

    # Zona 1 e 2: artigo completo cabe no limite
    if len(conteudo) <= CHUNK_MAX:
        return [conteudo]

    # Zona 3: divisão estrutural
    partes = _PATTERN_SEMANTICO.split(conteudo)
    chunks: list[str] = []
    buffer = ""

    for parte in partes:
        if not parte.strip():
            continue

        if len(buffer) + len(parte) <= CHUNK_TARGET:
            buffer += parte
        else:
            if buffer:
                chunks.append(buffer.strip())

            # Alínea maior que CHUNK_TARGET → fallback genérico
            if len(parte) > CHUNK_TARGET:
                for frag in _splitter.split_text(parte):
                    chunks.append(frag.strip())
                buffer = ""
            else:
                buffer = parte

    if buffer:
        chunks.append(buffer.strip())

    return chunks