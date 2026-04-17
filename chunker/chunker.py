"""
chunker.py
----------
Responsabilidade única: dividir o conteúdo de um artigo em chunks
com cabeçalho contextual (Context-Augmented Indexing).

Constantes importadas de settings.py — não redefinir aqui.
O separador CHUNK_HEADER_SEP é partilhado com search.py,
garantindo que escrita e leitura usam sempre o mesmo valor.
"""

import re
from langchain_text_splitters import RecursiveCharacterTextSplitter

from settings import CHUNK_TARGET, CHUNK_OVERLAP, CHUNK_MAX, CHUNK_HEADER_SEP

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


def construir_cabecalho(filename: str, cap_titulo: str, art_id: str, art_titulo: str) -> str:
    """
    Cria o prefixo contextual para o embedding (Context-Augmented Indexing).

    O separador CHUNK_HEADER_SEP termina o cabeçalho — search.py usa
    a mesma constante para o remover, mantendo os dois lados sincronizados.
    """
    return f"FICHEIRO: {filename} | CAP: {cap_titulo} | ART: {art_id} - {art_titulo}{CHUNK_HEADER_SEP}"


def dividir_em_chunks(
    filename: str,
    cap_titulo: str,
    art_id: str,
    art_titulo: str,
    conteudo: str,
) -> list[str]:
    """
    Divide o artigo em chunks seguindo a estratégia de 3 zonas:

      Zona 1+2 — artigo completo cabe no limite → devolve um único chunk.
      Zona 3   — divisão estrutural por padrão semântico (alíneas, números).
                 Se uma alínea isolada exceder CHUNK_TARGET, aplica fallback
                 com RecursiveCharacterTextSplitter.
    """
    conteudo  = conteudo.strip()
    cabecalho = construir_cabecalho(filename, cap_titulo, art_id, art_titulo)

    if len(conteudo) <= CHUNK_MAX:
        return [cabecalho + conteudo]

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
                chunks.append(cabecalho + buffer.strip())

            if len(parte) > CHUNK_TARGET:
                for frag in _splitter.split_text(parte):
                    chunks.append(cabecalho + frag.strip())
                buffer = ""
            else:
                buffer = parte

    if buffer:
        chunks.append(cabecalho + buffer.strip())

    return chunks