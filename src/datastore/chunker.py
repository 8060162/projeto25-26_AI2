"""
chunker.py
----------
Chunking de tamanho fixo com overlap para artigos legislativos.

Estratégia:
  - A unidade semântica mínima é o artigo completo.
  - Se o conteúdo couber em CHUNK_SIZE → um único chunk, sem marcação.
  - Se não couber → dividido em chunks de CHUNK_SIZE com OVERLAP de
    caracteres; cada chunk é identificado com sufixo _part1, _part2, …
    O artigo completo fica sempre recuperável no JSON de origem via
    (source, artigo_id) — o retriever usa esse par para o fallback.

Parâmetros:
  CHUNK_SIZE   — tamanho máximo de cada chunk em caracteres (500–600).
  OVERLAP      — sobreposição em caracteres entre chunks consecutivos,
                 para preservar contexto na fronteira de corte.
"""

CHUNK_SIZE = 550
OVERLAP    = 80


def dividir_conteudo(conteudo: str) -> tuple[list[str], bool]:
    """
    Divide o conteúdo de um artigo em chunks de tamanho fixo.

    Args:
        conteudo: texto integral do artigo (campo `conteudo` do JSON).

    Returns:
        chunks    — lista de strings. Um elemento se o artigo couber
                    num único chunk; vários se for necessário dividir.
        truncated — False se o artigo não foi dividido (chunk único).
                    True  se foi dividido; nesse caso o retriever deve
                    recuperar o artigo completo a partir do JSON.
    """
    conteudo = conteudo.strip()
    if not conteudo:
        return [], False

    if len(conteudo) <= CHUNK_SIZE:
        return [conteudo], False

    chunks: list[str] = []
    start = 0
    while start < len(conteudo):
        end = start + CHUNK_SIZE
        chunks.append(conteudo[start:end])
        start = end - OVERLAP          # recua OVERLAP para o próximo chunk

    return chunks, True