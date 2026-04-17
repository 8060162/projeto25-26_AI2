"""
metadata_keys.py
----------------
Contrato formal das chaves de metadados armazenadas no ChromaDB.

Centraliza os nomes das chaves usadas em três pontos do sistema:
  - document_parser.py  (escrita via to_metadata)
  - ingest.py           (escrita de campos adicionais)
  - search.py           (leitura e mapeamento para ArtigoContexto)
  - evaluate.py         (leitura de métricas)

REGRA: nenhum módulo deve escrever strings literais de chaves de metadados.
       Qualquer adição ou renomeação é feita exclusivamente aqui.

ALTERAÇÃO (refactor): adicionada a função is_truncated() como Single Source
of Truth para interpretar o campo 'truncated'. O ChromaDB serializa
metadados como strings, pelo que "true"/"false" e True/False são ambos
possíveis. Centralizar esta lógica evita que search.py e evaluate.py
divirjam silenciosamente se o formato mudar.
"""


class MetaKey:
    SOURCE:     str = "source"
    DOC_TITULO: str = "doc_titulo"
    ART_TITULO: str = "art_titulo"
    CAPITULO:   str = "capitulo"
    ARTIGO_ID:  str = "artigo_id"
    PAGINA:     str = "pagina"
    TRUNCATED:  str = "truncated"
    PART:       str = "part"


def is_truncated(meta: dict) -> bool:
    """
    Interpreta o campo 'truncated' dos metadados do ChromaDB de forma
    robusta, independentemente de estar serializado como bool ou string.

    Esta é a única função que deve ser usada para ler este campo.
    Usada em search.py e evaluate.py — evita divergência entre consumidores.

    Args:
        meta: dicionário de metadados do ChromaDB.

    Returns:
        True se o artigo foi dividido em múltiplos chunks, False caso contrário.
    """
    val = meta.get(MetaKey.TRUNCATED, False)
    if isinstance(val, bool):
        return val
    return str(val).lower() == "true"