"""
metadata_keys.py
----------------
Contrato formal das chaves de metadados armazenadas no ChromaDB.

Centraliza os nomes das chaves usadas em três pontos do sistema:
  - document_parser.py  (escrita via to_metadata)
  - ingest.py           (escrita de campos adicionais)
  - search.py           (leitura e mapeamento para ArtigoContexto)

REGRA: nenhum módulo deve escrever strings literais de chaves de metadados.
       Qualquer adição ou renomeação é feita exclusivamente aqui.
"""


class MetaKey:
    SOURCE:      str = "source"
    DOC_TITULO:  str = "doc_titulo"
    ART_TITULO:  str = "art_titulo"
    CAPITULO:    str = "capitulo"
    ARTIGO_ID:   str = "artigo_id"
    PAGINA:      str = "pagina"
    TRUNCATED:   str = "truncated"
    PART:        str = "part"