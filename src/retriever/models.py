"""
models.py
---------
Estruturas de dados partilhadas entre retriever e generator.
Sem dependências internas — pode ser importado por qualquer módulo.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ArtigoContexto:
    """
    Artigo completo com metadados, pronto a enviar ao generator.

    Campos obrigatórios:
        artigo_id  — identificador único do artigo (ex: "artigo_12")
        conteudo   — texto completo do artigo
        source     — nome do ficheiro JSON de origem
        pagina     — página de origem no documento original

    Campos opcionais (None quando não disponíveis na fonte):
        artigo_titulo   — título do artigo, se extraível
        capitulo_titulo — título do capítulo em que o artigo se insere
        doc_numero      — número do documento regulamentar
    """
    artigo_id:       str
    conteudo:        str
    source:          str
    artigo_titulo:   Optional[str] = None
    capitulo_titulo: Optional[str] = None
    doc_numero:      Optional[str] = None
    pagina:          Optional[str] = None