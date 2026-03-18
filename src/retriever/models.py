"""
models.py
---------
Estruturas de dados partilhadas entre retriever e generator.
Sem dependências internas — pode ser importado por qualquer módulo.
"""

from dataclasses import dataclass


@dataclass
class ArtigoContexto:
    """Artigo completo com metadados, pronto a enviar ao generator."""
    artigo_id:       str
    artigo_titulo:   str
    capitulo_titulo: str
    doc_numero:      str
    pagina:          str
    source:          str
    conteudo:        str