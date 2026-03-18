"""
metadata_extractor.py
Extrai metadados do documento a partir do nome do ficheiro e do preâmbulo.

Responsabilidade única: produzir o dicionário document_info.
Não sabe nada sobre elementos da API nem sobre a estrutura do JSON final.
"""

import os
from utils.regex_patterns import YEAR_FILENAME, YEAR_PREAMBLE


class MetadataExtractor:

    @staticmethod
    def extract(filename: str, preamble: str = "") -> dict:
        """
        Devolve o dicionário document_info.

        Estratégia para o ano:
          1. Nome do ficheiro  (mais fiável — vem do sistema de gestão)
          2. Texto do preâmbulo (fallback)
          3. "N/A"             (último recurso)
        """
        doc_id = os.path.splitext(filename)[0]

        ano = "N/A"
        m = YEAR_FILENAME.search(filename)
        if m:
            ano = m.group(0)
        else:
            m = YEAR_PREAMBLE.search(preamble)
            if m:
                ano = m.group(1)

        return {
            "doc_id": doc_id,
            "escola": "GERAL",
            "ano":    ano,
            "status": "VIGENTE",
        }