"""
pdf_indexer.py  (v2 — integra DocumentBuilder e relatório OCR)
--------------------------------------------------------------
Orquestra o processamento de um único PDF: carrega, faz parse,
extrai metadados e constrói o JSON final navegável.

Alterações v2:
  - Usa DocumentBuilder em vez de _build_document inline
  - Propaga o relatório OCR (se presente no primeiro elemento)
  - Conta elementos processados para o _meta do JSON
"""

import logging
import os
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from loader_protocol  import LoaderProtocol
from anchor_parser    import AnchorParser
from document_builder import DocumentBuilder

logger = logging.getLogger(__name__)


class PDFIndexer:

    def __init__(self, loader: LoaderProtocol):
        self._loader  = loader
        self._parser  = AnchorParser()
        self._builder = DocumentBuilder()

    def run(self, pdf_path: str) -> dict:
        """
        Processa um PDF e devolve o documento JSON estruturado e navegável.

        Returns:
            Documento JSON completo (ver document_builder.py para esquema).

        Raises:
            Exception: em caso de erro grave — capturado pelo batch_processor.
        """
        elements = self._loader.load(pdf_path)

        # Extrai o relatório OCR se presente (injectado pelo PDFLoader)
        ocr_report = elements[0].pop("_ocr_report", None) if elements else None

        parsed = self._parser.parse(elements)

        doc = self._builder.build(
            filename      = os.path.basename(pdf_path),
            parsed        = parsed,
            ocr_report    = ocr_report,
            element_count = len(elements),
        )

        logger.info(
            "PDFIndexer: '%s' → %d capítulos, %d artigos",
            os.path.basename(pdf_path),
            len(doc.get("estrutura", {})),
            doc.get("_meta", {}).get("extraction_info", {}).get("total_artigos", 0),
        )

        return doc