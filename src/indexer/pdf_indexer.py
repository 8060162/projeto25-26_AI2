"""
pdf_indexer.py
Orquestra o processamento de um único PDF: carrega, faz parse e junta metadados.

É o único módulo que conhece todos os colaboradores (loader, parser, extractor).
Recebe o loader por injecção de dependência — permite trocar PDFLoader (produção)
por DevLoader (desenvolvimento) sem alterar código.
"""

import os
import sys
from pathlib import Path

# Garante que src/ está no path independentemente de como o módulo é invocado
_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from indexer.loader_protocol  import LoaderProtocol
from indexer.anchor_parser    import AnchorParser
from indexer.metadata_extractor import MetadataExtractor


class PDFIndexer:

    def __init__(self, loader: LoaderProtocol):
        self._loader = loader
        self._parser = AnchorParser()

    def run(self, pdf_path: str) -> dict:
        """
        Processa um PDF e devolve o documento estruturado completo:

            {
                "document_info": { … },
                "preambulo":     str,
                "estrutura":     { … }
            }

        Lança excepção em caso de erro — é responsabilidade do
        batch_processor capturá-la e registá-la no relatório.
        """
        filename = os.path.basename(pdf_path)
        elements = self._loader.load(pdf_path)
        parsed   = self._parser.parse(elements)

        return {
            "document_info": MetadataExtractor.extract(
                filename=filename,
                preamble=parsed["preambulo"],
            ),
            "preambulo": parsed["preambulo"],
            "estrutura": parsed["estrutura"],
        }