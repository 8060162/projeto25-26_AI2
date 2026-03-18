"""
loader_protocol.py
Define o contrato que qualquer loader deve cumprir.
Permite trocar PDFLoader por DevLoader (ou qualquer outro) sem tocar na Pipeline.
"""

from typing import Protocol


class LoaderProtocol(Protocol):
    def load(self, file_path: str) -> list[dict]:
        """
        Recebe o caminho para um PDF e devolve uma lista ordenada de elementos.

        Cada elemento é um dicionário com exactamente três chaves:
            {
                "text":     str,  # conteúdo textual do bloco, já strip()
                "category": str,  # tipo Unstructured: Title, NarrativeText, …
                "page":     int   # página de origem, 1-based
            }
        """
        ...