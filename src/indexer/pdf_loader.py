"""
pdf_loader.py
Implementação de LoaderProtocol que chama a Unstructured Platform API.

Responsabilidade única: enviar o PDF à API e devolver a lista normalizada
de elementos. Não faz parsing, não filtra categorias, não toca em metadados.
"""

import os

import requests

from utils.element_utils import normalise_elements


class PDFLoader:
    """Carrega um PDF via Unstructured Platform API."""

    _API_URL = "https://api.unstructuredapp.io/general/v0/general"

    def __init__(self, api_key: str | None = None):
        key = api_key or os.environ.get("UNSTRUCTURED_API_KEY", "")
        if not key:
            raise ValueError(
                "UNSTRUCTURED_API_KEY não definida. "
                "Passa api_key= ao construtor ou define a variável de ambiente."
            )
        self._api_key = key

    def load(self, file_path: str) -> list[dict]:
        """
        Envia o PDF e devolve a lista de elementos normalizada.

        Raises:
            requests.HTTPError: Se a API responder com código de erro HTTP.
            ValueError: Se a API devolver um formato de resposta inesperado.
        """
        with open(file_path, "rb") as fh:
            response = requests.post(
                self._API_URL,
                headers={"unstructured-api-key": self._api_key},
                files={"files": (os.path.basename(file_path), fh, "application/pdf")},
                data={
                    "strategy":            "hi_res",
                    "languages":           ["por"],
                    "include_page_breaks": "true",
                    "output_format":       "application/json",
                },
                timeout=300,
            )

        response.raise_for_status()
        return self._parse_response(response.json(), file_path)

    # ── privado ───────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_response(raw: object, file_path: str) -> list[dict]:
        """
        Valida o tipo da resposta antes de normalizar.

        A API deve devolver sempre uma lista. Um dicionário com status 200
        indica geralmente um erro semântico não sinalizado por HTTP.

        Raises:
            ValueError: Se a resposta não for uma lista.
        """
        if not isinstance(raw, list):
            raise ValueError(
                f"[PDFLoader] Resposta inesperada da API ao processar '{file_path}': "
                f"esperada list, recebido {type(raw).__name__}. "
                f"Conteúdo (primeiros 200 chars): {str(raw)[:200]}"
            )
        return normalise_elements(raw)