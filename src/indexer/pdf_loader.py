"""
pdf_loader.py
Implementação de LoaderProtocol que chama a Unstructured Platform API.

Responsabilidade única: enviar o PDF à API e devolver a lista normalizada
de elementos. Não faz parsing, não filtra categorias, não toca em metadados.
"""

import os
import requests


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
        Lança requests.HTTPError se a API responder com erro.
        """
        with open(file_path, "rb") as fh:
            response = requests.post(
                self._API_URL,
                headers={"unstructured-api-key": self._api_key},
                files={"files": (os.path.basename(file_path), fh, "application/pdf")},
                data={
                    "strategy": "hi_res",
                    "languages": ["por"],
                    "include_page_breaks": "true",
                    "output_format": "application/json",
                },
                timeout=300,
            )

        response.raise_for_status()
        return self._normalise(response.json())

    # ── privado ───────────────────────────────────────────────────────────────

    @staticmethod
    def _normalise(raw: list) -> list[dict]:
        """Converte a resposta da API para o formato interno do projecto."""
        elements = []
        for el in raw:
            text = (el.get("text") or "").strip()
            if not text:
                continue
            elements.append({
                "text":     text,
                "category": el.get("type", "Uncategorized"),
                "page":     (el.get("metadata") or {}).get("page_number", 1) or 1,
            })
        return elements