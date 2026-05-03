"""
pdf_loader.py  (v2 — com detecção de qualidade e fallback OCR)
--------------------------------------------------------------
Implementação de LoaderProtocol que chama a Unstructured Platform API
e activa automaticamente o fallback OCR quando o texto devolvido é de
baixa qualidade (garbled, encoding corrupto, fontes não embebidas).

Fluxo de decisão:
  1. Chama a Unstructured API com hi_res + extracção de tabelas
  2. Avalia a qualidade do texto devolvido (score_text_quality)
  3a. Qualidade OK → devolve elementos normalizados da API
  3b. Qualidade baixa → activa ocr_fallback.extract_with_ocr_fallback()
      que usa pdfplumber + pytesseract página a página

A escolha entre API e OCR é totalmente transparente para o PDFIndexer
— o contrato do LoaderProtocol é sempre respeitado.

Parâmetros ajustáveis no construtor:
  - quality_threshold: score abaixo do qual activa OCR (default 0.45)
  - force_ocr:         ignora a API e usa sempre OCR local
  - extract_tables:    activa pdf_infer_table_structure na API
"""

import logging
import os
from typing import Optional

import requests

from text_quality import needs_ocr, score_text_quality
from ocr_fallback import extract_with_ocr_fallback
from element_utils import normalise_elements

logger = logging.getLogger(__name__)


class PDFLoader:
    """
    Carrega um PDF com detecção automática de qualidade e fallback OCR.

    Compatível com LoaderProtocol — devolve sempre:
        [{"text": str, "category": str, "page": int}, …]
    """

    _API_URL = "https://api.unstructuredapp.io/general/v0/general"

    def __init__(
        self,
        api_key:           Optional[str]  = None,
        quality_threshold: float          = 0.45,
        force_ocr:         bool           = False,
        extract_tables:    bool           = True,
    ):
        """
        Args:
            api_key:           chave da Unstructured API (ou UNSTRUCTURED_API_KEY env).
            quality_threshold: score mínimo para aceitar texto da API (0–1).
            force_ocr:         força OCR local, ignorando a API.
            extract_tables:    activa extracção de tabelas na API (pdf_infer_table_structure).
        """
        key = api_key or os.environ.get("UNSTRUCTURED_API_KEY", "")
        if not key and not force_ocr:
            raise ValueError(
                "UNSTRUCTURED_API_KEY não definida. "
                "Passa api_key= ao construtor, define a variável de ambiente, "
                "ou usa force_ocr=True para extracção local."
            )
        self._api_key           = key
        self._quality_threshold = quality_threshold
        self._force_ocr         = force_ocr
        self._extract_tables    = extract_tables

    def load(self, file_path: str) -> list[dict]:
        """
        Carrega o PDF e devolve elementos normalizados.

        Raises:
            requests.HTTPError: Se a API responder com código de erro HTTP.
            FileNotFoundError:  Se o ficheiro não existir.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"[PDFLoader] Ficheiro não encontrado: '{file_path}'")

        # Modo OCR forçado — ignora completamente a API
        if self._force_ocr:
            logger.info("[PDFLoader] force_ocr=True → OCR local activado")
            return self._load_ocr(file_path)

        # Tentativa via API
        try:
            api_elements = self._call_api(file_path)
        except requests.exceptions.ConnectionError as exc:
            logger.warning(
                "[PDFLoader] Sem acesso à API (%s) → fallback OCR local", exc
            )
            return self._load_ocr(file_path)
        except requests.HTTPError as exc:
            logger.error("[PDFLoader] Erro HTTP da API: %s", exc)
            raise

        # Avaliação de qualidade — decide se OCR é necessário
        if needs_ocr(api_elements, threshold=self._quality_threshold):
            logger.warning(
                "[PDFLoader] Qualidade da API abaixo do threshold (%.2f) "
                "→ activando OCR fallback para '%s'",
                self._quality_threshold, os.path.basename(file_path)
            )
            return self._load_ocr(file_path)

        logger.info(
            "[PDFLoader] API OK: %d elementos extraídos de '%s'",
            len(api_elements), os.path.basename(file_path)
        )
        return api_elements

    # ── Métodos privados ──────────────────────────────────────────────────────

    def _call_api(self, file_path: str) -> list[dict]:
        """Chama a Unstructured API e devolve elementos normalizados."""
        data = {
            "strategy":            "hi_res",
            "languages":           ["por"],
            "include_page_breaks": "true",
            "output_format":       "application/json",
        }
        if self._extract_tables:
            data["pdf_infer_table_structure"] = "true"
            data["skip_infer_table_types"]    = "[]"

        with open(file_path, "rb") as fh:
            response = requests.post(
                self._API_URL,
                headers={"unstructured-api-key": self._api_key},
                files={"files": (os.path.basename(file_path), fh, "application/pdf")},
                data=data,
                timeout=300,
            )

        response.raise_for_status()
        raw = response.json()

        if not isinstance(raw, list):
            raise ValueError(
                f"[PDFLoader] Resposta inesperada da API: "
                f"esperada list, recebido {type(raw).__name__}. "
                f"Conteúdo: {str(raw)[:200]}"
            )

        return normalise_elements(raw)

    def _load_ocr(self, file_path: str) -> list[dict]:
        """Extracção local com pdfplumber + pytesseract."""
        elements, report = extract_with_ocr_fallback(
            file_path,
            force_ocr=self._force_ocr,
        )
        # Injeta relatório OCR no primeiro elemento como metadata extra
        # (invisível ao parser, útil para auditoria)
        if elements:
            elements[0]["_ocr_report"] = report
        return elements