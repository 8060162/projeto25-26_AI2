"""
ocr_fallback.py
---------------
Motor de OCR de fallback, activado quando a extracção de texto da
Unstructured API devolve conteúdo com qualidade abaixo do threshold.

Estratégia em dois níveis:
  1. pdfplumber  → extracção de texto nativa (sem rasterização), rápida.
     Também extrai tabelas como estruturas Python → convertidas para Markdown.
  2. pytesseract → rasteriza cada página a 300 DPI e aplica OCR.
     Activado per-página se pdfplumber devolver lixo nessa página.

O resultado é uma lista de elementos no mesmo formato do LoaderProtocol,
podendo ser usada como drop-in replacement dos elementos da API.

Dependências: pdfplumber, pytesseract, Pillow (pdf2image opcional mas recomendada)
"""

import io
import logging
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Threshold de qualidade por página — abaixo disto, pytesseract é chamado
_PAGE_OCR_THRESHOLD = 0.40

# DPI para rasterização OCR — 300 é o mínimo recomendado para texto legal
_OCR_DPI = 300

# Línguas tesseract — português + inglês (fallback para siglas e termos técnicos)
_TESS_LANG = "por+eng"


def _table_to_markdown(table: list[list]) -> str:
    """
    Converte uma tabela pdfplumber (lista de listas) para Markdown GFM.

    A primeira linha é tratada como cabeçalho se tiver conteúdo distinto.
    Células None são substituídas por string vazia.
    """
    if not table:
        return ""

    # Normaliza células
    rows = [[str(cell or "").strip() for cell in row] for row in table]

    # Remove linhas completamente vazias
    rows = [r for r in rows if any(c for c in r)]
    if not rows:
        return ""

    col_count = max(len(r) for r in rows)
    # Padeia linhas curtas
    rows = [r + [""] * (col_count - len(r)) for r in rows]

    # Determina larguras de coluna para formatação legível
    widths = [max(len(rows[i][j]) for i in range(len(rows))) for j in range(col_count)]
    widths = [max(w, 3) for w in widths]

    def fmt_row(cells):
        return "| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(cells)) + " |"

    lines = [fmt_row(rows[0])]
    lines.append("| " + " | ".join("-" * w for w in widths) + " |")
    for row in rows[1:]:
        lines.append(fmt_row(row))

    return "\n".join(lines)


def _pdfplumber_extract_page(page) -> list[dict]:
    """
    Extrai elementos de uma única página pdfplumber.

    Devolve lista de elementos no formato LoaderProtocol:
        {"text": str, "category": str, "page": int}
    """
    try:
        from text_quality import score_text_quality
    except ImportError:
        from utils.text_quality import score_text_quality

    page_num = page.page_number
    elements = []

    # 1. Tabelas — extraídas antes do texto para evitar duplicação
    table_bboxes = []
    try:
        tables = page.find_tables()
        for table_obj in tables:
            table_bboxes.append(table_obj.bbox)
            data = table_obj.extract()
            if data:
                md = _table_to_markdown(data)
                if md.strip():
                    elements.append({
                        "text":     md,
                        "category": "Table",
                        "page":     page_num,
                        "source":   "pdfplumber",
                    })
    except Exception as exc:
        logger.debug("Falha ao extrair tabelas da página %d: %s", page_num, exc)

    # 2. Texto — excluindo as áreas de tabelas já extraídas
    try:
        if table_bboxes:
            # Filtra palavras fora das bounding boxes das tabelas
            words = page.extract_words()
            # Agrupa palavras não cobertas por tabelas
            free_words = [
                w for w in words
                if not any(
                    w["x0"] >= bbox[0] and w["top"] >= bbox[1]
                    and w["x1"] <= bbox[2] and w["bottom"] <= bbox[3]
                    for bbox in table_bboxes
                )
            ]
            raw_text = " ".join(w["text"] for w in free_words)
        else:
            raw_text = page.extract_text(x_tolerance=3, y_tolerance=3) or ""

        if raw_text.strip():
            # Avalia qualidade — se for lixo, sinaliza para OCR
            quality = score_text_quality(raw_text)
            if quality >= _PAGE_OCR_THRESHOLD:
                # Divide em blocos por parágrafo (heurística simples)
                for block in re.split(r'\n{2,}', raw_text):
                    block = block.strip()
                    if block:
                        elements.append({
                            "text":     block,
                            "category": _infer_category(block),
                            "page":     page_num,
                            "source":   "pdfplumber",
                        })
            else:
                logger.debug(
                    "Página %d: qualidade pdfplumber=%.2f → escalando para OCR",
                    page_num, quality
                )
                return []  # Sinal para o chamador usar OCR nesta página

    except Exception as exc:
        logger.warning("Falha ao extrair texto da página %d: %s", page_num, exc)

    return elements


def _ocr_page(page, page_num: int) -> list[dict]:
    """
    Rasteriza uma página e aplica pytesseract OCR.

    Devolve lista de elementos no formato LoaderProtocol.
    """
    try:
        import pytesseract
        from PIL import Image

        # Converte página pdfplumber para imagem PIL
        img = page.to_image(resolution=_OCR_DPI).original

        # Configuração tesseract para documentos legais/administrativos
        custom_config = (
            f"--oem 3 --psm 6 "  # LSTM + assume bloco de texto uniforme
            f"-l {_TESS_LANG} "
        )

        # OCR com dados de posição (para categorização futura)
        ocr_text = pytesseract.image_to_string(img, config=custom_config)

        elements = []
        for block in re.split(r'\n{2,}', ocr_text):
            block = block.strip()
            if block and len(block) > 2:  # ignora fragmentos de 1-2 chars
                elements.append({
                    "text":     block,
                    "category": _infer_category(block),
                    "page":     page_num,
                    "source":   "ocr_tesseract",
                })

        logger.debug("Página %d: OCR → %d blocos", page_num, len(elements))
        return elements

    except ImportError:
        logger.error("pytesseract ou Pillow não instalados — OCR indisponível")
        return []
    except Exception as exc:
        logger.error("Falha OCR na página %d: %s", page_num, exc)
        return []


def _infer_category(text: str) -> str:
    """
    Infere a categoria de um bloco de texto usando heurísticas simples.
    Compatível com as categorias esperadas pelo AnchorParser.
    """
    stripped = text.strip()
    lines    = stripped.splitlines()
    first    = lines[0].strip() if lines else ""

    # Linha única curta → provável título
    if len(lines) == 1 and len(first) < 120 and first.isupper():
        return "Title"

    # Começa com número/letra seguido de ponto ou parêntese → lista
    if re.match(r'^[\d]+[.)]\s', stripped) or re.match(r'^[a-z][.)]\s', stripped):
        return "ListItem"

    # Padrão de artigo/capítulo
    if re.match(r'^\s*(Artigo|CAPÍTULO|Artículo|Article)\s+\d', stripped, re.IGNORECASE):
        return "Title"

    return "NarrativeText"


def extract_with_ocr_fallback(
    file_path: str,
    force_ocr: bool = False,
    ocr_threshold: float = _PAGE_OCR_THRESHOLD,
) -> tuple[list[dict], dict]:
    """
    Extracção completa com fallback automático para OCR por página.

    Algoritmo:
      Para cada página:
        1. Tenta pdfplumber (rápido, preserva estrutura)
        2. Se qualidade < threshold OU force_ocr → pytesseract (300 DPI)
        3. Combina resultados mantendo ordem de páginas

    Args:
        file_path:      caminho para o PDF.
        force_ocr:      força OCR em todas as páginas (ignora threshold).
        ocr_threshold:  score mínimo para aceitar texto pdfplumber (0–1).

    Returns:
        Tuplo (elementos, relatório_de_qualidade)
        - elementos: lista de dicts {text, category, page, source}
        - relatório: {total_pages, ocr_pages, plumber_pages, avg_quality}
    """
    import pdfplumber

    # importação local para evitar circular entre módulos
    try:
        from text_quality import score_text_quality
    except ImportError:
        from utils.text_quality import score_text_quality

    elements  = []
    ocr_pages = []
    plb_pages = []
    qualities = []

    with pdfplumber.open(file_path) as pdf:
        total_pages = len(pdf.pages)
        logger.info("OCR Fallback: a processar %d páginas de '%s'", total_pages, file_path)

        for page in pdf.pages:
            pnum = page.page_number

            if force_ocr:
                page_elements = _ocr_page(page, pnum)
                ocr_pages.append(pnum)
                qualities.append(0.0)
            else:
                page_elements = _pdfplumber_extract_page(page)

                if not page_elements:
                    # pdfplumber falhou ou qualidade baixa → OCR
                    page_elements = _ocr_page(page, pnum)
                    ocr_pages.append(pnum)
                    qualities.append(0.0)
                else:
                    plb_pages.append(pnum)
                    avg_q = sum(
                        score_text_quality(el["text"])
                        for el in page_elements
                        if el.get("text")
                    ) / max(len(page_elements), 1)
                    qualities.append(avg_q)

            elements.extend(page_elements)

    report = {
        "total_pages":  total_pages,
        "ocr_pages":    sorted(ocr_pages),
        "plumber_pages": sorted(plb_pages),
        "avg_quality":  round(sum(qualities) / max(len(qualities), 1), 3),
        "ocr_ratio":    round(len(ocr_pages) / max(total_pages, 1), 3),
    }

    logger.info(
        "OCR Fallback concluído: %d elementos | %.0f%% páginas via OCR | qualidade média=%.2f",
        len(elements), report["ocr_ratio"] * 100, report["avg_quality"]
    )

    return elements, report