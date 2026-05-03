"""
element_utils.py  (v2 — com suporte rico a tabelas, listas e metadados de qualidade)
-------------------------------------------------------------------------------------
Normaliza os elementos raw da Unstructured API para o formato interno
usado pelo LoaderProtocol.

Formato de saída por elemento:
    {
        "text":       str,   # conteúdo textual limpo
        "category":   str,   # categoria do bloco (Title, NarrativeText, Table, …)
        "page":       int,   # página de origem (1-based)

        # Campos opcionais (presentes apenas quando relevante):
        "table_md":   str,   # tabela em Markdown (se category == "Table")
        "table_html": str,   # tabela em HTML original (se disponível da API)
        "items":      list,  # lista de strings (se category == "ListItem" agrupado)
        "source":     str,   # origem: "api" | "pdfplumber" | "ocr_tesseract"
        "quality":    float, # score de qualidade do texto (0–1)
    }
"""

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ── Categorias reconhecidas da API ───────────────────────────────────────────

_KNOWN_CATEGORIES = frozenset({
    "Title", "NarrativeText", "ListItem", "Table",
    "Header", "Footer", "PageBreak", "PageNumber",
    "UncategorizedText", "Image", "FigureCaption",
    "Formula", "CodeSnippet",
})


def normalise_elements(raw: list) -> list[dict]:
    """
    Normaliza a lista raw da API para o formato interno.

    Processa em duas passagens:
      1ª passagem: normalização individual de cada elemento
      2ª passagem: agrupamento de ListItems consecutivos (opcional)

    Args:
        raw: lista de dicts da API (formato Unstructured v0).

    Returns:
        lista de elementos normalizados.
    """
    if not isinstance(raw, list):
        logger.error("normalise_elements: esperada list, recebido %s", type(raw))
        return []

    normalised = []
    for idx, item in enumerate(raw):
        try:
            el = _normalise_one(item)
            if el:
                normalised.append(el)
        except Exception as exc:
            logger.warning("Erro ao normalizar elemento %d: %s", idx, exc)

    return normalised


def _normalise_one(item: dict) -> Optional[dict]:
    """
    Normaliza um único elemento raw da API.

    Devolve None se o elemento for vazio ou irrelevante.
    """
    if not isinstance(item, dict):
        return None

    # --- Campos base --------------------------------------------------------
    raw_text     = item.get("text", "") or ""
    category     = _safe_category(item.get("type", ""))
    metadata     = item.get("metadata", {}) or {}
    page         = _safe_page(metadata.get("page_number"))
    element_id   = item.get("element_id", "")

    text = raw_text.strip()

    # Ignora elementos completamente vazios
    if not text and category not in ("Image",):
        return None

    # --- Construção do elemento normalizado ---------------------------------
    el: dict = {
        "text":     text,
        "category": category,
        "page":     page,
        "source":   "api",
    }

    # --- Tratamento rico de Tabelas ----------------------------------------
    if category == "Table":
        el.update(_process_table(item, metadata, text))

    # --- Imagens: caption e alt text ----------------------------------------
    elif category == "Image":
        caption = metadata.get("image_base64", "")  # ignoramos base64 no JSON
        alt     = text or metadata.get("alt_text", "")
        el["text"]  = alt or "[Imagem sem descrição]"
        el["image_caption"] = text if text else ""

    # --- FigureCaption ------------------------------------------------------
    elif category == "FigureCaption":
        el["is_caption"] = True

    # --- Metadados de coordenadas (útil para debugging de layouts) ----------
    coords = metadata.get("coordinates")
    if coords and isinstance(coords, dict):
        el["bbox"] = _extract_bbox(coords)

    # --- Score de qualidade (apenas para produção, não sobrecarrega o JSON) --
    if text:
        try:
            from text_quality import score_text_quality
            q = score_text_quality(text)
            if q < 0.7:  # só registamos quando é notável
                el["quality"] = round(q, 3)
        except ImportError:
            pass

    return el


def _process_table(item: dict, metadata: dict, fallback_text: str) -> dict:
    """
    Processa um elemento Table da API, extraindo HTML e Markdown.

    A API pode devolver:
      - text_as_html: tabela em HTML (preferencial)
      - text:         tabela em texto (fallback)

    Converte HTML → Markdown para facilitar ingestão em LLMs.
    """
    extra = {}

    html_raw = metadata.get("text_as_html", "") or item.get("text_as_html", "")
    if html_raw:
        extra["table_html"] = html_raw
        md = _html_table_to_markdown(html_raw)
        if md:
            extra["table_md"] = md
            extra["text"]     = md  # sobrescreve o text com Markdown limpo
        else:
            extra["text"] = fallback_text
    else:
        extra["text"] = fallback_text

    return extra


def _html_table_to_markdown(html: str) -> str:
    """
    Converte uma tabela HTML simples para Markdown GFM.

    Usa regex em vez de BeautifulSoup para evitar dependência extra.
    Funciona correctamente para o HTML simples gerado pela Unstructured API.
    """
    # Remove atributos de tags mas mantém a estrutura
    html = re.sub(r'<(td|th|tr|thead|tbody|table)[^>]*>', lambda m: f'<{m.group(1)}>', html)

    rows = re.findall(r'<tr>(.*?)</tr>', html, re.DOTALL | re.IGNORECASE)
    if not rows:
        return ""

    table_rows = []
    for row in rows:
        cells = re.findall(r'<t[dh]>(.*?)</t[dh]>', row, re.DOTALL | re.IGNORECASE)
        # Limpa HTML interno das células
        cells = [re.sub(r'<[^>]+>', ' ', c).strip() for c in cells]
        cells = [re.sub(r'\s+', ' ', c) for c in cells]
        if any(cells):
            table_rows.append(cells)

    if not table_rows:
        return ""

    col_count = max(len(r) for r in table_rows)
    table_rows = [r + [""] * (col_count - len(r)) for r in table_rows]

    widths = [
        max(len(table_rows[i][j]) for i in range(len(table_rows)))
        for j in range(col_count)
    ]
    widths = [max(w, 3) for w in widths]

    def fmt(cells):
        return "| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(cells)) + " |"

    lines = [fmt(table_rows[0])]
    lines.append("| " + " | ".join("-" * w for w in widths) + " |")
    for row in table_rows[1:]:
        lines.append(fmt(row))

    return "\n".join(lines)


def _safe_category(raw: str) -> str:
    """Valida e normaliza o tipo/categoria do elemento."""
    if raw in _KNOWN_CATEGORIES:
        return raw
    if raw:
        logger.debug("Categoria desconhecida '%s' → 'UncategorizedText'", raw)
    return "UncategorizedText"


def _safe_page(raw) -> int:
    """Converte page_number para int de forma segura."""
    try:
        return int(raw or 1)
    except (TypeError, ValueError):
        return 1


def _extract_bbox(coords: dict) -> Optional[list]:
    """Extrai bounding box normalizada das coordenadas da API."""
    points = coords.get("points")
    if points and len(points) >= 2:
        try:
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            return [min(xs), min(ys), max(xs), max(ys)]
        except (IndexError, TypeError):
            pass
    return None