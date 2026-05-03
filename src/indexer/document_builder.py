"""
document_builder.py
-------------------
Constrói o JSON final do documento de forma navegável, rastreável
e visualmente inspeccionável.

Estrutura de saída:
{
  "_meta": {
    "schema_version":  "2.0",
    "generated_at":    "2024-01-15T10:30:00",
    "filename":        "Regulamento_633-2024.pdf",
    "extraction_info": {
      "strategy":       "api" | "ocr_fallback" | "hybrid",
      "total_pages":    42,
      "ocr_pages":      [3, 7, 12],
      "avg_quality":    0.87,
      "total_elements": 234
    },
    "quality_flags": {
      "has_tables":         true,
      "has_images":         false,
      "has_lists":          true,
      "ocr_was_used":       false,
      "low_quality_pages":  []
    }
  },

  "_index": {
    "capitulos": ["CAP_I", "CAP_II", …],
    "artigos":   {"ART_1": "CAP_I", "ART_2": "CAP_I", …},
    "paginas":   {"1": ["ART_1"], "2": ["ART_1", "ART_2"], …}
  },

  "document_info": {
    "titulo":     str,
    "tipo":       str,
    "numero":     str,
    "ano":        str,
    "entidade":   str,
    "data":       str
  },

  "preambulo": str,

  "estrutura": {
    "CAP_I": {
      "titulo": str,
      "artigos": {
        "ART_1": {
          "titulo":    str,
          "conteudo":  str,
          "pagina":    int,
          "tabelas":   [str, …],   # Markdown
          "listas":    [[str], …], # listas de itens
          "notas":     [str, …]    # rodapés/notas detectadas
        }
      }
    }
  }
}

REGRA DE RASTREABILIDADE: cada artigo tem "pagina" (int) e pode ter
"partes" se foi truncado em múltiplos chunks. O índice _index.paginas
permite navegar do número de página para os artigos nessa página.
"""

import logging
import re
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = "2.0"

# Padrões para extracção de metadados do preâmbulo
_RE_NUMERO   = re.compile(r'(?:n[º°.]?\s*|N[úu]mero\s+)(\d+[-/]\d+)', re.IGNORECASE)
_RE_ANO      = re.compile(r'\b(20\d{2}|19\d{2})\b')
_RE_DATA     = re.compile(
    r'\b(\d{1,2}\s+de\s+\w+\s+de\s+(?:20|19)\d{2}|\d{1,2}[/-]\d{1,2}[/-]\d{4})\b',
    re.IGNORECASE
)
_RE_TIPO     = re.compile(
    r'\b(Regulamento|Decreto(?:-Lei)?|Portaria|Despacho|Deliberação|'
    r'Aviso|Lei|Resolução|Instrução|Circular|Ordem\s+de\s+Serviço)\b',
    re.IGNORECASE
)


class DocumentBuilder:
    """
    Constrói o envelope JSON final a partir dos dados parseados.

    Separa a responsabilidade de montar o JSON de saída do AnchorParser
    (que apenas sabe parsear estrutura) e do MetadataExtractor
    (que apenas extrai metadados do preâmbulo).
    """

    def build(
        self,
        filename:      str,
        parsed:        dict,
        ocr_report:    Optional[dict] = None,
        element_count: int = 0,
    ) -> dict:
        """
        Monta o documento JSON completo.

        Args:
            filename:      nome do ficheiro PDF (só o basename).
            parsed:        resultado do AnchorParser: {"preambulo": str, "estrutura": dict}
            ocr_report:    relatório do OCR fallback (se usado), ou None.
            element_count: número total de elementos processados.

        Returns:
            dicionário JSON completo e navegável.
        """
        preambulo = parsed.get("preambulo", "")
        estrutura = parsed.get("estrutura", {})

        # Enriquece artigos com tabelas/listas separadas do conteúdo
        estrutura = self._enrich_estrutura(estrutura)

        # Constrói o índice de navegação
        index = self._build_index(estrutura)

        # Extrai metadados do documento
        doc_info = self._extract_doc_info(filename, preambulo)

        # Informação de extracção e qualidade
        extraction_info = self._build_extraction_info(
            ocr_report, element_count, estrutura
        )

        return {
            "_meta": {
                "schema_version":  _SCHEMA_VERSION,
                "generated_at":    datetime.now(timezone.utc).isoformat(),
                "filename":        filename,
                "extraction_info": extraction_info,
                "quality_flags":   self._quality_flags(estrutura, ocr_report),
            },
            "_index":       index,
            "document_info": doc_info,
            "preambulo":    preambulo,
            "estrutura":    estrutura,
        }

    # ── Enriquecimento da estrutura ───────────────────────────────────────────

    def _enrich_estrutura(self, estrutura: dict) -> dict:
        """
        Processa cada artigo para separar tabelas e listas do texto corrido.

        Tabelas Markdown (linhas começando com |) são extraídas para "tabelas".
        Listas (linhas começando com - / * / N.) são extraídas para "listas".
        O restante fica em "conteudo".
        """
        for cap_id, cap in estrutura.items():
            for art_id, art in cap.get("artigos", {}).items():
                conteudo = art.get("conteudo", "")
                if not conteudo:
                    continue

                tabelas, listas, texto_limpo = self._split_content(conteudo)
                art["conteudo"] = texto_limpo

                if tabelas:
                    art["tabelas"] = tabelas
                if listas:
                    art["listas"] = listas

        return estrutura

    @staticmethod
    def _split_content(conteudo: str) -> tuple[list, list, str]:
        """
        Separa tabelas Markdown, listas e texto corrido num conteúdo misto.

        Returns:
            (tabelas, listas, texto_limpo)
        """
        tabelas    = []
        listas     = []
        text_lines = []

        lines = conteudo.split("\n")
        i = 0
        while i < len(lines):
            line = lines[i]

            # Detecta bloco de tabela Markdown
            if line.strip().startswith("|"):
                table_lines = []
                while i < len(lines) and lines[i].strip().startswith("|"):
                    table_lines.append(lines[i])
                    i += 1
                tabelas.append("\n".join(table_lines))
                continue

            # Detecta item de lista
            if re.match(r'^\s*[-*•]\s', line) or re.match(r'^\s*\d+[.)]\s', line):
                list_items = []
                while i < len(lines) and (
                    re.match(r'^\s*[-*•]\s', lines[i]) or
                    re.match(r'^\s*\d+[.)]\s', lines[i])
                ):
                    item = re.sub(r'^\s*[-*•\d.]+\s*', '', lines[i]).strip()
                    if item:
                        list_items.append(item)
                    i += 1
                if list_items:
                    listas.append(list_items)
                continue

            text_lines.append(line)
            i += 1

        return tabelas, listas, "\n".join(text_lines).strip()

    # ── Índice de navegação ───────────────────────────────────────────────────

    @staticmethod
    def _build_index(estrutura: dict) -> dict:
        """
        Constrói o índice de navegação do documento.

        Permite:
          - listar todos os capítulos e artigos
          - encontrar o capítulo de um artigo
          - encontrar os artigos de uma página
        """
        capitulos   = list(estrutura.keys())
        artigos_map = {}  # ART_N → CAP_X
        paginas_map = {}  # "N" → [ART_N, …]

        for cap_id, cap in estrutura.items():
            for art_id, art in cap.get("artigos", {}).items():
                artigos_map[art_id] = cap_id
                page_key = str(art.get("pagina", 1))
                paginas_map.setdefault(page_key, []).append(art_id)

        return {
            "capitulos": capitulos,
            "artigos":   artigos_map,
            "paginas":   paginas_map,
        }

    # ── Metadados do documento ────────────────────────────────────────────────

    @staticmethod
    def _extract_doc_info(filename: str, preambulo: str) -> dict:
        """
        Extrai metadados do documento a partir do nome do ficheiro e preâmbulo.

        Usa regex robustas que toleram variações de formatação comuns
        em documentos legais portugueses.
        """
        texto = f"{filename} {preambulo}"

        # Número do documento
        m_num = _RE_NUMERO.search(texto)
        numero = m_num.group(1) if m_num else ""

        # Ano
        m_ano = _RE_ANO.search(filename)  # prioriza o nome do ficheiro
        if not m_ano:
            m_ano = _RE_ANO.search(preambulo)
        ano = m_ano.group(0) if m_ano else ""

        # Data
        m_data = _RE_DATA.search(preambulo)
        data = m_data.group(0) if m_data else ""

        # Tipo de documento
        m_tipo = _RE_TIPO.search(texto)
        tipo = m_tipo.group(0).title() if m_tipo else "Documento"

        # Título: primeiras palavras significativas do preâmbulo
        titulo = _extract_title(preambulo) or filename.replace("_", " ").replace(".pdf", "")

        return {
            "titulo":   titulo,
            "tipo":     tipo,
            "numero":   numero,
            "ano":      ano,
            "data":     data,
            "entidade": _extract_entity(preambulo),
            "filename": filename,
        }

    # ── Informação de extracção ───────────────────────────────────────────────

    @staticmethod
    def _build_extraction_info(
        ocr_report:    Optional[dict],
        element_count: int,
        estrutura:     dict,
    ) -> dict:
        """Constrói o bloco de informação sobre a extracção."""
        total_arts = sum(
            len(cap.get("artigos", {}))
            for cap in estrutura.values()
        )

        if ocr_report:
            strategy   = "ocr_fallback" if ocr_report.get("ocr_ratio", 0) > 0.8 else "hybrid"
            total_pgs  = ocr_report.get("total_pages", 0)
            ocr_pages  = ocr_report.get("ocr_pages", [])
            avg_quality = ocr_report.get("avg_quality", 1.0)
        else:
            strategy    = "api"
            total_pgs   = 0
            ocr_pages   = []
            avg_quality = 1.0

        return {
            "strategy":        strategy,
            "total_pages":     total_pgs,
            "total_elements":  element_count,
            "total_artigos":   total_arts,
            "ocr_pages":       ocr_pages,
            "avg_quality":     avg_quality,
        }

    @staticmethod
    def _quality_flags(estrutura: dict, ocr_report: Optional[dict]) -> dict:
        """Flags de qualidade para inspecção rápida do documento."""
        has_tables = any(
            bool(art.get("tabelas"))
            for cap in estrutura.values()
            for art in cap.get("artigos", {}).values()
        )
        has_lists = any(
            bool(art.get("listas"))
            for cap in estrutura.values()
            for art in cap.get("artigos", {}).values()
        )

        ocr_used   = bool(ocr_report and ocr_report.get("ocr_pages"))
        ocr_pages  = ocr_report.get("ocr_pages", []) if ocr_report else []

        return {
            "has_tables":        has_tables,
            "has_lists":         has_lists,
            "ocr_was_used":      ocr_used,
            "low_quality_pages": ocr_pages,
        }


# ── Helpers de extracção de texto ─────────────────────────────────────────────

def _extract_title(preambulo: str) -> str:
    """Extrai um título razoável das primeiras linhas do preâmbulo."""
    if not preambulo:
        return ""
    # Usa a primeira frase que termine em ponto (heurística para títulos legais)
    sentences = re.split(r'(?<=[.!?])\s+', preambulo.strip())
    for s in sentences[:3]:
        s = s.strip()
        if 10 < len(s) < 200:
            return s
    return preambulo[:150].strip()


def _extract_entity(preambulo: str) -> str:
    """
    Tenta extrair a entidade emissora do documento (câmara, ministério, etc.).
    Heurística simples: procura padrões comuns em cabeçalhos PT.
    """
    patterns = [
        r'(Câmara\s+Municipal\s+d[eo]\s+\w+)',
        r'(Ministério\s+d[oa]\s+\w+)',
        r'(Instituto\s+\w+)',
        r'(Universidade\s+d[eo]\s+\w+)',
        r'(Município\s+d[eo]\s+\w+)',
        r'(Serviço\s+\w+)',
    ]
    for pattern in patterns:
        m = re.search(pattern, preambulo, re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return ""