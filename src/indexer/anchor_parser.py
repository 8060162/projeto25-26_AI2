"""
anchor_parser.py
Constrói o JSON hierárquico a partir da lista de elementos da Unstructured API.

Estratégia de âncoras em dois níveis (resiliente a falhas de classificação da API):

  Nível 1 — category == "Title"  (classificação da API)
    Confiamos plenamente. Um Title que bate no padrão é sempre âncora.

  Nível 2 — qualquer category    (regex como fallback)
    Aplicado apenas se o texto começa exactamente com "Artigo N" ou "CAPÍTULO X".
    A guarda é o regex, não a categoria.
    Um NarrativeText que contém "Artigo 3" a meio de frase NUNCA bate,
    porque ART_PATTERN exige início de linha (^\\s*Artigo).

Fluxo por elemento:
  ┌─ category em _DISCARD               → ignorado (Header/Footer/PageBreak)
  ├─ CAP_PATTERN.match(text)            → âncora de capítulo (qualquer category)
  ├─ ART_PATTERN.match(text)            → âncora de artigo   (qualquer category)
  │    └─ se ainda não há capítulo      → cria capítulo implícito
  ├─ category == "Title" + art activo   → título solto do artigo
  ├─ category em _CONTENT + art activo  → acumula conteúdo
  └─ category em _CONTENT + sem art    → preâmbulo (máx. _MAX_PREAMBLE_BLOCKS)

Responsabilidade: APENAS parsing. Não sabe o nome do ficheiro,
não extrai metadados, não faz I/O.

ALTERAÇÃO (refactor): a conversão de page para int passou a ser feita
com tratamento explícito de ValueError/TypeError. Um elemento com valor
de página inválido (None, "N/A", string arbitrária) já não interrompe
o parsing do documento inteiro — é usado o valor 1 como fallback e
emitido um aviso de logging com contexto suficiente para diagnóstico.
"""

import logging
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from utils.regex_patterns import CAP_PATTERN, ART_PATTERN

logger = logging.getLogger(__name__)

# ── Categorias ────────────────────────────────────────────────────────────────

# Categorias descartadas — ruído estrutural da API
_DISCARD: frozenset[str] = frozenset({
    "Header", "Footer", "PageBreak", "PageNumber",
})

# Categorias que transportam conteúdo semântico (texto de artigo)
_CONTENT: frozenset[str] = frozenset({
    "NarrativeText", "ListItem", "Table", "UncategorizedText",
})

# ── Constantes de estrutura ───────────────────────────────────────────────────

_MAX_PREAMBLE_BLOCKS = 10
_NO_TITLE            = "Sem Título"

# ID e título do capítulo implícito criado quando um artigo surge sem capítulo.
# O prefixo "__" garante que nunca colidirá com IDs gerados pelo regex
# (que têm o formato "CAP_<LABEL>", ex: "CAP_I", "CAP_1").
_IMPLICIT_CAP_ID    = "__NO_CHAPTER__"
_IMPLICIT_CAP_TITLE = "Sem Capítulos"


def _parse_page(raw_page: object, text_preview: str) -> int:
    """
    Converte o valor de página para int de forma segura.

    Um elemento com página inválida (None, "N/A", string arbitrária) usa
    o valor 1 como fallback e emite aviso com contexto para diagnóstico.
    Evita que um único elemento corrompido interrompa o parsing do
    documento inteiro.

    Args:
        raw_page:     valor raw do campo "page" do elemento.
        text_preview: início do texto do elemento (para contexto no log).

    Returns:
        Número de página (int), mínimo 1.
    """
    try:
        return int(raw_page or 1)
    except (ValueError, TypeError):
        logger.warning(
            "Valor de página inválido '%s' no elemento '%.50s…' — usando página 1.",
            raw_page,
            text_preview,
        )
        return 1


class AnchorParser:
    """
    Transforma a lista de elementos num dicionário com a estrutura:

        {
            "preambulo": str,
            "estrutura": {
                "CAP_X": {
                    "titulo": str,
                    "artigos": {
                        "ART_N": {
                            "titulo":   str,
                            "conteudo": str,
                            "pagina":   int
                        }
                    }
                }
            }
        }
    """

    def parse(self, elements: list[dict]) -> dict:
        self._result: dict = {"preambulo": "", "estrutura": {}}
        self._current_cap_id:  str | None  = None
        self._current_art_id:  str | None  = None
        self._preamble_blocks: list[str]   = []
        self._preamble_closed: bool        = False

        for el in elements:
            category = el.get("category", "")
            text     = el.get("text", "").strip()
            page     = _parse_page(el.get("page"), text)

            if not text or category in _DISCARD:
                continue

            if self._try_chapter_anchor(text):
                continue
            if self._try_article_anchor(text, page):
                continue
            if self._try_loose_title(category, text):
                continue
            self._try_content(category, text)

        self._result["preambulo"] = " ".join(self._preamble_blocks)
        return self._result

    # ── handlers privados ─────────────────────────────────────────────────────

    def _try_chapter_anchor(self, text: str) -> bool:
        """
        Tenta reconhecer uma âncora de capítulo.
        Aplica-se a qualquer category — o regex é a guarda suficiente.

        Returns:
            True se o elemento foi consumido como âncora de capítulo.
        """
        cap_m = CAP_PATTERN.match(text)
        if not cap_m:
            return False

        cap_label             = cap_m.group(1).upper()
        self._current_cap_id  = f"CAP_{cap_label}"
        self._current_art_id  = None
        self._preamble_closed = True

        self._result["estrutura"][self._current_cap_id] = {
            "titulo":  text,
            "artigos": {},
        }
        return True

    def _try_article_anchor(self, text: str, page: int) -> bool:
        """
        Tenta reconhecer uma âncora de artigo.
        Aplica-se a qualquer category — o regex é a guarda suficiente.

        Se não houver capítulo activo, cria um capítulo implícito com ID
        _IMPLICIT_CAP_ID (prefixo "__" impede colisão com IDs gerados por regex).

        Returns:
            True se o elemento foi consumido como âncora de artigo.
        """
        art_m = ART_PATTERN.match(text)
        if not art_m:
            return False

        art_num      = art_m.group(1)
        inline_title = art_m.group(2).strip(" .º°-")

        self._current_art_id  = f"ART_{art_num}"
        self._preamble_closed = True

        # Documento sem capítulos → capítulo implícito (criado uma única vez)
        if self._current_cap_id is None:
            self._current_cap_id = _IMPLICIT_CAP_ID
            self._result["estrutura"][_IMPLICIT_CAP_ID] = {
                "titulo":  _IMPLICIT_CAP_TITLE,
                "artigos": {},
            }

        self._result["estrutura"][self._current_cap_id]["artigos"][self._current_art_id] = {
            "titulo":   inline_title or _NO_TITLE,
            "conteudo": "",
            "pagina":   page,
        }
        return True

    def _try_loose_title(self, category: str, text: str) -> bool:
        """
        Tenta aplicar um Title solto ao artigo activo.

        Returns:
            True se o elemento foi consumido como título solto.
        """
        if category != "Title" or self._current_art_id is None:
            return False

        art = self._result["estrutura"][self._current_cap_id]["artigos"][self._current_art_id]
        if art["titulo"] == _NO_TITLE:
            art["titulo"] = text
        return True

    def _try_content(self, category: str, text: str) -> None:
        """
        Acumula conteúdo no artigo activo ou no preâmbulo.

        Conteúdo sem artigo activo vai para o preâmbulo, até ao limite
        de _MAX_PREAMBLE_BLOCKS blocos.
        """
        if category not in _CONTENT:
            return

        if self._current_art_id is not None:
            art = self._result["estrutura"][self._current_cap_id]["artigos"][self._current_art_id]
            art["conteudo"] += (" " if art["conteudo"] else "") + text
        elif not self._preamble_closed:
            if len(self._preamble_blocks) < _MAX_PREAMBLE_BLOCKS:
                self._preamble_blocks.append(text)