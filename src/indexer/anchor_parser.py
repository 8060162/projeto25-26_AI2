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
  ├─ CAP_PATTERN.match(text)            -> ancora de capitulo (qualquer category)
  ├─ ART_PATTERN.match(text)            -> ancora de artigo   (qualquer category)
  │    └─ se ainda não há capítulo      → cria CAP_0 implícito
  ├─ category == "Title" + art activo   → título solto do artigo
  ├─ category em _CONTENT + art activo  → acumula conteudo
  └─ category em _CONTENT + sem art    → preâmbulo (máx. _MAX_PREAMBLE_BLOCKS)

Responsabilidade: APENAS parsing. Não sabe o nome do ficheiro,
não extrai metadados, não faz I/O.
"""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from utils.regex_patterns import CAP_PATTERN, ART_PATTERN

# Categorias descartadas — ruído estrutural da API
_DISCARD: frozenset[str] = frozenset({
    "Header", "Footer", "PageBreak", "PageNumber",
})

# Categorias que transportam conteúdo semântico (texto de artigo)
_CONTENT: frozenset[str] = frozenset({
    "NarrativeText", "ListItem", "Table", "UncategorizedText",
})

_MAX_PREAMBLE_BLOCKS = 10
_NO_TITLE            = "Sem Título"


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
        result = {
            "preambulo": "",
            "estrutura": {},
        }

        current_cap_id: str | None = None
        current_art_id: str | None = None
        preamble_blocks: list[str] = []
        preamble_closed = False

        for el in elements:
            category = el.get("category", "")
            text     = el.get("text", "").strip()
            page     = int(el.get("page", 1))

            if not text or category in _DISCARD:
                continue

            # ── ÂNCORA: CAPÍTULO ─────────────────────────────────────────────
            # Aplica-se a qualquer categoria — o regex é a guarda suficiente.
            # "CAPÍTULO I" no início de linha nunca aparece a meio de um parágrafo.
            cap_m = CAP_PATTERN.match(text)
            if cap_m:
                cap_label      = cap_m.group(1).upper()
                current_cap_id = f"CAP_{cap_label}"
                current_art_id = None
                preamble_closed = True
                result["estrutura"][current_cap_id] = {
                    "titulo":  text,
                    "artigos": {},
                }
                continue

            # ── ÂNCORA: ARTIGO ────────────────────────────────────────────────
            # Aplica-se a qualquer categoria — o regex é a guarda suficiente.
            # ART_PATTERN exige "^\\s*Artigo\\s+\\d+" — nunca bate a meio de frase.
            art_m = ART_PATTERN.match(text)
            if art_m:
                art_num      = art_m.group(1)
                inline_title = art_m.group(2).strip(" .º°-")
                current_art_id = f"ART_{art_num}"
                preamble_closed = True

                # documento sem capítulos → CAP_0 implícito (criado uma vez)
                if current_cap_id is None:
                    current_cap_id = "CAP_0"
                    result["estrutura"]["CAP_0"] = {
                        "titulo":  "Sem Capítulos",
                        "artigos": {},
                    }

                result["estrutura"][current_cap_id]["artigos"][current_art_id] = {
                    "titulo":   inline_title or _NO_TITLE,
                    "conteudo": "",
                    "pagina":   page,
                }
                continue

            # ── TÍTULO SOLTO ──────────────────────────────────────────────────
            # Um Title que vem após uma âncora de artigo e o artigo ainda não
            # tem título resolvido. Ex: "Artigo 1.º" + (próxima linha) "ÂMBITO"
            if category == "Title" and current_art_id is not None:
                art = result["estrutura"][current_cap_id]["artigos"][current_art_id]
                if art["titulo"] == _NO_TITLE:
                    art["titulo"] = text
                continue

            # ── CONTEÚDO ÚTIL ─────────────────────────────────────────────────
            if category in _CONTENT:
                if current_art_id is not None:
                    art = result["estrutura"][current_cap_id]["artigos"][current_art_id]
                    art["conteudo"] += (" " if art["conteudo"] else "") + text
                elif not preamble_closed:
                    if len(preamble_blocks) < _MAX_PREAMBLE_BLOCKS:
                        preamble_blocks.append(text)

        result["preambulo"] = " ".join(preamble_blocks)
        return result