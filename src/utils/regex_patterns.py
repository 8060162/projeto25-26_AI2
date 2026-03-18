"""
regex_patterns.py
Padrões de expressão regular usados no parsing de regulamentos do P.PORTO.

Mantidos num módulo único para facilitar ajustes e testes isolados.
"""

import re

# ── âncoras estruturais ───────────────────────────────────────────────────────

# "CAPÍTULO I", "CAPITULO IV", "CAPÍTULO 1", …
CAP_PATTERN = re.compile(
    r"^\s*CAP[IÍ]TULO\s+([IVXLCDM\d]+)",
    re.IGNORECASE,
)

# "Artigo 1.º", "Artigo 1.", "Artigo 1 -", "Artigo 12.º —", …
# Grupo 1 → número do artigo (só dígitos)
# Grupo 2 → resto da linha (possível título inline), pode ser vazio
ART_PATTERN = re.compile(
    r"^\s*Artigo\s+(\d+)[.º°\s\-]*(.*)$",
    re.IGNORECASE,
)

# ── metadados ─────────────────────────────────────────────────────────────────

# Ano no nome do ficheiro: 2023, 2024, 2025, …
YEAR_FILENAME = re.compile(r"20\d{2}")

# Ano no preâmbulo: "ano letivo 2024", "ano 2023"
YEAR_PREAMBLE = re.compile(
    r"(?:ano\s+letivo|ano)\s+(\d{4})",
    re.IGNORECASE,
)