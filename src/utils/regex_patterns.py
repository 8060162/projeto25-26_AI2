import re

CAP_PATTERN = re.compile(
    r"CAP[IÍ]TULO\s+([IVXLCDM]+)",
    re.IGNORECASE
)

ART_PATTERN = re.compile(
    r"Artigo\s+(\d+)[.ºo]?",
    re.IGNORECASE
)
# Justificação: Alguns capítulos têm títulos em linhas separadas.
TITULO_PATTERN = re.compile(r"^[A-ZÁÀÂÃÉÈÊÍÓÒÔÕÚÇ\s]{5,}$")