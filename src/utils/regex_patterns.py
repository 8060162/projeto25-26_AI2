import re

# ESTRUTURA
CAP_PATTERN = re.compile(r"^\s*CAP[IÍ]TULO\s+([IVXLCDM\d]+)", re.IGNORECASE)
# Captura "Artigo 1", "Artigo 1.", "Artigo 1.º", "ARTIGO 1 -", etc.
ART_PATTERN = re.compile(r"^\s*Artigo\s+(\d+)[.º°o\s\-]*", re.IGNORECASE)

# RUÍDO (Cabeçalhos e Rodapés)
PAGE_NOISE = re.compile(r"Diário da República|Pág\.|Regulamento n\.º|P\.PORTO/P-", re.IGNORECASE)

# METADADOS
YEAR_FILENAME_PATTERN = re.compile(r"20\d{2}")
YEAR_PREAMBLE_PATTERN = re.compile(r"(?:ano\s+letivo|ano)\s+(\d{4})", re.IGNORECASE)