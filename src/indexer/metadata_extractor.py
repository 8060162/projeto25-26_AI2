import os
from utils.regex_patterns import YEAR_FILENAME_PATTERN, YEAR_PREAMBLE_PATTERN

class MetadataExtractor:
    @staticmethod
    def extract(preamble_text: str, filename: str) -> dict:
        metadata = {
            "doc_id": os.path.splitext(filename)[0],
            "escola": "GERAL",
            "ano": "N/A",
            "status": "VIGENTE"
        }

        # 1. Tenta extrair ano do nome do ficheiro (mais fiável)
        ano_filename = YEAR_FILENAME_PATTERN.search(filename)
        if ano_filename:
            metadata["ano"] = ano_filename.group(0)
            return metadata

        # 2. Fallback: Procura no texto do preâmbulo
        ano_match = YEAR_PREAMBLE_PATTERN.search(preamble_text)
        if ano_match:
            metadata["ano"] = ano_match.group(1)

        return metadata