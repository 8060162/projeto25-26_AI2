import re
import os


class MetadataExtractor:
    @staticmethod
    def extract(preamble_text, filename):
        metadata = {
            "doc_id": os.path.splitext(filename)[0],
            "escola": "GERAL",
            "ano": "N/A",
            "status": "VIGENTE"
        }

        # Tenta primeiro extrair o ano do próprio nome do ficheiro (mais fiável)
        ano_filename = re.search(r"20\d{2}", filename)
        if ano_filename:
            metadata["ano"] = ano_filename.group(0)
            return metadata

        # Fallback: procura o ano próximo de palavras-chave de datação no preâmbulo
        # Evita anos de referências legislativas no meio do texto
        ano_match = re.search(
            r"(?:aprovado|publicado|emitido|data(?:do)?)\s+(?:em\s+)?(?:\d+\s+de\s+\w+\s+de\s+)?(20\d{2})",
            preamble_text,
            re.IGNORECASE
        )
        if ano_match:
            metadata["ano"] = ano_match.group(1)

        return metadata