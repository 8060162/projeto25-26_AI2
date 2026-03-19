"""
dev_loader.py
Implementação de LoaderProtocol para desenvolvimento local.

Lê respostas da Unstructured API previamente guardadas em disco
(formato JSON raw da API), evitando chamadas repetidas durante o
desenvolvimento e poupando as páginas do plano gratuito.

Uso:
    1. Guarda a resposta raw da API:
           import json, requests
           resp = requests.post(...)
           Path("data/raw_elements/doc.json").write_text(
               json.dumps(resp.json(), ensure_ascii=False, indent=2)
           )

    2. Substitui PDFLoader na Pipeline:
           pipeline = Pipeline(loader=DevLoader("data/raw_elements"))
"""

import json
import os

from utils.element_utils import normalise_elements


class DevLoader:
    """Carrega elementos a partir de respostas da API guardadas em disco."""

    def __init__(self, elements_dir: str):
        self._dir = elements_dir

    def load(self, file_path: str) -> list[dict]:
        """
        Lê o JSON correspondente ao PDF e devolve os elementos normalizados.

        O ficheiro JSON esperado tem o mesmo stem que o PDF:
            data/raw_elements/<stem>.json

        Raises:
            FileNotFoundError: Se o JSON correspondente não existir em elements_dir.
        """
        stem      = os.path.splitext(os.path.basename(file_path))[0]
        json_path = os.path.join(self._dir, stem + ".json")

        if not os.path.exists(json_path):
            raise FileNotFoundError(
                f"[DevLoader] JSON não encontrado para '{file_path}'. "
                f"Esperado em: '{json_path}'. "
                f"Guarda primeiro a resposta da API com o nome '{stem}.json' "
                f"na pasta '{self._dir}'."
            )

        with open(json_path, encoding="utf-8") as fh:
            raw = json.load(fh)

        return normalise_elements(raw)