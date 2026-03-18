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


class DevLoader:
    """Carrega elementos a partir de respostas da API guardadas em disco."""

    def __init__(self, elements_dir: str):
        self._dir = elements_dir

    def load(self, file_path: str) -> list[dict]:
        stem = os.path.splitext(os.path.basename(file_path))[0]
        json_path = os.path.join(self._dir, stem + ".json")

        with open(json_path, encoding="utf-8") as fh:
            raw = json.load(fh)

        return self._normalise(raw)

    # ── privado ───────────────────────────────────────────────────────────────

    @staticmethod
    def _normalise(raw: list) -> list[dict]:
        """Mesmo contrato de normalização que PDFLoader._normalise."""
        elements = []
        for el in raw:
            text = (el.get("text") or "").strip()
            if not text:
                continue
            elements.append({
                "text":     text,
                "category": el.get("type", "Uncategorized"),
                "page":     (el.get("metadata") or {}).get("page_number", 1) or 1,
            })
        return elements