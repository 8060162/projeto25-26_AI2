"""
json_store.py
-------------
Responsabilidade única: acesso de leitura aos JSONs processados.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datastore.config import JSON_FOLDER


def get_artigo_completo(source: str, artigo_id: str) -> str | None:
    filepath = os.path.join(JSON_FOLDER, source)
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        for cap in data.get("estrutura", {}).values():
            if artigo_id in cap.get("artigos", {}):
                return cap["artigos"][artigo_id].get("conteudo", "")
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return None
    return None