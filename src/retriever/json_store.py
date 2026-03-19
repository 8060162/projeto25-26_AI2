"""
json_store.py
-------------
Responsabilidade única: ler o conteúdo completo de um artigo
a partir dos ficheiros JSON de origem.
"""

import json
import logging
import os

from retriever.settings import JSON_FOLDER

logger = logging.getLogger(__name__)


def get_artigo_completo(source: str, artigo_id: str) -> str | None:
    """
    Devolve o conteúdo completo de um artigo a partir do ficheiro JSON.

    Args:
        source:    nome do ficheiro JSON (ex: "regulamento_2024.json")
        artigo_id: identificador do artigo (ex: "artigo_12")

    Returns:
        Conteúdo textual do artigo, ou None se não encontrado.

    Raises:
        Nunca — erros são registados e devolvem None.
    """
    filepath = os.path.join(JSON_FOLDER, source)

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        logger.warning("Ficheiro JSON não encontrado: %s", filepath)
        return None
    except json.JSONDecodeError as exc:
        logger.error("JSON inválido em '%s': %s", filepath, exc)
        return None

    for cap in data.get("estrutura", {}).values():
        artigos = cap.get("artigos", {})
        if artigo_id in artigos:
            return artigos[artigo_id].get("conteudo") or None

    logger.debug("Artigo '%s' não encontrado em '%s'.", artigo_id, source)
    return None