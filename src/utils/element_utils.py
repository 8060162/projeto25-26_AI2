"""
utils/element_utils.py
Utilitários partilhados para normalização de elementos da Unstructured API.

Single Source of Truth para a conversão do formato raw da API
para o formato interno do projecto.
"""


def normalise_elements(raw: list) -> list[dict]:
    """
    Converte a resposta raw da Unstructured API para o formato interno.

    Contrato de saída — cada elemento é um dict com exactamente três chaves:
        {
            "text":     str,  # conteúdo textual do bloco, já strip()
            "category": str,  # tipo Unstructured: Title, NarrativeText, …
            "page":     int   # página de origem, 1-based
        }

    Elementos com texto vazio (após strip) são descartados.

    Args:
        raw: Lista de elementos tal como devolvida pela Unstructured API.

    Returns:
        Lista filtrada e normalizada de elementos.
    """
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