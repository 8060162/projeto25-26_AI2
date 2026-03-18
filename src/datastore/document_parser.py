"""
document_parser.py
------------------
Lê os ficheiros JSON processados e devolve uma lista de objectos `Artigo`.

Cada `Artigo` carrega toda a informação necessária para:
  1. Construir o texto do chunk (apenas `conteudo`).
  2. Preencher os metadados de rastreabilidade no ChromaDB.
  3. Recuperar o artigo completo a partir do JSON de origem quando
     o chunk está marcado como `truncated`.
"""

import json
from dataclasses import dataclass, field


@dataclass
class Artigo:
    """Representa um artigo extraído do JSON processado."""

    # --- Identificação do documento de origem ---
    filename: str           # nome do ficheiro JSON (para retrieval do original)
    doc_titulo: str         # título do documento (ex.: "Regulamento de Propinas")
    doc_numero: str         # número/referência oficial (ex.: "Despacho P.PORTO-P-043/2025")
    doc_data: str           # data de publicação

    # --- Localização dentro do documento ---
    cap_id: str             # chave do capítulo no JSON (ex.: "capitulo_1")
    cap_titulo: str         # título legível do capítulo
    art_id: str             # identificador do artigo (ex.: "Artigo 5.º")
    art_titulo: str         # título do artigo

    # --- Conteúdo e localização física ---
    conteudo: str           # texto integral do artigo
    pagina: str             # página no documento original


def _extrair_doc_info(data: dict) -> tuple[str, str, str]:
    """Extrai título, número e ano do bloco `document_info`.

    Suporta variações de nomes de campos observadas nos JSONs do P.PORTO:
      - doc_id  →  usado como número/referência do documento
      - ano     →  ano de publicação
      - titulo / title / nome  →  título legível
    """
    info = data.get("document_info", {})
    # Número/referência oficial (ex.: "Despacho P.PORTO-P-043-2025_Regulamento de Propinas")
    numero = info.get("doc_id", info.get("numero", info.get("referencia", "")))
    # Título legível — o doc_id pode servir de fallback se não houver campo título separado
    titulo = info.get("titulo", info.get("title", info.get("nome", numero)))
    # Data / ano de publicação
    data_pub = info.get("data", info.get("date", info.get("ano", "")))
    return titulo, numero, str(data_pub)


def parse_ficheiro(filepath: str, filename: str) -> list[Artigo]:
    """
    Lê um ficheiro JSON e devolve lista de Artigos.

    Args:
        filepath: caminho absoluto para o ficheiro JSON.
        filename: nome do ficheiro (usado como referência nos metadados).

    Raises:
        json.JSONDecodeError: se o ficheiro não for JSON válido.
        FileNotFoundError:    se o ficheiro não existir.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    doc_titulo, doc_numero, doc_data = _extrair_doc_info(data)

    artigos: list[Artigo] = []
    estrutura = data.get("estrutura", {})

    for cap_id, cap_data in estrutura.items():
        cap_titulo = cap_data.get("titulo", "Sem Título")

        for art_id, art_data in cap_data.get("artigos", {}).items():
            artigos.append(
                Artigo(
                    filename=filename,
                    doc_titulo=doc_titulo,
                    doc_numero=doc_numero,
                    doc_data=doc_data,
                    cap_id=cap_id,
                    cap_titulo=cap_titulo,
                    art_id=art_id,
                    art_titulo=art_data.get("titulo", ""),
                    conteudo=art_data.get("conteudo", ""),
                    pagina=str(art_data.get("pagina", "N/A")),
                )
            )

    return artigos