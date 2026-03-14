import json
from dataclasses import dataclass


@dataclass
class Artigo:
    """Representa um artigo extraído do JSON processado."""
    filename: str
    cap_titulo: str
    art_id: str
    art_titulo: str
    conteudo: str
    pagina: str


def parse_ficheiro(filepath: str, filename: str) -> list[Artigo]:
    """Lê um ficheiro JSON e devolve lista de Artigos.

    Raises:
        json.JSONDecodeError: Se o ficheiro não for JSON válido.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    artigos = []
    estrutura = data.get("estrutura", {})

    for cap_data in estrutura.values():
        cap_titulo = cap_data.get("titulo", "Sem Título")

        for art_id, art_data in cap_data.get("artigos", {}).items():
            artigos.append(Artigo(
                filename=filename,
                cap_titulo=cap_titulo,
                art_id=art_id,
                art_titulo=art_data.get("titulo", ""),
                conteudo=art_data.get("conteudo", ""),
                pagina=str(art_data.get("pagina", "N/A")),
            ))

    return artigos