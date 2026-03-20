"""
document_parser.py
------------------
Responsabilidade única: ler ficheiros JSON processados e devolver
uma lista de objectos `Artigo`.

Cada `Artigo`:
  - Transporta o conteúdo e os metadados de rastreabilidade.
  - Expõe `to_metadata()` para desacoplar consumidores da estrutura interna.
  - Expõe `to_chunks_args()` para desacoplar chamadas ao chunker.
"""

import json
import logging
from dataclasses import dataclass

from shared.metadata_keys import MetaKey

logger = logging.getLogger(__name__)


@dataclass
class Artigo:
    """Representa um artigo extraído do JSON processado."""

    # --- Identificação do documento de origem ---
    filename:   str     # nome do ficheiro JSON
    doc_titulo: str     # título do documento
    doc_numero: str     # número/referência oficial
    doc_data:   str     # data de publicação

    # --- Localização dentro do documento ---
    cap_id:     str     # chave do capítulo no JSON (ex.: "capitulo_1")
    cap_titulo: str     # título legível do capítulo
    art_id:     str     # identificador do artigo (ex.: "Artigo 5.º")
    art_titulo: str     # título do artigo

    # --- Conteúdo e localização física ---
    conteudo: str       # texto integral do artigo
    pagina:   str       # página no documento original

    # ── Interface pública ────────────────────────────────────────────────────

    def to_metadata(self) -> dict:
        """
        Devolve os metadados de rastreabilidade prontos a inserir no ChromaDB.

        Centraliza o mapeamento campo→chave usando MetaKey,
        isolando os consumidores de alterações internas ao dataclass.

        Nota: 'art_titulo' foi adicionado para corrigir o mapeamento
        semântico em search.py — anteriormente 'doc_titulo' era atribuído
        erroneamente ao campo artigo_titulo do ArtigoContexto.
        """
        return {
            MetaKey.SOURCE:     self.filename,
            MetaKey.DOC_TITULO: self.doc_titulo,
            MetaKey.ART_TITULO: self.art_titulo,
            MetaKey.CAPITULO:   self.cap_titulo,
            MetaKey.ARTIGO_ID:  self.art_id,
            MetaKey.PAGINA:     self.pagina,
            MetaKey.TRUNCATED:  "false",  # valor base; ingest.py actualiza se necessário
        }

    def to_chunks_args(self) -> dict:
        """
        Devolve os argumentos necessários para chamar `dividir_em_chunks`.

        Evita que ingest.py aceda directamente aos atributos do dataclass
        para construir a chamada ao chunker.
        """
        return {
            "filename":   self.filename,
            "cap_titulo": self.cap_titulo,
            "art_id":     self.art_id,
            "art_titulo": self.art_titulo,
            "conteudo":   self.conteudo,
        }


# ── Parsing ───────────────────────────────────────────────────────────────────

def _extrair_doc_info(data: dict, filepath: str) -> tuple[str, str, str]:
    """
    Extrai título, número e data do bloco `document_info`.

    Suporta variações de nomes de campos observadas nos JSONs do P.PORTO:
      - doc_id   → número/referência do documento
      - ano      → ano de publicação
      - titulo / title / nome → título legível

    Emite aviso se campos críticos estiverem ausentes, para que documentos
    mal formados sejam detectados sem bloquear a ingestão.
    """
    info     = data.get("document_info", {})
    numero   = info.get("doc_id",  info.get("numero",    info.get("referencia", "")))
    titulo   = info.get("titulo",  info.get("title",     info.get("nome", numero)))
    data_pub = str(info.get("data", info.get("date",     info.get("ano", ""))))

    if not numero:
        logger.warning(
            "Campo 'doc_id' ausente em '%s' — verifique o bloco document_info. "
            "O documento será indexado sem referência oficial.",
            filepath,
        )
    if not titulo:
        logger.warning(
            "Campo 'titulo' ausente em '%s' — metadado doc_titulo ficará vazio.",
            filepath,
        )

    return titulo, numero, data_pub


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

    doc_titulo, doc_numero, doc_data = _extrair_doc_info(data, filepath)

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