"""
analysis/visualize.py
----------------------
Visualização interactiva dos embeddings no Spotlight (Renumics).

Responsabilidade única: carregar o DataFrame e abrir o Spotlight.
Toda a lógica de acesso ao ChromaDB está em _chromadb_loader.py.

Uso:
    python -m src.analysis.visualize

Requisito:
    pip install renumics-spotlight

Browser:
    http://localhost:7860
"""

import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    try:
        from renumics import spotlight
    except ImportError:
        logger.error("Spotlight não instalado. Executa: pip install renumics-spotlight")
        sys.exit(1)

    from _chromadb_loader import carregar_dataframe

    df = carregar_dataframe()

    logger.info("A abrir o Spotlight em http://localhost:7860 ...")
    logger.info("Filtra por 'doc_titulo', 'capitulo' ou 'truncated' para inspecionar grupos.")
    logger.info("Ctrl+C para terminar.")

    spotlight.show(
        df,
        dtype={"embedding": spotlight.Embedding},
        port=7860,
        wait=True,
    )


if __name__ == "__main__":
    main()