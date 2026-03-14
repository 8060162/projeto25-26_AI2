"""
batch_processor.py
Lê todos os PDFs de uma pasta de entrada, faz o parse de cada um
e guarda o JSON resultante na pasta de saída com o mesmo nome base.

Uso:
    python -m indexer.batch_processor                         # defaults
    python -m indexer.batch_processor --input data/raw --output data/processed
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Garante que a raiz do projecto está no path, independentemente de como
# o script é invocado (python file.py, python -m, ou via IDE)
# __file__ = .../src/indexer/batch_processor.py
# parent       = .../src/indexer/
# parent.parent = .../src/              ← aqui está o módulo "indexer"
_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from indexer.structure_pipeline import StructurePipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def collect_pdfs(input_dir: Path) -> list[Path]:
    """Devolve lista ordenada de PDFs encontrados na pasta de entrada."""
    pdfs = sorted(input_dir.glob("*.pdf"))
    if not pdfs:
        log.warning("Nenhum PDF encontrado em '%s'", input_dir)
    return pdfs


def process_batch(input_dir: Path, output_dir: Path) -> dict:
    """
    Processa todos os PDFs da pasta de entrada.
    Devolve um relatório com os resultados: sucesso, falha e ficheiros ignorados.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline = StructurePipeline()

    pdfs = collect_pdfs(input_dir)
    log.info("PDFs encontrados: %d", len(pdfs))

    report = {"sucesso": [], "falha": [], "ignorado": []}

    for pdf_path in pdfs:
        output_path = output_dir / (pdf_path.stem + ".json")

        # Salta ficheiros já processados
        if output_path.exists():
            log.info("Ignorado (já existe): %s", output_path.name)
            report["ignorado"].append(pdf_path.name)
            continue

        log.info("A processar: %s", pdf_path.name)
        try:
            result = pipeline.run(str(pdf_path))

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=4, ensure_ascii=False)

            log.info("  → Guardado: %s", output_path.name)
            report["sucesso"].append(pdf_path.name)

        except Exception as exc:
            log.error("  → Falhou: %s | %s", pdf_path.name, exc)
            report["falha"].append({"ficheiro": pdf_path.name, "erro": str(exc)})

    return report


def main():
    parser = argparse.ArgumentParser(description="Batch PDF → JSON parser")
    parser.add_argument(
        "--input", default="data/raw",
        help="Pasta com os PDFs a processar (default: data/raw)"
    )
    parser.add_argument(
        "--output", default="data/processed",
        help="Pasta de destino dos JSONs (default: data/processed)"
    )
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        log.error("Pasta de entrada não encontrada: '%s'", input_dir)
        sys.exit(1)

    report = process_batch(input_dir, output_dir)

    print("\n── Relatório ──────────────────────────────")
    print(f"  Sucesso : {len(report['sucesso'])}")
    print(f"  Falha   : {len(report['falha'])}")
    print(f"  Ignorado: {len(report['ignorado'])}")
    if report["falha"]:
        print("\n  Ficheiros com erro:")
        for item in report["falha"]:
            print(f"    • {item['ficheiro']}: {item['erro']}")
    print("───────────────────────────────────────────\n")


if __name__ == "__main__":
    main()