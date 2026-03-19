"""
batch_processor.py
Processa todos os PDFs de uma pasta de entrada e guarda um JSON por ficheiro.

Uso directo:
    python batch_processor.py
    python batch_processor.py --input data/raw --output data/processed

Modo de desenvolvimento (sem chamadas à API):
    python batch_processor.py --dev --dev-dir data/raw_elements

Uso com -m (a partir de src/):
    python -m indexer.batch_processor
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
# ─────────────────────────────────────────────────────────────────────────────

# ── Carrega .env automaticamente (python-dotenv) ──────────────────────────────
try:
    from dotenv import load_dotenv
    _ROOT = _SRC.parent
    load_dotenv(_ROOT / ".env")
except ImportError:
    pass  # dotenv não instalado — variáveis de ambiente têm de ser definidas manualmente
# ─────────────────────────────────────────────────────────────────────────────

from indexer.pdf_indexer import PDFIndexer
from indexer.pdf_loader  import PDFLoader
from indexer.dev_loader  import DevLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def run_batch(input_dir: Path, output_dir: Path, pipeline: PDFIndexer) -> dict:
    """
    Itera os PDFs de input_dir, processa cada um com pipeline e guarda
    o JSON resultante em output_dir/<stem>.json.

    PDFs cujo JSON de destino já existe são ignorados (idempotente).

    Returns:
        {
            "sucesso":  [str, …],
            "falha":    [{"ficheiro": str, "erro": str}, …],
            "ignorado": [str, …],
        }
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(input_dir.glob("*.pdf"))
    if not pdfs:
        log.warning("Nenhum PDF encontrado em '%s'", input_dir)

    report: dict[str, list] = {"sucesso": [], "falha": [], "ignorado": []}

    for pdf_path in pdfs:
        out_path = output_dir / (pdf_path.stem + ".json")

        if out_path.exists():
            log.info("Ignorado (já existe): %s", out_path.name)
            report["ignorado"].append(pdf_path.name)
            continue

        log.info("A processar: %s", pdf_path.name)
        try:
            result = pipeline.run(str(pdf_path))
            out_path.write_text(
                json.dumps(result, indent=4, ensure_ascii=False),
                encoding="utf-8",
            )
            log.info("  → Guardado: %s", out_path.name)
            report["sucesso"].append(pdf_path.name)

        except Exception as exc:
            log.error("  → Falhou: %s | %s", pdf_path.name, exc)
            report["falha"].append({"ficheiro": pdf_path.name, "erro": str(exc)})

    return report


def _build_loader(args: argparse.Namespace) -> PDFLoader | DevLoader:
    """
    Selecciona e instancia o loader adequado com base nos argumentos do CLI.

    Separa a decisão de qual loader usar da lógica de processamento batch,
    tornando ambos testáveis independentemente.
    """
    if args.dev:
        log.info("Modo de desenvolvimento: a usar DevLoader (dir='%s')", args.dev_dir)
        return DevLoader(args.dev_dir)

    return PDFLoader()


def _print_report(report: dict) -> None:
    print("\n── Relatório ──────────────────────────────")
    print(f"  Sucesso : {len(report['sucesso'])}")
    print(f"  Falha   : {len(report['falha'])}")
    print(f"  Ignorado: {len(report['ignorado'])}")
    if report["falha"]:
        print("\n  Ficheiros com erro:")
        for item in report["falha"]:
            print(f"    • {item['ficheiro']}: {item['erro']}")
    print("───────────────────────────────────────────\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch PDF → JSON indexer")
    parser.add_argument(
        "--input",   default="data/raw",
        help="Pasta de PDFs de entrada (default: data/raw)",
    )
    parser.add_argument(
        "--output",  default="data/processed",
        help="Pasta de saída JSON (default: data/processed)",
    )
    parser.add_argument(
        "--dev",     action="store_true",
        help="Usa DevLoader em vez de PDFLoader (sem chamadas à API)",
    )
    parser.add_argument(
        "--dev-dir", default="data/raw_elements",
        help="Pasta com os JSONs raw para o DevLoader (default: data/raw_elements)",
    )
    return parser.parse_args()


def main() -> None:
    args       = _parse_args()
    input_dir  = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        log.error("Pasta de entrada não encontrada: '%s'", input_dir)
        sys.exit(1)

    loader   = _build_loader(args)
    pipeline = PDFIndexer(loader=loader)
    report   = run_batch(input_dir, output_dir, pipeline)
    _print_report(report)


if __name__ == "__main__":
    main()