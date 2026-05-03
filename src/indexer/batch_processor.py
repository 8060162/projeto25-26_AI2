"""
batch_processor.py  (v2 — retry, logging estruturado, relatório de qualidade)
------------------------------------------------------------------------------
Processa todos os PDFs de uma pasta de entrada e guarda um JSON por ficheiro.

Novidades v2:
  - Retry automático com backoff exponencial (--retries N)
  - Relatório de qualidade no final (score médio, páginas OCR, etc.)
  - Modo --force-ocr para forçar OCR em todos os PDFs
  - Modo --reprocess para ignorar JSONs já existentes e reprocessar
  - Log estruturado com timestamps e contexto de ficheiro

Uso:
    python batch_processor.py
    python batch_processor.py --input data/raw --output data/processed
    python batch_processor.py --dev --dev-dir data/raw_elements
    python batch_processor.py --force-ocr
    python batch_processor.py --retries 3 --reprocess
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# ── Path setup ─────────────────────────────────────────────────────────────
_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# ── .env ────────────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(_SRC.parent / ".env")
except ImportError:
    pass

from pdf_indexer  import PDFIndexer
from pdf_loader   import PDFLoader
from dev_loader      import DevLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("batch")


# ── Processamento ────────────────────────────────────────────────────────────

def _process_with_retry(
    pipeline: PDFIndexer,
    pdf_path: Path,
    max_retries: int,
) -> dict:
    """
    Executa pipeline.run() com retry exponencial.

    Raises a última excepção se todos os retries falharem.
    """
    last_exc = None
    for attempt in range(1, max_retries + 2):  # +2: tentativa inicial + N retries
        try:
            return pipeline.run(str(pdf_path))
        except Exception as exc:
            last_exc = exc
            if attempt <= max_retries:
                wait = 2 ** attempt  # backoff: 2s, 4s, 8s, …
                log.warning(
                    "  [tentativa %d/%d] Falhou: %s — aguarda %ds",
                    attempt, max_retries + 1, exc, wait
                )
                time.sleep(wait)
    raise last_exc


def run_batch(
    input_dir:   Path,
    output_dir:  Path,
    pipeline:    PDFIndexer,
    reprocess:   bool = False,
    max_retries: int  = 1,
) -> dict:
    """
    Itera os PDFs de input_dir, processa cada um e guarda JSON em output_dir.

    Args:
        input_dir:   pasta com PDFs de entrada.
        output_dir:  pasta de saída para JSONs.
        pipeline:    instância de PDFIndexer configurada.
        reprocess:   se True, ignora JSONs existentes e reprocessa.
        max_retries: número de retries em caso de falha.

    Returns:
        Relatório de execução com métricas de qualidade.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(input_dir.glob("*.pdf"))
    if not pdfs:
        log.warning("Nenhum PDF encontrado em '%s'", input_dir)

    report: dict = {
        "sucesso":  [],
        "falha":    [],
        "ignorado": [],
        "qualidade": [],  # métricas por ficheiro processado com sucesso
    }

    for pdf_path in pdfs:
        out_path = output_dir / (pdf_path.stem + ".json")

        if out_path.exists() and not reprocess:
            log.info("Ignorado (já existe): %s  [usa --reprocess para forçar]", out_path.name)
            report["ignorado"].append(pdf_path.name)
            continue

        log.info("━━ A processar: %s", pdf_path.name)
        t0 = time.monotonic()

        try:
            result = _process_with_retry(pipeline, pdf_path, max_retries)
            elapsed = time.monotonic() - t0

            out_path.write_text(
                json.dumps(result, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            # Recolhe métricas de qualidade
            meta    = result.get("_meta", {})
            ext     = meta.get("extraction_info", {})
            flags   = meta.get("quality_flags", {})

            log.info(
                "  ✓ Guardado: %s  [%.1fs | %d artigos | qualidade=%.2f%s]",
                out_path.name,
                elapsed,
                ext.get("total_artigos", 0),
                ext.get("avg_quality", 1.0),
                " | OCR" if flags.get("ocr_was_used") else "",
            )

            report["sucesso"].append(pdf_path.name)
            report["qualidade"].append({
                "ficheiro":    pdf_path.name,
                "estrategia":  ext.get("strategy", "?"),
                "artigos":     ext.get("total_artigos", 0),
                "qualidade":   ext.get("avg_quality", 1.0),
                "ocr_paginas": ext.get("ocr_pages", []),
                "tempo_s":     round(elapsed, 1),
            })

        except Exception as exc:
            elapsed = time.monotonic() - t0
            log.error("  ✗ Falhou: %s | %s  [%.1fs]", pdf_path.name, exc, elapsed)
            report["falha"].append({"ficheiro": pdf_path.name, "erro": str(exc)})

    return report


# ── Relatório final ──────────────────────────────────────────────────────────

def _print_report(report: dict) -> None:
    q = report.get("qualidade", [])
    avg_q = sum(r["qualidade"] for r in q) / max(len(q), 1) if q else 0.0
    ocr_count = sum(1 for r in q if r.get("ocr_paginas")) if q else 0

    print()
    print("╔══════════════════════════════════════════════╗")
    print("║              RELATÓRIO DE BATCH              ║")
    print("╠══════════════════════════════════════════════╣")
    print(f"║  ✓ Sucesso  : {len(report['sucesso']):>3}                           ║")
    print(f"║  ✗ Falha    : {len(report['falha']):>3}                           ║")
    print(f"║  ↷ Ignorado : {len(report['ignorado']):>3}                           ║")
    print(f"║  ◎ Qualidade média : {avg_q:.2f}                    ║")
    print(f"║  ⊕ PDFs com OCR    : {ocr_count:>3}                           ║")
    print("╚══════════════════════════════════════════════╝")

    if report["falha"]:
        print("\n  Ficheiros com erro:")
        for item in report["falha"]:
            print(f"    • {item['ficheiro']}: {item['erro']}")

    if q:
        print("\n  Detalhe por ficheiro processado:")
        for r in q:
            ocr_tag = f"  [OCR: pgs {r['ocr_paginas']}]" if r.get("ocr_paginas") else ""
            print(
                f"    • {r['ficheiro']}"
                f"  estratégia={r['estrategia']}"
                f"  artigos={r['artigos']}"
                f"  q={r['qualidade']:.2f}"
                f"  {r['tempo_s']}s"
                f"{ocr_tag}"
            )
    print()


# ── CLI ──────────────────────────────────────────────────────────────────────

def _build_loader(args: argparse.Namespace):
    if args.dev:
        log.info("Modo dev: DevLoader (dir='%s')", args.dev_dir)
        return DevLoader(args.dev_dir)

    return PDFLoader(
        quality_threshold=args.quality_threshold,
        force_ocr=args.force_ocr,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch PDF → JSON indexer (v2 com OCR fallback)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input",  default="data/raw",       help="Pasta de PDFs de entrada")
    parser.add_argument("--output", default="data/processed", help="Pasta de saída JSON")
    parser.add_argument("--dev",    action="store_true",      help="Usa DevLoader (sem API)")
    parser.add_argument("--dev-dir", default="data/raw_elements", help="Pasta JSONs raw para DevLoader")
    parser.add_argument("--force-ocr", action="store_true",   help="Força OCR local (ignora API)")
    parser.add_argument("--reprocess", action="store_true",   help="Reprocessa JSONs existentes")
    parser.add_argument("--retries", type=int, default=1,     help="Número de retries em falha")
    parser.add_argument(
        "--quality-threshold", type=float, default=0.45,
        help="Score mínimo para aceitar texto da API (0–1)"
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
    report   = run_batch(
        input_dir, output_dir, pipeline,
        reprocess=args.reprocess,
        max_retries=args.retries,
    )
    _print_report(report)


if __name__ == "__main__":
    main()