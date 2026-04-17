"""
main.py
-------
Ponto de entrada principal do sistema RAG P.PORTO.

Expõe dois modos de operação via CLI:
  query   — modo interactivo de perguntas e respostas
  ingest  — pipeline de ingestão de documentos JSON para ChromaDB

Uso:
    python main.py query
    python main.py ingest

ALTERAÇÃO (refactor): main.py estava vazio. A sua ausência tornava o
ponto de entrada do projecto opaco para novos colaboradores, que tinham
de ler a documentação interna dos módulos para saber como executar o sistema.
"""

import argparse
import sys


def _cmd_query(_args: argparse.Namespace) -> None:
    """Modo interactivo: pergunta → resposta RAG."""
    from generator.generator import gerar_resposta, GeneratorError
    from settings import LLM_BACKEND, MODEL_DISPLAY_NAME

    print("\n" + "═" * 60)
    print(f"  RAG ACTIVO  |  backend: {LLM_BACKEND}  |  modelo: {MODEL_DISPLAY_NAME}")
    print("═" * 60)

    while True:
        try:
            pergunta = input("\nQuestão (ou 'sair'): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nA sair.")
            break

        if pergunta.lower() in ["sair", "q"]:
            break
        if not pergunta:
            continue

        print("\n[A processar...]")
        try:
            resposta = gerar_resposta(pergunta)
            print("\n" + "─" * 20 + " RESPOSTA " + "─" * 20)
            print(resposta)
        except GeneratorError as e:
            print(f"\n[ERRO] {e}", file=sys.stderr)
        print("─" * 50)


def _cmd_ingest(_args: argparse.Namespace) -> None:
    """Ingestão de documentos JSON para ChromaDB."""
    from datastore.ingest import run_ingestion
    run_ingestion()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sistema RAG P.PORTO — consulta de regulamentos académicos.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Exemplos:\n"
            "  python main.py query    # modo interactivo\n"
            "  python main.py ingest   # indexar documentos JSON\n"
        ),
    )
    sub = parser.add_subparsers(dest="comando", required=True)
    sub.add_parser("query",  help="Modo interactivo de perguntas e respostas")
    sub.add_parser("ingest", help="Pipeline de ingestão de documentos para ChromaDB")
    return parser


def main() -> None:
    parser  = _build_parser()
    args    = parser.parse_args()
    comandos = {
        "query":  _cmd_query,
        "ingest": _cmd_ingest,
    }
    comandos[args.comando](args)


if __name__ == "__main__":
    main()