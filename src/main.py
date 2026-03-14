import os
import sys
from pathlib import Path

# Adiciona o diretório atual ao sys.path para garantir que o módulo indexer seja encontrado
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from indexer.batch_processor import process_batch

def start_indexing():
    # .parent é 'src', .parent.parent é a raiz 'projeto25-26_AI2'
    base_dir = Path(__file__).resolve().parent.parent 
    
    input_folder = base_dir / "data" / "raw"
    output_folder = base_dir / "data" / "processed"

    # Resto do código...
    print(f"input_folder: {input_folder}")
    print(f"output_folder: {output_folder}")
    
    # Chama o maestro
    report = process_batch(input_folder, output_folder)

    # Exibe o sumário final
    print("\n── Relatório Final ────────────────────────")
    print(f"  Sucesso : {len(report['sucesso'])}")
    print(f"  Falha   : {len(report['falha'])}")
    print(f"  Ignorado: {len(report['ignorado'])}")
    
    if report["falha"]:
        print("\nDetalhamento de erros:")
        for item in report["falha"]:
            print(f"    • {item['ficheiro']}: {item['erro']}")
    print("───────────────────────────────────────────")

if __name__ == "__main__":
    start_indexing()