import os
import json
import requests
from pathlib import Path
from dotenv import load_dotenv

def executar_teste_v1(pdf_path):
    # 1. Carregar o .env
    load_dotenv()
    
    # Tenta obter a chave. Se falhar, avisa.
    api_key = os.getenv("UNSTRUCTURED_API_KEY")
    # URL que indicaste como estando na tua documentação
    api_url = "https://api.unstructuredapp.io/general/v0/general"

    if not api_key:
        print("[!] Erro: UNSTRUCTURED_API_KEY não encontrada no .env")
        return

    path = Path(pdf_path)
    if not path.exists():
        print(f"[!] Erro: Ficheiro não encontrado: {pdf_path}")
        return

    # 2. Configurar os cabeçalhos e parâmetros
    headers = {
        "Accept": "application/json",
        "unstructured-api-key": api_key
    }

    # Parâmetros para OCR Vision de Alta Resolução
    data = {
        "strategy": "hi_res",
        "pdf_infer_table_structure": "true", # Crucial para extrair tabelas em HTML
        "languages": ["por"],                # Português
        "chunking_strategy": "by_title",     # Tenta manter a lógica de Artigos/Títulos
        "skip_infer_table_types": "[]",      # Garante que tenta extrair todas as tabelas
    }

    print(f"[*] A ligar a: {api_url}")
    print(f"[*] Ficheiro: {path.name}")

    try:
        with open(path, "rb") as f:
            files = {"files": (path.name, f, "application/pdf")}
            
            print("[*] Pedido enviado. A processar visão computacional (pode demorar)...")
            
            response = requests.post(
                api_url, 
                headers=headers, 
                data=data, 
                files=files, 
                timeout=300 # 5 minutos de limite
            )

        # 3. Validar Resposta
        if response.status_code == 200:
            resultado = response.json()
            output_file = path.stem + "_unstructured_v1.json"
            
            with open(output_file, "w", encoding="utf-8") as out:
                json.dump(resultado, out, indent=4, ensure_ascii=False)
            
            print(f"[V] Sucesso! Gerados {len(resultado)} elementos.")
            print(f"[V] Resultado guardado em: {output_file}")
        else:
            print(f"[X] Erro {response.status_code}: {response.text}")

    except requests.exceptions.ConnectionError:
        print("[X] Erro de Rede: Não foi possível resolver o nome 'api.unstructuredapp.io'.")
        print("    Sugestão: Verifica se o teu Mac tem o DNS 8.8.8.8 configurado.")
    except Exception as e:
        print(f"[X] Erro inesperado: {e}")

if __name__ == "__main__":
    # Ajusta para o nome do teu PDF
    nome_pdf = "data/raw/Regulamento nº 633-2024_Alteração ao Regulamento de Avaliação de Aproveitamento.pdf"
    executar_teste_v1(nome_pdf)