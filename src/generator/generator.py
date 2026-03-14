#este ficheiro deve ser removido foi restruturado e separado as responsabilidades 
import os
import sys
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# Configuração de caminhos para importação
current_file_path = os.path.abspath(__file__)
generator_dir = os.path.dirname(current_file_path)
src_dir = os.path.dirname(generator_dir)

if src_dir not in sys.path:
    sys.path.append(src_dir)

# Importar Retriever
try:
    from retriever.query import get_retriever
except ImportError:
    print("ERRO: Não foi possível encontrar 'src/retriever/query.py'.")
    sys.exit(1)

from prompt_builder import formatar_contexto, construir_prompt

# Ollama (local)
#from ollama_client import chamar_modelo, MODEL_NAME

# iaedu / OpenAI (cloud)
from openai_client import chamar_modelo, MODEL_NAME


def gerar_resposta_rag(pergunta: str) -> str:
    """Pipeline RAG Local usando Ollama."""
    try:
        # ETAPA 1: RETRIEVAL
        collection = get_retriever()
        results = collection.query(query_texts=[pergunta], n_results=4)

        if not results['documents'][0]:
            return "Não encontrei informações nos documentos locais para responder a isso."

        # ETAPA 2: PROMPT
        contexto = formatar_contexto(results)
        prompt_sistema, prompt_utilizador = construir_prompt(pergunta, contexto)

        # ETAPA 3: GENERATION
        return chamar_modelo(prompt_sistema, prompt_utilizador)

    except Exception as e:
        if "not found" in str(e).lower():
            return f"Erro: O modelo '{MODEL_NAME}' não está instalado no Ollama. Corre 'ollama run {MODEL_NAME}' no terminal."
        return f"Ocorreu um erro no Ollama: {str(e)}"


# --- INTERFACE ---
if __name__ == "__main__":
    print("\n" + "═"*60)
    print(f"  RAG LOCAL ATIVO (Ollama: {MODEL_NAME})")
    print("  Privacidade Total - Sem dependência de API Externa")
    print("═"*60)

    while True:
        duvida = input("\nQuestão (ou 'sair'): ").strip()
        if duvida.lower() in ['sair', 'q']: break
        if not duvida: continue

        print("\n[A processar localmente...]")
        resposta = gerar_resposta_rag(duvida)

        print("\n" + "─"*20 + " RESPOSTA FUNDAMENTADA " + "─"*20)
        print(resposta)
        print("─" * 63)