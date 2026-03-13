import os
import sys
import warnings
import ollama  # Importa a biblioteca oficial do Ollama

# 1. Silenciar avisos
warnings.filterwarnings("ignore", category=UserWarning)

# 2. Configuração de Caminhos para Importação
current_file_path = os.path.abspath(__file__)
generator_dir = os.path.dirname(current_file_path)
src_dir = os.path.dirname(generator_dir)

if src_dir not in sys.path:
    sys.path.append(src_dir)

# 3. Importar Retriever
try:
    from retriever.query import procurar_contexto, get_retriever
except ImportError:
    print("ERRO: Não foi possível encontrar 'src/retriever/query.py'.")
    sys.exit(1)

# --- CONFIGURAÇÃO OLLAMA ---
# Podes usar "qwen2.5:7b", "llama3:8b", "mistral", etc.
MODEL_NAME = "qwen2.5:7b" 

def gerar_resposta_rag(pergunta):
    """
    Pipeline RAG Local usando Ollama.
    """
    try:
        # --- ETAPA 1: RETRIEVAL (ChromaDB) ---
        collection = get_retriever()
        results = collection.query(query_texts=[pergunta], n_results=4)
        
        if not results['documents'][0]:
            return "Não encontrei informações nos documentos locais para responder a isso."

        # Formatação do Contexto
        contexto_acumulado = ""
        for i in range(len(results['documents'][0])):
            texto = results['documents'][0][i]
            meta = results['metadatas'][0][i]
            
            artigo = meta.get('artigo_id', 'N/A')
            pagina = meta.get('pagina', 'N/A')
            doc = meta.get('source', 'Documento')
            
            contexto_acumulado += f"\n[FONTE: {doc} | Artigo {artigo} | Pág. {pagina}]\n{texto}\n"

        # --- ETAPA 2: PROMPT ---
        prompt_sistema = """
        És um assistente virtual universitário. Responde sempre em Português de Portugal.
        REGRAS:
        1. Responde APENAS com base no CONTEXTO fornecido.
        2. Cita sempre o Artigo e Página (ex: "Segundo o Artigo 5º (pág. 2)...").
        3. Se não souberes, diz que o regulamento não refere o assunto.
        """

        prompt_utilizador = f"CONTEXTO:\n{contexto_acumulado}\n\nPERGUNTA: {pergunta}"

        # --- ETAPA 3: GENERATION (Ollama Local) ---
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {'role': 'system', 'content': prompt_sistema},
                {'role': 'user', 'content': prompt_utilizador},
            ],
            options={
                'temperature': 0.1,  # Respostas factuais
                'num_ctx': 4096      # Tamanho do contexto (ajusta se necessário)
            }
        )
        
        return response['message']['content']

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