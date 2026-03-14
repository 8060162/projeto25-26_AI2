def formatar_contexto(results) -> str:
    """Formata os resultados do ChromaDB numa string de contexto para o prompt."""
    contexto_acumulado = ""
    for i in range(len(results['documents'][0])):
        texto = results['documents'][0][i]
        meta = results['metadatas'][0][i]

        artigo = meta.get('artigo_id', 'N/A')
        pagina = meta.get('pagina', 'N/A')
        doc = meta.get('source', 'Documento')

        contexto_acumulado += f"\n[FONTE: {doc} | Artigo {artigo} | Pág. {pagina}]\n{texto}\n"
    return contexto_acumulado


def construir_prompt(pergunta: str, contexto: str) -> tuple[str, str]:
    """Devolve o par (prompt_sistema, prompt_utilizador) prontos a enviar ao modelo."""
    prompt_sistema = """
    És um assistente virtual universitário. Responde sempre em Português de Portugal.
    REGRAS:
    1. Responde APENAS com base no CONTEXTO fornecido.
    2. Cita sempre o Artigo e Página (ex: "Segundo o Artigo 5º (pág. 2)...").
    3. Se não souberes, diz que o regulamento não refere o assunto.
    """
    prompt_utilizador = f"CONTEXTO:\n{contexto}\n\nPERGUNTA: {pergunta}"
    return prompt_sistema, prompt_utilizador