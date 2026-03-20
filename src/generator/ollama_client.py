"""
ollama_client.py
----------------
Cliente Ollama local.
Responsabilidade única: enviar prompt e devolver texto da resposta.
"""

import ollama

from settings import OLLAMA_MODEL


def chamar_modelo(prompt_sistema: str, prompt_utilizador: str) -> str:
    """Envia o prompt ao Ollama e devolve o texto da resposta.

    Raises:
        RuntimeError: Se a chamada ao Ollama falhar.
    """
    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system",  "content": prompt_sistema},
                {"role": "user",    "content": prompt_utilizador},
            ],
            options={
                "temperature": 0.1,
                "num_ctx":     4096,
            },
        )
        return response["message"]["content"]

    except Exception as e:
        raise RuntimeError(f"Erro ao contactar o modelo Ollama '{OLLAMA_MODEL}': {e}") from e