import ollama

MODEL_NAME = "qwen2.5:7b"


def chamar_modelo(prompt_sistema: str, prompt_utilizador: str) -> str:
    """Envia o prompt ao Ollama e devolve o texto da resposta."""
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {'role': 'system', 'content': prompt_sistema},
            {'role': 'user', 'content': prompt_utilizador},
        ],
        options={
            'temperature': 0.1,
            'num_ctx': 4096
        }
    )
    return response['message']['content']