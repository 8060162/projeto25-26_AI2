import json
import requests

# --- CONFIGURAÇÃO ---
ENDPOINT = "https://api.iaedu.pt/agent-chat//api/v1/agent/cmamvd3n40000c801qeacoad2/stream"
API_KEY = "sk-usr-rbpxai2ruuivighi9yhf7ra8dxbbs0gq87"
CHANNEL_ID = "cmmpjioe255mehv012ur51uss"
THREAD_ID = "9K-tgj8aKn7O5qw-4hnou"

MODEL_NAME = "iaedu-agent"


def chamar_modelo(prompt_sistema: str, prompt_utilizador: str) -> str:
    """Envia o prompt ao endpoint iaedu e devolve o texto da resposta.

    Interface idêntica ao ollama_client.chamar_modelo — o generator.py
    não precisa de qualquer alteração ao trocar de cliente.
    """
    mensagem_completa = f"{prompt_sistema.strip()}\n\n{prompt_utilizador.strip()}"

    response = requests.post(
        ENDPOINT,
        headers={"x-api-key": API_KEY},
        files={
            "channel_id": (None, CHANNEL_ID),
            "thread_id":  (None, THREAD_ID),
            "user_info":  (None, "{}"),
            "message":    (None, mensagem_completa),
        },
    )
    response.raise_for_status()
    return _parse_stream(response.text)


def _parse_stream(raw: str) -> str:
    """Extrai o texto da resposta do stream JSON-lines do iaedu.

    O stream tem três tipos de eventos relevantes:
      - type "token"   — fragmentos de texto a chegar em tempo real
      - type "message" — objecto completo com a resposta final montada
      - type "done"    — sinal de fim de stream

    Usamos o evento "message" como fonte de verdade — já vem com o
    texto completo e bem formado, sem necessidade de concatenar tokens.
    Se por algum motivo não existir, fazemos fallback pelos tokens.
    """
    tokens = []
    for linha in raw.splitlines():
        linha = linha.strip()
        if not linha:
            continue
        try:
            evento = json.loads(linha)
        except json.JSONDecodeError:
            continue

        tipo = evento.get("type")

        if tipo == "message":
            conteudo = evento.get("content", {})
            texto = conteudo.get("content", "")
            if texto:
                return texto

        elif tipo == "token":
            tokens.append(evento.get("content", ""))

    # Fallback: concatenação dos tokens individuais
    return "".join(tokens).strip()