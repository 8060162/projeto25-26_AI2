"""
openai_client.py
----------------
Cliente HTTP para o endpoint iaedu.
Responsabilidade única: enviar prompt e devolver texto da resposta.

ALTERAÇÃO (refactor): a validação das credenciais obrigatórias foi movida
para cá (de settings.py), aplicando lazy validation — o erro só ocorre
quando o cliente é efectivamente utilizado, não na importação do módulo.
Isto torna settings.py importável em qualquer contexto sem efeitos laterais.
"""

import json
import requests

from settings import IAEDU_ENDPOINT, IAEDU_API_KEY, IAEDU_CHANNEL_ID, IAEDU_THREAD_ID

# Timeout em segundos para a chamada HTTP
_REQUEST_TIMEOUT = 60

# Chaves obrigatórias para o backend OpenAI
_REQUIRED: dict[str, str] = {
    "IAEDU_ENDPOINT":   IAEDU_ENDPOINT,
    "IAEDU_API_KEY":    IAEDU_API_KEY,
    "IAEDU_CHANNEL_ID": IAEDU_CHANNEL_ID,
    "IAEDU_THREAD_ID":  IAEDU_THREAD_ID,
}


def _validar_credenciais() -> None:
    """
    Valida que todas as credenciais obrigatórias estão definidas.

    Chamada no início de chamar_modelo — garante que o erro é explícito
    e ocorre apenas quando o cliente é efectivamente usado.

    Raises:
        EnvironmentError: se alguma variável obrigatória estiver ausente.
    """
    ausentes = [nome for nome, valor in _REQUIRED.items() if not valor]
    if ausentes:
        raise EnvironmentError(
            f"Variáveis de ambiente obrigatórias não definidas para o backend 'openai': "
            f"{', '.join(ausentes)}. Verifica o ficheiro .env."
        )


def chamar_modelo(prompt_sistema: str, prompt_utilizador: str) -> str:
    """Envia o prompt ao endpoint iaedu e devolve o texto da resposta.

    Raises:
        EnvironmentError: se as credenciais obrigatórias não estiverem definidas.
        RuntimeError:     em caso de falha de rede, timeout ou resposta HTTP não-2xx.
        ValueError:       se o stream devolver conteúdo vazio ou não parseável.
    """
    _validar_credenciais()

    mensagem_completa = f"{prompt_sistema.strip()}\n\n{prompt_utilizador.strip()}"

    try:
        response = requests.post(
            IAEDU_ENDPOINT,
            headers={"x-api-key": IAEDU_API_KEY},
            files={
                "channel_id": (None, IAEDU_CHANNEL_ID),
                "thread_id":  (None, IAEDU_THREAD_ID),
                "user_info":  (None, "{}"),
                "message":    (None, mensagem_completa),
            },
            timeout=_REQUEST_TIMEOUT,
        )
        response.raise_for_status()

    except requests.exceptions.Timeout as e:
        raise RuntimeError(
            f"Timeout após {_REQUEST_TIMEOUT}s na chamada ao modelo iaedu."
        ) from e
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Erro de rede ao contactar o modelo iaedu: {e}") from e

    resultado = _parse_stream(response.text)

    if not resultado:
        raise ValueError(
            "O stream do modelo iaedu devolveu uma resposta vazia ou num formato inesperado."
        )

    return resultado


def _parse_stream(raw: str) -> str:
    """Extrai o texto da resposta do stream JSON-lines do iaedu.

    Estratégia:
      1. Procura evento "message" — contém a resposta final completa.
      2. Fallback: concatenação dos eventos "token" individuais.
    """
    tokens: list[str] = []

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
            texto = evento.get("content", {}).get("content", "")
            if texto:
                return texto

        elif tipo == "token":
            tokens.append(evento.get("content", ""))

    return "".join(tokens).strip()