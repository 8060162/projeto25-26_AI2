"""
Derivação do user_key — Módulo 4.

Dois modos, mesma primitiva HMAC-SHA256:

  Autenticado → HMAC(pepper, client_id + "|" + user_token)
  Anónimo     → HMAC(pepper, client_id + "|" + "_anon")

O separador "|" previne ataques de concatenação entre campos.
O sentinel "_anon" é literal fixo — não colide com user_tokens reais
porque o schema rejeita user_token="_anon" por convenção.
Prefixo "uk_" identifica o tipo de chave em logs sem expor o valor.
"""
import hmac
import hashlib

_ANON_SENTINEL = "_anon"


def derive_user_key(pepper: str, client_id: str, user_token: str | None) -> str:
    """
    Retorna user_key determinístico para o par (client_id, user_token).
    user_token=None produz a chave anónima da aplicação.
    """
    token   = user_token if user_token is not None else _ANON_SENTINEL
    message = f"{client_id}|{token}".encode()
    digest  = hmac.new(
        key       = pepper.encode(),
        msg       = message,
        digestmod = hashlib.sha256,
    ).hexdigest()
    return f"uk_{digest}"


def is_anonymous_key(pepper: str, client_id: str, user_key: str) -> bool:
    """Verifica se um user_key corresponde ao modo anónimo de uma aplicação."""
    return user_key == derive_user_key(pepper, client_id, None)