import hashlib
import secrets
from datetime import datetime

from rag_api.config.settings import get_settings
from rag_api.schemas.identity import CreateKeyResponse


def generate_api_key() -> tuple[str, str, str]:
    """
    Gera uma API key criptograficamente segura.

    Retorna: (raw_key, key_hash, key_hint)
    - raw_key:  mostrar ao cliente uma única vez, nunca persistir
    - key_hash: persistir na DB — SHA-256 da raw_key
    - key_hint: primeiros 12 chars para identificação visual futura
    """
    settings = get_settings()
    token = secrets.token_urlsafe(settings.api_key_entropy_bytes)
    raw_key = f"{settings.api_key_prefix}_{token}"

    key_hash = _hash_key(raw_key)
    key_hint = raw_key[:12]

    return raw_key, key_hash, key_hint


def hash_key(raw_key: str) -> str:
    """
    Interface pública para hashing — usada na validação de pedidos.
    SHA-256 é suficiente porque a key já tem 256 bits de entropia.
    (bcrypt seria over-engineering aqui — serve para passwords humanas)
    """
    return _hash_key(raw_key)


def _hash_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode()).hexdigest()


def build_create_key_response(
    raw_key: str,
    hint: str,
    client_id: str,
    scopes: list[str],
    expires_at: datetime | None,
) -> CreateKeyResponse:
    """Constrói a resposta de criação — única vez que raw_key é exposta."""
    return CreateKeyResponse(
        key=raw_key,
        hint=hint,
        client_id=client_id,
        scopes=scopes,
        expires_at=expires_at,
    )