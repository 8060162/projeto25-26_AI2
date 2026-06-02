from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # MongoDB
    mongodb_url:      str = "mongodb://localhost:27017"
    mongodb_database: str = "ragdb"

    # Redis
    redis_url:                 str = "redis://localhost:6379/0"
    auth_cache_ttl_seconds:    int = 300
    rate_limit_window_seconds: int = 60
    session_ttl_seconds:       int = 1800

    # Auth
    api_key_prefix:        str = "rag"
    api_key_entropy_bytes: int = 32
    default_rate_limit:    int = 100

    # Pipeline — lidas pelo appsettings.json via env vars
    # O rag-api vive dentro do PROJETO25-26_AI2 — não precisa de PIPELINE_ROOT
    openai_api_key:         str = ""
    chroma_api_key:         str = ""
    external_gpt4o_api_key: str = ""

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache
def get_settings() -> Settings:
    return Settings()