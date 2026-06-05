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

    # Memory (Módulo 4)
    memory_ttl_days:    int = 90
    memory_hmac_pepper: str = ""   # obrigatório em produção — guard na startup

    # Pipeline — lidas pelo appsettings.json via env vars
    openai_api_key:         str = ""
    chroma_api_key:         str = ""
    external_gpt4o_api_key: str = ""

    # Reporting — Scheduler (signals)
    snapshot_realtime_minutes: int = 15
    snapshot_hourly_minutes:   int = 60
    snapshot_daily_hour:       int = 0
    snapshot_daily_minute:     int = 0
    snapshot_monthly_day:      int = 1
    snapshot_monthly_hour:     int = 0
    snapshot_purge_hour:       int = 2
    snapshot_lock_ttl_seconds: int = 60
    snapshot_timezone:         str = "Europe/Lisbon"

    # Memory Reporting — Scheduler
    memory_report_daily_hour:   int = 1      # 01:00 — após o snapshot de signals
    memory_report_daily_minute: int = 0
    memory_report_purge_hour:   int = 3      # 03:00 — limpeza de snapshots antigos
    memory_report_purge_days:   int = 548    # ~18 meses — consistente com reporting

    class Config:
        env_file       = ".env"
        case_sensitive = False


@lru_cache
def get_settings() -> Settings:
    return Settings()