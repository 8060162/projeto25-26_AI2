"""
Ponto de entrada da aplicação.
Responsabilidade: wiring — ligar todas as peças sem conter lógica.

O rag-api vive dentro de PROJETO25-26_AI2/rag-api/.
O pipeline (retrieval/, embedding/, Chunking/) vive em PROJETO25-26_AI2/.
O sys.path é configurado automaticamente no módulo de arranque —
não requer PYTHONPATH externo nem variáveis de ambiente adicionais.

Ordem dos middlewares (exterior → interior):
  AccessLogMiddleware → auth (via Depends) → rate_limit (via Depends) → handler
"""
import os
import sys
import pathlib
import structlog

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from motor.motor_asyncio import AsyncIOMotorClient
from starlette.exceptions import HTTPException

from rag_api.api.error_handlers import (
    http_exception_handler,
    unhandled_exception_handler,
    validation_exception_handler,
)
from rag_api.api.middleware.logging import AccessLogMiddleware
from rag_api.api.middleware.rate_limit import RateLimiter
from rag_api.api.routes.applications import router as applications_router
from rag_api.api.routes.metrics import router as metrics_router
from rag_api.api.routes.pipeline import router as pipeline_router
from rag_api.api.routes.query import router as query_router
from rag_api.applications.store import MongoApplicationRepository, init_application_indexes
from rag_api.auth.cache import RedisAuthCache
from rag_api.auth.store import MongoKeyRepository, init_indexes
from rag_api.auth.strategies import APIKeyAuthStrategy
from rag_api.config.settings import get_settings
from rag_api.rag import create_rag_controller
from rag_api.reporting import (
    MongoSnapshotAggregator, RedisSchedulerLock,
    create_scheduler, SchedulerConfig,
)
from rag_api.schemas.responses import HealthResponse
from rag_api.signals.init_collection import init_signals_collection
from rag_api.signals.store import MongoSignalStore
from rag_api.signals.reader import MongoMetricsReader


def _register_pipeline_path() -> None:
    """
    Regista o path da raiz do projecto (PROJETO25-26_AI2/) no sys.path.

    O rag-api está em:   PROJETO25-26_AI2/rag-api/rag_api/api/main.py
    A raiz do projecto:  PROJETO25-26_AI2/
    Calculado como:      __file__ subindo 4 níveis (main.py → api → rag_api → rag-api → raiz)
    """
    this_file    = pathlib.Path(__file__).resolve()
    project_root = this_file.parents[3]

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
        structlog.get_logger(__name__).info(
            "pipeline_path_registered",
            path=str(project_root),
        )


_register_pipeline_path()


def create_app() -> FastAPI:
    settings = get_settings()

    if settings.openai_api_key:
        os.environ.setdefault("OPENAI_API_KEY", settings.openai_api_key)
    if settings.chroma_api_key:
        os.environ.setdefault("CHROMA_API_KEY", settings.chroma_api_key)
    if settings.external_gpt4o_api_key:
        os.environ.setdefault("EXTERNAL_GPT4O_API_KEY", settings.external_gpt4o_api_key)

    app = FastAPI(
        title     = "RAG API",
        version   = "1.0.0",
        docs_url  = "/docs",
        redoc_url = None,
    )

    # ── Middlewares ───────────────────────────────────────────────────────────
    app.add_middleware(AccessLogMiddleware)

    # ── Error handlers ────────────────────────────────────────────────────────
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)

    # ── Routes ────────────────────────────────────────────────────────────────
    app.include_router(query_router)
    app.include_router(applications_router)
    app.include_router(pipeline_router)
    app.include_router(metrics_router)

    @app.get("/v1/health", response_model=HealthResponse, tags=["infra"])
    async def health():
        """Liveness check — sem autenticação."""
        return HealthResponse(status="ok", version="1.0.0")

    # ── Lifecycle ─────────────────────────────────────────────────────────────
    @app.on_event("startup")
    async def startup():
        from redis.asyncio import from_url as redis_from_url

        # MongoDB
        mongo_client = AsyncIOMotorClient(settings.mongodb_url)
        db           = mongo_client[settings.mongodb_database]
        await init_indexes(db)
        await init_application_indexes(db)
        await init_signals_collection(db)

        # Redis
        redis = await redis_from_url(
            settings.redis_url, decode_responses=True
        )

        # Wiring — auth
        key_repo = MongoKeyRepository(db)
        app_repo = MongoApplicationRepository(db)
        cache    = RedisAuthCache(redis)
        strategy = APIKeyAuthStrategy(key_repo, cache)

        app.state.mongo_client      = mongo_client
        app.state.auth_strategy     = strategy
        app.state.key_repo          = key_repo
        app.state.application_repo  = app_repo
        app.state.rate_limiter      = RateLimiter(redis)

        # Wiring — signals
        app.state.signal_store   = MongoSignalStore(db)
        app.state.metrics_reader = MongoMetricsReader(db)

        # Wiring — reporting
        aggregator = MongoSnapshotAggregator(db)
        lock       = RedisSchedulerLock(redis, settings.snapshot_lock_ttl_seconds)
        config     = SchedulerConfig(
            realtime_minutes = settings.snapshot_realtime_minutes,
            hourly_minutes   = settings.snapshot_hourly_minutes,
            daily_hour       = settings.snapshot_daily_hour,
            daily_minute     = settings.snapshot_daily_minute,
            monthly_day      = settings.snapshot_monthly_day,
            monthly_hour     = settings.snapshot_monthly_hour,
            purge_hour       = settings.snapshot_purge_hour,
            timezone         = settings.snapshot_timezone,
        )
        scheduler = create_scheduler(aggregator, lock, config)
        scheduler.start()

        app.state.snapshot_aggregator = aggregator
        app.state.scheduler           = scheduler

        # RAG Controller — real se o pipeline estiver disponível
        app.state.rag_controller = create_rag_controller()

        structlog.get_logger(__name__).info(
            "startup_complete",
            version     = "1.0.0",
            rag_backend = type(app.state.rag_controller).__name__,
        )

    @app.on_event("shutdown")
    async def shutdown():
        if hasattr(app.state, "scheduler"):
            app.state.scheduler.shutdown(wait=False)
        if hasattr(app.state, "mongo_client"):
            app.state.mongo_client.close()

    return app


app = create_app()