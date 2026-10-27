"""FastAPI приложение — точка входа."""

import structlog
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.requests import Request
from slowapi import Limiter
from slowapi.util import get_remote_address
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine, async_sessionmaker

from src.api.dependencies import set_session_factory
from src.api.routers import auth, health, prediction
from src.core.config import get_settings
from src.core.logging import get_logger, setup_logging, generate_request_id, clear_request_context
from src.core.rate_limiter import limiter

log = get_logger(__name__)
settings = get_settings()

# Глобальные объекты для lifespan
_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker | None = None

# Rate limiting
limiter = Limiter(key_func=get_remote_address)


async def request_id_middleware(request: Request, call_next):
    """Middleware: добавляет request_id в контекст логирования."""
    request_id = request.headers.get("X-Request-ID", generate_request_id())
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(request_id=request_id)

    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Инициализация при старте."""
    global _engine, _session_factory

    setup_logging()
    log.info("Application starting up")

    # Создаём движок БД
    _engine = create_async_engine(
        settings.DATABASE_URL,
        echo=settings.DATABASE_ECHO,
    )
    _session_factory = async_sessionmaker(_engine, expire_on_commit=False)
    set_session_factory(_session_factory)

    log.info("Database connected", url=settings.DATABASE_URL)

    yield

    log.info("Application shutting down")
    await _engine.dispose()


app = FastAPI(
    title=settings.APP_NAME,
    version="0.1.0",
    lifespan=lifespan,
)

# Middleware для request_id
app.middleware("http")(request_id_middleware)

# Rate limiting
app.state.limiter = limiter

# Роутеры v1
app.include_router(health.router, prefix="/api/v1")
app.include_router(auth.router, prefix="/api/v1")
app.include_router(prediction.router, prefix="/api/v1")
