"""FastAPI приложение — точка входа."""

import re
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine

from src.api.dependencies import set_session_factory
from src.api.routers import auth, health, prediction
from src.core.config import get_settings
from src.core.logging import clear_request_context, generate_request_id, get_logger, setup_logging
from src.core.rate_limiter import limiter

log = get_logger(__name__)
settings = get_settings()

# Глобальные объекты для lifespan
_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker | None = None

# X-Request-ID приходит от клиента — принимаем только безопасный формат (защита от log injection)
_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


async def request_id_middleware(request: Request, call_next):
    """Middleware: добавляет request_id в контекст логирования."""
    incoming = request.headers.get("X-Request-ID", "")
    request_id = incoming if _REQUEST_ID_RE.fullmatch(incoming) else generate_request_id()
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(request_id=request_id)

    try:
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response
    finally:
        clear_request_context()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Инициализация при старте."""
    global _engine, _session_factory

    setup_logging(structured=settings.is_production)
    log.info("Application starting up")

    # Создаём движок БД
    _engine = create_async_engine(
        settings.DATABASE_URL,
        echo=settings.DATABASE_ECHO,
        pool_pre_ping=True,
    )
    _session_factory = async_sessionmaker(_engine, expire_on_commit=False)
    set_session_factory(_session_factory)

    # Пароль в логи не попадает
    log.info("Database engine created", url=_engine.url.render_as_string(hide_password=True))

    yield

    log.info("Application shutting down")
    if _engine is not None:
        await _engine.dispose()


app = FastAPI(
    title=settings.APP_NAME,
    version="0.1.0",
    lifespan=lifespan,
)

# Middleware для request_id
app.middleware("http")(request_id_middleware)

# Rate limiting: тот же limiter, что используют декораторы в роутерах
# (раньше здесь создавался второй Limiter, а обработчика RateLimitExceeded не было,
# поэтому превышение лимита превращалось в 500 вместо 429)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Роутеры v1
app.include_router(health.router, prefix="/api/v1")
app.include_router(auth.router, prefix="/api/v1")
app.include_router(prediction.router, prefix="/api/v1")
