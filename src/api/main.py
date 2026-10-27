"""FastAPI приложение — точка входа."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from sqlalchemy.ext.asyncio import async_sessionmaker

from src.api.routers import auth, prediction, health
from src.core.config import get_settings
from src.core.logging import setup_logging, get_logger

log = get_logger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Инициализация при старте."""
    setup_logging()
    log.info("Application starting up")
    yield
    log.info("Application shutting down")


app = FastAPI(
    title=settings.APP_NAME,
    version="0.1.0",
    lifespan=lifespan,
)

# Роутеры
app.include_router(health.router, prefix="/api")
app.include_router(auth.router, prefix="/api")
app.include_router(prediction.router, prefix="/api")


def init_app(session_factory: async_sessionmaker):
    """Инициализация зависимостей (вызывается перед запуском)."""
    from src.api.dependencies import set_session_factory
    set_session_factory(session_factory)
