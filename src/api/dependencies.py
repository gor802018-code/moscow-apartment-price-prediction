"""Зависимости FastAPI (DI): БД, текущий пользователь."""

from fastapi import Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.core.security import decode_access_token
from src.repositories.user_repository import UserRepository
from src.core.logging import get_logger

log = get_logger(__name__)


# Глобальные переменные для настройки через lifespan
_async_session_factory: async_sessionmaker | None = None


def set_session_factory(factory: async_sessionmaker):
    """Устанавливает фабрику сессий (вызывается при старте приложения)."""
    global _async_session_factory
    _async_session_factory = factory


async def get_db() -> AsyncSession:
    """Dependency: получает сессию БД."""
    if _async_session_factory is None:
        raise RuntimeError("Session factory not initialized")
    async with _async_session_factory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise


async def get_current_user(token: str = Depends(...)) -> dict:
    """Dependency: декодирует JWT и возвращает пользователя."""
    email = decode_access_token(token)
    if not email:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )
    return {"email": email}
