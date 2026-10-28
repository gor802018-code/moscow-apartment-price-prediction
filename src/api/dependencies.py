"""Зависимости FastAPI (DI): БД, текущий пользователь, токен."""

from collections.abc import AsyncIterator

from fastapi import Depends, Header, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.core.logging import get_logger
from src.core.security import decode_access_token
from src.models.user import User

log = get_logger(__name__)


# Глобальные переменные для настройки через lifespan
_async_session_factory: async_sessionmaker | None = None


def set_session_factory(factory: async_sessionmaker):
    """Устанавливает фабрику сессий (вызывается при старте приложения)."""
    global _async_session_factory
    _async_session_factory = factory


async def get_db() -> AsyncIterator[AsyncSession]:
    """Dependency: получает сессию БД (генератор)."""
    if _async_session_factory is None:
        raise RuntimeError("Session factory not initialized")
    async with _async_session_factory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise


async def get_token(authorization: str | None = Header(default=None)) -> str:
    """Dependency: извлекает JWT токен из Authorization заголовка."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )
    return authorization.split(" ", 1)[1]


async def get_current_user(
    token: str = Depends(get_token),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Dependency: декодирует JWT, находит пользователя по id и возвращает его данные."""
    subject = decode_access_token(token)
    try:
        user_id = int(subject) if subject else None
    except ValueError:
        # Токен со «старым» subject (например, email) — это невалидный токен, а не ошибка сервера
        user_id = None
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )

    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )

    return {"user_id": user.id, "email": user.email}
