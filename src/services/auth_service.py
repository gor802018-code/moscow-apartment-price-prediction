"""Сервис аутентификации."""

from sqlalchemy.ext.asyncio import AsyncSession

from src.core.logging import get_logger
from src.core.security import (
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)
from src.repositories.user_repository import UserRepository

log = get_logger(__name__)

# Одинаковое сообщение для защиты от user enumeration
_AUTH_ERROR = "Invalid email or password"


class AuthService:
    """Бизнес-логика аутентификации."""

    def __init__(self, session: AsyncSession):
        self.session = session
        self.user_repo = UserRepository(session)

    async def register(self, email: str, password: str) -> dict:
        """Регистрация пользователя.

        Args:
            email: Email пользователя.
            password: Пароль в открытом виде.

        Returns:
            Словарь с JWT токеном.

        Raises:
            ValueError: Если пользователь уже существует.
        """
        existing = await self.user_repo.get_by_email(email)
        if existing:
            raise ValueError("User already exists")

        hashed = hash_password(password)
        user = await self.user_repo.create_user(email, hashed)

        token = create_access_token(str(user.id))
        log.info("User registered", user_id=user.id)

        return {"access_token": token, "token_type": "bearer", "user_id": user.id}

    async def login(self, email: str, password: str) -> dict:
        """Логин пользователя, возвращает JWT.

        Args:
            email: Email пользователя.
            password: Пароль.

        Returns:
            Словарь с JWT токеном.

        Raises:
            ValueError: Если учётные данные неверны.
        """
        user = await self.user_repo.get_by_email(email)
        if user is None or not verify_password(password, user.hashed_password):
            raise ValueError(_AUTH_ERROR)

        token = create_access_token(str(user.id))
        log.info("User logged in", user_id=user.id)

        return {"access_token": token, "token_type": "bearer", "user_id": user.id}

    async def verify_token(self, token: str) -> str | None:
        """Проверяет JWT токен и возвращает user_id.

        Args:
            token: JWT токен.

        Returns:
            User ID или None.
        """
        return decode_access_token(token)
