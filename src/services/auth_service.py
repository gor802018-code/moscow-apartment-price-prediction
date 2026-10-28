"""Сервис аутентификации."""

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.logging import get_logger
from src.core.security import (
    create_access_token,
    decode_access_token,
    dummy_verify,
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
        email = self._normalize_email(email)
        existing = await self.user_repo.get_by_email(email)
        if existing:
            raise ValueError("User already exists")

        hashed = hash_password(password)
        try:
            user = await self.user_repo.create_user(email, hashed)
        except IntegrityError as exc:
            # Два параллельных запроса с одним email: проверка выше прошла у обоих,
            # а уникальный индекс в БД сработал только у второго.
            await self.session.rollback()
            raise ValueError("User already exists") from exc

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
        user = await self.user_repo.get_by_email(self._normalize_email(email))
        if user is None:
            # Тратим то же время, что и на проверку реального пароля, чтобы по времени ответа
            # нельзя было отличить «нет такого email» от «неверный пароль».
            dummy_verify()
            raise ValueError(_AUTH_ERROR)
        if not verify_password(password, user.hashed_password):
            raise ValueError(_AUTH_ERROR)

        token = create_access_token(str(user.id))
        log.info("User logged in", user_id=user.id)

        return {"access_token": token, "token_type": "bearer", "user_id": user.id}

    @staticmethod
    def _normalize_email(email: str) -> str:
        """Email регистронезависим: 'A@x.com' и 'a@x.com' — один пользователь."""
        return email.strip().lower()

    async def verify_token(self, token: str) -> str | None:
        """Проверяет JWT токен и возвращает user_id.

        Args:
            token: JWT токен.

        Returns:
            User ID или None.
        """
        return decode_access_token(token)
