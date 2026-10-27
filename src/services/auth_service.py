"""Сервис аутентификации."""

from sqlalchemy.ext.asyncio import AsyncSession

from src.core.security import hash_password, verify_password, create_access_token, decode_access_token
from src.repositories.user_repository import UserRepository
from src.core.logging import get_logger

log = get_logger(__name__)


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
            raise ValueError(f"User {email} already exists")

        hashed = hash_password(password)
        user = await self.user_repo.create_user(email, hashed)

        token = create_access_token(email)
        log.info("User registered", email=email)

        return {"access_token": token, "token_type": "bearer", "user_id": user.id}

    async def login(self, email: str, password: str) -> dict:
        """Логин пользователя.

        Args:
            email: Email пользователя.
            password: Пароль.

        Returns:
            Словарь с JWT токеном.

        Raises:
            ValueError: Если пользователь не найден или пароль неверный.
        """
        user = await self.user_repo.get_by_email(email)
        if not user:
            raise ValueError("User not found")

        if not verify_password(password, user.hashed_password):
            raise ValueError("Invalid password")

        token = create_access_token(email)
        log.info("User logged in", email=email)

        return {"access_token": token, "token_type": "bearer", "user_id": user.id}

    async def verify_token(self, token: str) -> str | None:
        """Проверяет JWT токен и возвращает email.

        Args:
            token: JWT токен.

        Returns:
            Email пользователя или None.
        """
        return decode_access_token(token)
