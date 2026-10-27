"""Утилиты безопасности: хэширование паролей и JWT."""

from datetime import UTC, datetime, timedelta

from jose import JWTError, jwt
from passlib.context import CryptContext

from src.core.config import get_settings

# Используем pbkdf2_sha256 вместо bcrypt из-за несовместимости passlib с bcrypt 4.x
pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")


def hash_password(password: str) -> str:
    """Хэширует пароль с помощью pbkdf2_sha256."""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Проверяет пароль против хэша."""
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(subject: str, expires_delta: int | None = None) -> str:
    """Создаёт JWT токен.

    Args:
        subject: Subject токена (user_id).
        expires_delta: Время жизни в минутах (по умолчанию из настроек).

    Returns:
        Закодированный JWT токен.
    """
    settings = get_settings()
    expire_minutes = expires_delta or settings.ACCESS_TOKEN_EXPIRE_MINUTES
    expires = datetime.now(UTC) + timedelta(minutes=expire_minutes)

    payload = {
        "sub": subject,
        "exp": expires,
        "iat": datetime.now(UTC),
    }
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def decode_access_token(token: str) -> str | None:
    """Декодирует JWT токен.

    Args:
        token: JWT токен для декодирования.

    Returns:
        Subject токена (user_id) или None при ошибке.
    """
    settings = get_settings()
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        return payload.get("sub")
    except JWTError:
        return None
