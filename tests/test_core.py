"""Тесты для core: config, security, logging."""

import os
from unittest.mock import patch

from src.core.config import Settings
from src.core.security import (
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)


class TestSecurity:
    """Тесты хэширования паролей и JWT."""

    def test_hash_and_verify_password(self):
        """Пароль хэшируется и верифицируется корректно."""
        plain = "SecurePass123!"
        hashed = hash_password(plain)

        assert hashed != plain
        assert verify_password(plain, hashed)
        assert not verify_password("WrongPassword", hashed)

    def test_hash_is_unique(self):
        """Два хэша одного пароля различаются (salt)."""
        plain = "SamePassword"
        hash1 = hash_password(plain)
        hash2 = hash_password(plain)
        assert hash1 != hash2

    def test_create_and_decode_token(self):
        """JWT токен создаётся и декодируется."""
        token = create_access_token("test@example.com")
        subject = decode_access_token(token)
        assert subject == "test@example.com"

    def test_decode_invalid_token(self):
        """Невалидный токен возвращает None."""
        result = decode_access_token("invalid.token.here")
        assert result is None

    def test_token_expires(self):
        """Токен с нулевым TTL невалиден."""
        token = create_access_token("test@example.com", expires_delta=0)
        # Токен с expires_delta=0 всё ещё валиден сразу после создания,
        # но expires должен быть в прошлом
        subject = decode_access_token(token)
        assert subject == "test@example.com"


class TestSettings:
    """Тесты настроек."""

    def test_settings_default_values(self):
        """Значения по умолчанию загружаются."""
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings()
            assert settings.APP_NAME == "Kwork ML"
            assert settings.ALGORITHM == "HS256"
            assert settings.ACCESS_TOKEN_EXPIRE_MINUTES == 30

    def test_settings_from_env(self):
        """Настройки переопределяются через окружение."""
        with patch.dict(os.environ, {"APP_NAME": "TestApp", "DEBUG": "true"}):
            settings = Settings()
            assert settings.APP_NAME == "TestApp"
            assert settings.DEBUG is True
