"""Тесты для core: config, security, logging."""

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

    def test_hash_password_length(self):
        """Хэш имеет корректную длину для bcrypt."""
        long_pass = "A" * 100
        hashed = hash_password(long_pass)
        assert len(hashed) > 0
        assert verify_password(long_pass, hashed)

    def test_create_and_decode_token(self):
        """JWT токен создаётся и декодируется."""
        token = create_access_token("test@example.com")
        subject = decode_access_token(token)
        assert subject == "test@example.com"

    def test_decode_invalid_token(self):
        """Невалидный токен возвращает None."""
        result = decode_access_token("invalid.token.here")
        assert result is None

    def test_expired_token_is_rejected(self):
        """Токен с отрицательным TTL уже просрочен (раньше TTL=0 молча менялся на дефолтный)."""
        token = create_access_token("test@example.com", expires_delta=-1)
        assert decode_access_token(token) is None

    def test_custom_ttl_is_valid(self):
        token = create_access_token("test@example.com", expires_delta=5)
        assert decode_access_token(token) == "test@example.com"


class TestSettings:
    """Тесты настроек.

    `_env_file=None` — тесты не должны зависеть от локального .env разработчика: иначе его
    SECRET_KEY подменяет дефолтный и проверку «production запрещает дефолтный ключ» не проверить.
    """

    def test_settings_default_values(self):
        """Значения по умолчанию загружаются."""
        with patch.dict("os.environ", {}, clear=True):
            settings = Settings(_env_file=None)
            assert settings.APP_NAME == "Kwork ML"
            assert settings.ALGORITHM == "HS256"
            assert settings.ACCESS_TOKEN_EXPIRE_MINUTES == 30

    def test_settings_from_env(self):
        """Настройки переопределяются через окружение."""
        with patch.dict("os.environ", {"APP_NAME": "TestApp", "DEBUG": "true"}):
            settings = Settings(_env_file=None)
            assert settings.APP_NAME == "TestApp"
            assert settings.DEBUG is True

    def test_default_secret_forbidden_in_production(self):
        import pytest

        with patch.dict("os.environ", {"ENVIRONMENT": "production"}, clear=True):
            with pytest.raises(ValueError):
                Settings(_env_file=None)

    def test_custom_secret_allowed_in_production(self):
        env = {"ENVIRONMENT": "production", "SECRET_KEY": "x" * 32}
        with patch.dict("os.environ", env, clear=True):
            assert Settings(_env_file=None).is_production

    def test_model_path_is_independent_of_cwd(self):
        """Относительный MODEL_PATH считается от корня проекта, а не от текущего каталога."""
        with patch.dict("os.environ", {}, clear=True):
            assert Settings(_env_file=None).model_path.is_absolute()
