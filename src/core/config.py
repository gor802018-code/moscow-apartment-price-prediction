"""Конфигурация приложения через pydantic-settings."""

from datetime import datetime, timezone
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Настройки приложения из окружения или .env файла."""

    model_config = SettingsConfigDict(
        env_file=(".env", ".env.local"),
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Приложение
    APP_NAME: str = "Kwork ML"
    DEBUG: bool = False

    # База данных
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/kwork"
    DATABASE_ECHO: bool = False

    # JWT
    SECRET_KEY: str = "dev-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # Парсинг
    SCRAPE_URL: str = "https://www.example.com"
    SCRAPE_DELAY: float = 1.0

    # ML
    MODEL_PATH: str = "src/models/price_model.joblib"

    @property
    def current_year(self) -> int:
        """Текущий год для расчёта возраста зданий."""
        return datetime.now(timezone.utc).year

    @property
    def model_path(self) -> Path:
        """Путь к модели в виде Path."""
        return Path(self.MODEL_PATH)


_settings: Settings | None = None


def get_settings() -> Settings:
    """Глобальный синглтон настроек."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
