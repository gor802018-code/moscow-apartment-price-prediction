"""Конфигурация приложения через pydantic-settings."""

from datetime import UTC, datetime
from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SECRET_KEY = "dev-secret-key-change-in-production"


class Settings(BaseSettings):
    """Настройки приложения из окружения или .env файла."""

    model_config = SettingsConfigDict(
        env_file=(".env", ".env.local"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        # В .env могут лежать переменные для docker-compose (POSTGRES_USER и т.п.),
        # которых нет в Settings. Без extra="ignore" приложение упало бы при старте.
        extra="ignore",
    )

    # Приложение
    APP_NAME: str = "Kwork ML"
    ENVIRONMENT: str = "development"  # development | production
    DEBUG: bool = False

    # База данных
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/kwork"
    DATABASE_ECHO: bool = False

    # JWT
    SECRET_KEY: str = DEFAULT_SECRET_KEY
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # Парсинг
    SCRAPE_URL: str = "https://www.example.com"
    SCRAPE_DELAY: float = 1.0

    # ML (путь можно переопределить переменной окружения MODEL_PATH)
    MODEL_PATH: str = "artifacts/price_model.joblib"

    @model_validator(mode="after")
    def _forbid_default_secret_in_production(self) -> "Settings":
        if self.ENVIRONMENT.lower() == "production" and self.SECRET_KEY == DEFAULT_SECRET_KEY:
            raise ValueError("SECRET_KEY must be changed when ENVIRONMENT=production")
        return self

    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT.lower() == "production"

    @property
    def current_year(self) -> int:
        """Текущий год для расчёта возраста зданий."""
        return datetime.now(UTC).year

    @property
    def model_path(self) -> Path:
        """Путь к модели. Относительные пути считаются от корня проекта, а не от CWD."""
        path = Path(self.MODEL_PATH)
        return path if path.is_absolute() else PROJECT_ROOT / path


_settings: Settings | None = None


def get_settings() -> Settings:
    """Глобальный синглтон настроек."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
