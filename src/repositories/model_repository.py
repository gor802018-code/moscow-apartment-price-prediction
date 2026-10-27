"""Репозиторий для загрузки ML-модели."""

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

from src.core.config import get_settings


class ModelRepository:
    """Загрузка и сохранение ML-модели с TTL-кэшированием."""

    _model: Any = None
    _cache_timestamp: datetime | None = None
    _CACHE_TTL_SECONDS: int = 3600  # 1 час

    def __init__(self, model_path: str | Path | None = None):
        self.model_path = model_path or get_settings().model_path

    def _is_cache_valid(self) -> bool:
        """Проверяет, что кэш ещё действителен."""
        if ModelRepository._model is None:
            return False
        if ModelRepository._cache_timestamp is None:
            return False
        elapsed = (datetime.now(timezone.utc) - ModelRepository._cache_timestamp).total_seconds()
        return elapsed < ModelRepository._CACHE_TTL_SECONDS

    def load(self) -> Any:
        """Загружает модель из файла .joblib с TTL-кэшированием."""
        if self._is_cache_valid():
            return ModelRepository._model

        path = Path(self.model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        ModelRepository._model = joblib.load(path)
        ModelRepository._cache_timestamp = datetime.now(timezone.utc)
        return ModelRepository._model

    @property
    def model(self) -> Any:
        """Ленивая загрузка модели (с TTL-кэшированием)."""
        return self.load()

    async def predict(self, features: list) -> list:
        """Выполняет предсказание через модель (не блокирует event loop)."""
        loop = asyncio.get_event_loop()
        model = self.model
        return await loop.run_in_executor(None, lambda: model.predict(features))

    @classmethod
    def invalidate_cache(cls) -> None:
        """Принудительно инвалидирует кэш модели."""
        cls._model = None
        cls._cache_timestamp = None
