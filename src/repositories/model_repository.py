"""Репозиторий для загрузки ML-модели."""

from pathlib import Path

import joblib

from src.core.config import get_settings


class ModelRepository:
    """Загрузка и сохранение ML-модели."""

    def __init__(self, model_path: str | None = None):
        self.model_path = model_path or get_settings().model_path
        self._model = None

    def load(self):
        """Загружает модель из файла .joblib."""
        path = Path(self.model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        self._model = joblib.load(path)
        return self._model

    @property
    def model(self):
        """Ленивая загрузка модели."""
        if self._model is None:
            self.load()
        return self._model

    async def predict(self, features: list) -> list:
        """Выполняет предсказание через модель."""
        model = self.model
        return model.predict(features)
