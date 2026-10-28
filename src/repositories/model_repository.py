"""Репозиторий для загрузки ML-модели."""

import asyncio
from pathlib import Path
from typing import Any, ClassVar

import joblib

from src.core.config import get_settings


class ModelRepository:
    """Загрузка ML-модели с кэшем по (путь файла, время изменения).

    Кэш общий на процесс, поэтому модель не читается с диска на каждый запрос.
    Ключ включает путь и mtime: разные пути не подменяют друг друга, а после переобучения
    (новый файл модели) новая версия подхватывается сразу, без рестарта и без TTL.
    """

    _cache: ClassVar[dict[str, tuple[int, Any]]] = {}

    def __init__(self, model_path: str | Path | None = None):
        self.model_path = model_path or get_settings().model_path

    def load(self) -> Any:
        """Загружает модель из файла .joblib (с кэшированием)."""
        path = Path(self.model_path)
        try:
            mtime = path.stat().st_mtime_ns
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Model file not found: {self.model_path}") from exc

        key = str(path.resolve())
        cached = ModelRepository._cache.get(key)
        if cached is not None and cached[0] == mtime:
            return cached[1]

        model = joblib.load(path)
        ModelRepository._cache[key] = (mtime, model)
        return model

    @property
    def model(self) -> Any:
        """Ленивая загрузка модели."""
        return self.load()

    async def predict(self, features: list) -> list:
        """Выполняет предсказание через модель, не блокируя event loop."""
        model = await asyncio.to_thread(self.load)
        return await asyncio.to_thread(model.predict, features)

    @classmethod
    def invalidate_cache(cls) -> None:
        """Принудительно сбрасывает кэш моделей."""
        cls._cache.clear()
