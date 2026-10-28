"""Сервис предсказаний."""

import asyncio
from datetime import datetime
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from src.core.logging import get_logger
from src.data.preprocessing import build_feature_vector
from src.repositories.model_repository import ModelRepository
from src.repositories.prediction_repository import PredictionRepository

log = get_logger(__name__)


class ValidationError(Exception):
    """Ошибка валидации входных данных."""

    def __init__(self, field: str, message: str):
        self.field = field
        self.message = message
        super().__init__(f"{field}: {message}")


def _iso(value: Any) -> str:
    """created_at в ISO-формате (для datetime) — стабильнее, чем str()."""
    return value.isoformat() if isinstance(value, datetime) else str(value)


class PredictionService:
    """Бизнес-логика предсказаний."""

    def __init__(self, session: AsyncSession, model_repo: ModelRepository | None = None):
        self.session = session
        self.prediction_repo = PredictionRepository(session)
        self.model_repo = model_repo or ModelRepository()

    async def predict(
        self,
        user_id: int,
        input_features: dict[str, Any],
    ) -> dict[str, Any]:
        """Выполняет предсказание цены квартиры.

        Args:
            user_id: ID пользователя.
            input_features: Словарь с базовыми признаками квартиры.

        Returns:
            Словарь с результатом предсказания.

        Raises:
            ValidationError: Если входные данные невалидны.
            FileNotFoundError: Если файл модели отсутствует.
        """
        log.info("Predicting price", user_id=user_id)

        self._validate_features(input_features)
        feature_vector = self._prepare_features(input_features)

        # Загрузка и predict модели синхронные — выносим из event loop
        model = await asyncio.to_thread(lambda: self.model_repo.model)
        raw_prediction = await asyncio.to_thread(model.predict, [feature_vector])
        prediction = float(raw_prediction[0])

        saved = await self.prediction_repo.save_prediction(
            user_id=user_id,
            input_features=input_features,
            predicted_price=prediction,
        )

        log.info("Prediction saved", prediction_id=saved.id, price=prediction)

        return {
            "id": saved.id,
            "user_id": user_id,
            "input_features": input_features,
            "predicted_price": prediction,
            "created_at": _iso(saved.created_at),
        }

    async def get_history(self, user_id: int, limit: int = 50) -> list[dict]:
        """Возвращает историю предсказаний пользователя.

        Args:
            user_id: ID пользователя.
            limit: Максимальное количество записей.

        Returns:
            Список предсказаний.
        """
        predictions = await self.prediction_repo.get_history(user_id, limit)
        return [
            {
                "id": p.id,
                "input_features": p.input_features,
                "predicted_price": p.predicted_price,
                "created_at": _iso(p.created_at),
            }
            for p in predictions
        ]

    @staticmethod
    def _validate_features(features: dict) -> None:
        """Проверки, не покрытые схемой: этаж не может быть выше этажности дома.

        Диапазоны отдельных полей проверяет Pydantic-схема PredictionRequest (единый источник
        правды: раньше здесь был второй, расходящийся набор границ).
        """
        floor = features.get("floor")
        total_floors = features.get("total_floors")
        if floor is not None and total_floors is not None and floor > total_floors:
            raise ValidationError(
                "floor",
                f"floor {floor} cannot exceed total_floors {total_floors}",
            )

    @staticmethod
    def _prepare_features(features: dict) -> list[float]:
        """Формирует вектор признаков для модели через ту же логику, что и при обучении."""
        try:
            return build_feature_vector(features)
        except ValueError as exc:
            raise ValidationError("features", str(exc)) from exc
