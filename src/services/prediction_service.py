"""Сервис предсказаний."""

from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from src.core.logging import get_logger
from src.repositories.prediction_repository import PredictionRepository
from src.repositories.model_repository import ModelRepository

log = get_logger(__name__)


class PredictionService:
    """Бизнес-логика предсказаний."""

    def __init__(self, session: AsyncSession, model_path: str | None = None):
        self.session = session
        self.prediction_repo = PredictionRepository(session)
        self.model_repo = ModelRepository(model_path)

    async def predict(
        self,
        user_id: int,
        input_features: dict[str, Any],
    ) -> dict[str, Any]:
        """Выполняет предсказание цены квартиры.

        Args:
            user_id: ID пользователя.
            input_features: Словарь с признаками квартиры.

        Returns:
            Словарь с результатом предсказания.
        """
        log.info("Predicting price", user_id=user_id, features=input_features)

        # Загружаем и выполняем предсказание
        model = self.model_repo.model

        # Формируем вектор признаков для модели
        feature_vector = self._prepare_features(input_features)

        prediction = model.predict([feature_vector])[0]

        # Сохраняем в БД
        saved = await self.prediction_repo.save_prediction(
            user_id=user_id,
            input_features=input_features,
            predicted_price=float(prediction),
        )

        log.info("Prediction saved", prediction_id=saved.id, price=prediction)

        return {
            "id": saved.id,
            "user_id": user_id,
            "input_features": input_features,
            "predicted_price": float(prediction),
            "created_at": str(saved.created_at),
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
                "created_at": str(p.created_at),
            }
            for p in predictions
        ]

    @staticmethod
    def _prepare_features(features: dict) -> list:
        """Формирует вектор признаков для модели."""
        feature_order = [
            "rooms", "area", "floor", "total_floors",
            "year_built", "distance_to_metro_min",
            "area_per_room", "floor_ratio", "building_age",
        ]
        return [features.get(col, 0) for col in feature_order]
