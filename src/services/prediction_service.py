"""Сервис предсказаний."""

from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from src.core.logging import get_logger
from src.repositories.model_repository import ModelRepository
from src.repositories.prediction_repository import PredictionRepository

log = get_logger(__name__)

# Допустимые диапазоны для входных данных
_FEATURE_RANGES = {
    "rooms": (1, 10),
    "area": (5.0, 500.0),
    "floor": (1, 300),
    "total_floors": (1, 300),
    "year_built": (1800, 2100),
    "distance_to_metro_min": (0.0, 120.0),
}


class ValidationError(Exception):
    """Ошибка валидации входных данных."""

    def __init__(self, field: str, message: str):
        self.field = field
        self.message = message
        super().__init__(f"{field}: {message}")


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
            input_features: Словарь с признаками квартиры.

        Returns:
            Словарь с результатом предсказания.

        Raises:
            ValidationError: Если входные данные невалидны.
        """
        log.info("Predicting price", user_id=user_id, features=input_features)

        # Валидация входных данных
        self._validate_features(input_features)

        # Формируем вектор признаков (с вычислением производных)
        feature_vector = self._prepare_features(input_features)

        # Предсказание
        model = self.model_repo.model
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
    def _validate_features(features: dict) -> None:
        """Валидирует входные признаки (только если поле присутствует)."""
        for field, (min_val, max_val) in _FEATURE_RANGES.items():
            value = features.get(field)
            if value is not None and (value < min_val or value > max_val):
                raise ValidationError(
                    field,
                    f"value {value} is out of range [{min_val}, {max_val}]",
                )

        # Дополнительная логика: этаж не может быть больше общего количества этажей
        floor = features.get("floor")
        total_floors = features.get("total_floors")
        if floor is not None and total_floors is not None and floor > total_floors:
            raise ValidationError(
                "floor",
                f"floor {floor} cannot exceed total_floors {total_floors}",
            )

    @staticmethod
    def _prepare_features(features: dict) -> list:
        """Формирует вектор признаков для модели.

        Вычисляет производные признаки, если они не переданы.
        """
        rooms = features.get("rooms", 0)
        area = features.get("area", 0)
        floor = features.get("floor", 0)
        total_floors = features.get("total_floors", 0)
        year_built = features.get("year_built", 0)
        distance_to_metro = features.get("distance_to_metro_min", 0)

        # Вычисляем производные признаки
        area_per_room = area / rooms if rooms > 0 else 0
        floor_ratio = (floor - 1) / (total_floors - 1) if total_floors > 1 else 0

        from datetime import datetime, timezone
        current_year = datetime.now(timezone.utc).year
        building_age = current_year - year_built if year_built else 0

        return [
            rooms,
            area,
            floor,
            total_floors,
            year_built,
            distance_to_metro,
            area_per_room,
            floor_ratio,
            building_age,
        ]
