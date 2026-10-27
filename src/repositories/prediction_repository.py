"""Репозиторий для работы с предсказаниями."""

from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.prediction import Prediction


class PredictionRepository:
    """Доступ к данным предсказаний."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def save_prediction(
        self,
        user_id: int,
        input_features: dict,
        predicted_price: float,
    ) -> Prediction:
        """Сохраняет новое предсказание."""
        prediction = Prediction(
            user_id=user_id,
            input_features=input_features,
            predicted_price=predicted_price,
        )
        self.session.add(prediction)
        await self.session.commit()
        await self.session.refresh(prediction)
        return prediction

    async def get_history(self, user_id: int, limit: int = 50) -> list[Prediction]:
        """Возвращает историю предсказаний пользователя (новые первые)."""
        result = await self.session.execute(
            select(Prediction)
            .where(Prediction.user_id == user_id)
            .order_by(Prediction.id.desc())
            .limit(limit)
        )
        return list(result.scalars().all())
