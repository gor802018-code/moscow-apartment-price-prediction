"""Модель предсказания."""

from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, Integer, JSON, ForeignKey, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.models.base import Base


class Prediction(Base):
    """Таблица predictions."""

    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    input_features: Mapped[dict] = mapped_column(JSON, nullable=False)
    predicted_price: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # Связи
    user: Mapped["User"] = relationship("User", back_populates="predictions")
