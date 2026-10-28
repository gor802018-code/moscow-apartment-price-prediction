"""Модель предсказания."""

from datetime import datetime

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.models.base import Base


class Prediction(Base):
    """Таблица predictions."""

    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    # JSONB в PostgreSQL, обычный JSON в SQLite (тесты)
    input_features: Mapped[dict] = mapped_column(JSON().with_variant(JSONB(), "postgresql"), nullable=False)
    predicted_price: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # Связи
    user: Mapped["User"] = relationship("User", back_populates="predictions")  # noqa: F821
