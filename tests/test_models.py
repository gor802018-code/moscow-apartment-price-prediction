"""Тесты для моделей и БД (SQLite in-memory)."""

from sqlalchemy import create_engine
from sqlalchemy import text as sa_text
from sqlalchemy.orm import Session

from src.models.base import Base
from src.models.prediction import Prediction
from src.models.user import User


def _get_test_db():
    """Создает тестовую БД в памяти."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    return Session(engine)


def test_create_tables():
    """Таблицы создаются без ошибок."""
    session = _get_test_db()
    try:
        # Проверяем что таблицы существуют
        result = session.execute(sa_text("SELECT name FROM sqlite_master WHERE type='table'"))
        tables = {row[0] for row in result}
        assert "users" in tables
        assert "predictions" in tables
    finally:
        session.close()


def test_create_user():
    """Пользователь создается корректно."""
    session = _get_test_db()
    try:
        user = User(email="test@example.com", hashed_password="hashed123")
        session.add(user)
        session.commit()

        fetched = session.query(User).filter_by(email="test@example.com").first()
        assert fetched is not None
        assert fetched.hashed_password == "hashed123"
    finally:
        session.close()


def test_create_prediction():
    """Предсказание создается и связывается с пользователем."""
    session = _get_test_db()
    try:
        user = User(email="user@example.com", hashed_password="pass")
        session.add(user)
        session.commit()

        prediction = Prediction(
            user_id=user.id,
            input_features={"rooms": 2, "area": 50.0},
            predicted_price=8500000.0,
        )
        session.add(prediction)
        session.commit()

        fetched = session.query(Prediction).first()
        assert fetched is not None
        assert fetched.predicted_price == 8500000.0
        assert fetched.user_id == user.id
    finally:
        session.close()


def test_cascade_delete():
    """При удалении пользователя удаляются его предсказания."""
    session = _get_test_db()
    try:
        user = User(email="delete@example.com", hashed_password="pass")
        session.add(user)
        session.commit()

        prediction = Prediction(
            user_id=user.id,
            input_features={"rooms": 1},
            predicted_price=5000000.0,
        )
        session.add(prediction)
        session.commit()

        session.delete(user)
        session.commit()

        assert session.query(Prediction).count() == 0
    finally:
        session.close()
