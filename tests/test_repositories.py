"""Тесты для repository layer (SQLite in-memory + mocks)."""

from unittest.mock import AsyncMock, MagicMock, patch

import joblib
import numpy as np
import pytest

from src.models.base import Base
from src.models.user import User
from src.models.prediction import Prediction
from src.repositories.user_repository import UserRepository
from src.repositories.prediction_repository import PredictionRepository
from src.repositories.model_repository import ModelRepository


@pytest.fixture
def async_session():
    """Создает тестовую сессию SQLAlchemy."""
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    from sqlalchemy.orm import sessionmaker

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")

    async def setup():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def get_session():
        async_session_maker = async_sessionmaker(engine, expire_on_commit=False)
        async with async_session_maker() as session:
            yield session

    async_session_maker = async_sessionmaker(engine, expire_on_commit=False)

    yield async_session_maker

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

    await engine.dispose()


@pytest.fixture
async def session(async_session):
    async with async_session() as s:
        yield s


@pytest.fixture
async def user_repo(session):
    return UserRepository(session)


@pytest.fixture
async def prediction_repo(session):
    return PredictionRepository(session)


class TestUserRepository:
    """Тесты UserRepository."""

    @pytest.mark.asyncio
    async def test_create_user(self, user_repo):
        """Создание пользователя."""
        user = await user_repo.create_user("test@example.com", "hashed123")
        assert user.email == "test@example.com"
        assert user.hashed_password == "hashed123"
        assert user.id is not None

    @pytest.mark.asyncio
    async def test_get_by_email_found(self, user_repo):
        """Поиск существующего пользователя."""
        await user_repo.create_user("find@example.com", "pass")
        user = await user_repo.get_by_email("find@example.com")
        assert user is not None
        assert user.email == "find@example.com"

    @pytest.mark.asyncio
    async def test_get_by_email_not_found(self, user_repo):
        """Поиск несуществующего пользователя."""
        user = await user_repo.get_by_email("missing@example.com")
        assert user is None


class TestPredictionRepository:
    """Тесты PredictionRepository."""

    @pytest.mark.asyncio
    async def test_save_prediction(self, session, prediction_repo):
        """Сохранение предсказания."""
        user = await session.execute(
            __import__("sqlalchemy", fromlist=["text"]).text(
                "INSERT INTO users (email, hashed_password) VALUES (:email, :password)"
            )
        )
        # Создаем пользователя вручную через SQLAlchemy sync
        from sqlalchemy import text
        session.execute(
            text("INSERT INTO users (email, hashed_password) VALUES (:email, :password)"),
            {"email": "pred@example.com", "password": "pass"},
        )
        await session.commit()

        # Получаем ID пользователя
        result = await session.execute(text("SELECT id FROM users WHERE email = :email"))
        user_id = result.scalar()

        pred = await prediction_repo.save_prediction(
            user_id=user_id,
            input_features={"rooms": 2, "area": 50},
            predicted_price=8500000.0,
        )
        assert pred.predicted_price == 8500000.0
        assert pred.input_features == {"rooms": 2, "area": 50}

    @pytest.mark.asyncio
    async def test_get_history(self, session, prediction_repo):
        """Получение истории предсказаний."""
        from sqlalchemy import text
        result = await session.execute(text("SELECT id FROM users WHERE email = :email"))
        user_id = result.scalar()

        await prediction_repo.save_prediction(
            user_id=user_id,
            input_features={"rooms": 1},
            predicted_price=5000000.0,
        )
        await prediction_repo.save_prediction(
            user_id=user_id,
            input_features={"rooms": 3},
            predicted_price=12000000.0,
        )

        history = await prediction_repo.get_history(user_id)
        assert len(history) == 2
        assert history[0].predicted_price == 12000000.0  # newest first


class TestModelRepository:
    """Тесты ModelRepository."""

    def test_load_model(self, tmp_path):
        """Загрузка модели из файла."""
        # Создаем тестовую модель
        model_path = tmp_path / "test_model.joblib"
        dummy_model = MagicMock(predict=lambda X: [1000000.0] * len(X))
        joblib.dump(dummy_model, model_path)

        with patch.object(__import__("src.core.config", fromlist=["get_settings"]), "get_settings") as mock_settings:
            mock_settings.return_value.model_path = str(model_path)
            repo = ModelRepository()
            loaded = repo.load()
            assert loaded is dummy_model

    @pytest.mark.asyncio
    async def test_predict(self, tmp_path):
        """Предсказание через модель."""
        model_path = tmp_path / "predict_model.joblib"
        dummy_model = MagicMock(predict=lambda X: [1000000.0, 2000000.0])
        joblib.dump(dummy_model, model_path)

        with patch.object(__import__("src.core.config", fromlist=["get_settings"]), "get_settings") as mock_settings:
            mock_settings.return_value.model_path = str(model_path)
            repo = ModelRepository()
            result = await repo.predict([[1, 30], [2, 50]])
            assert result == [1000000.0, 2000000.0]
