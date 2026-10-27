"""Тесты для repository layer (SQLite in-memory + mocks)."""


import joblib
import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from src.models.base import Base
from src.repositories.model_repository import ModelRepository
from src.repositories.prediction_repository import PredictionRepository
from src.repositories.user_repository import UserRepository


@pytest.fixture(scope="session")
def event_loop():
    """Создает event loop на сессию для pytest-asyncio."""
    import asyncio
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def db_engine():
    """Создает тестовый engine."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest.fixture
async def session(db_engine):
    """Создает тестовую сессию."""
    async_session_maker = async_sessionmaker(db_engine, expire_on_commit=False)
    async with async_session_maker() as s:
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
        await session.execute(
            text("INSERT INTO users (email, hashed_password) VALUES (:email, :password)"),
            {"email": "pred@example.com", "password": "pass"},
        )
        await session.commit()

        result = await session.execute(
            text("SELECT id FROM users WHERE email = :email"),
            {"email": "pred@example.com"},
        )
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
        await session.execute(
            text("INSERT INTO users (email, hashed_password) VALUES (:email, :password)"),
            {"email": "hist@example.com", "password": "pass"},
        )
        await session.commit()

        result = await session.execute(
            text("SELECT id FROM users WHERE email = :email"),
            {"email": "hist@example.com"},
        )
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
        assert history[0].predicted_price == 12000000.0  # newest first (by id)


class _DummyModel:
    """Простая модель для тестов (поддерживает pickle)."""

    def __init__(self, predictions):
        self._predictions = predictions

    def predict(self, X):
        return self._predictions[:len(X)]


class TestModelRepository:
    """Тесты ModelRepository."""

    def test_load_model(self, tmp_path):
        """Загрузка модели из файла."""
        model_path = str(tmp_path / "test_model.joblib")
        dummy_model = _DummyModel([1000000.0] * 5)
        joblib.dump(dummy_model, model_path)

        repo = ModelRepository(model_path=model_path)
        loaded = repo.load()
        assert loaded.predict([[1]]) == [1000000.0]

    def test_model_not_found(self):
        """Файл модели не найден."""
        repo = ModelRepository(model_path="/nonexistent/path/model.joblib")
        with pytest.raises(FileNotFoundError):
            repo.load()

    @pytest.mark.asyncio
    async def test_predict(self, tmp_path):
        """Предсказание через модель."""
        model_path = str(tmp_path / "predict_model.joblib")
        dummy_model = _DummyModel([1000000.0, 2000000.0])
        joblib.dump(dummy_model, model_path)

        repo = ModelRepository(model_path=model_path)
        result = await repo.predict([[1, 30], [2, 50]])
        assert result == [1000000.0, 2000000.0]
