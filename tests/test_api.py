"""Тесты для API Layer (FastAPI + TestClient + JWT)."""

from unittest.mock import MagicMock, patch

import pytest
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from src.api.main import app
from src.models.base import Base


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
async def session_factory(db_engine):
    """Создает factory сессий."""
    return async_sessionmaker(db_engine, expire_on_commit=False)


@pytest.fixture
async def session(session_factory):
    """Создает тестовую сессию."""
    async with session_factory() as s:
        yield s


@pytest.fixture
async def client(session_factory, session):
    """Создает TestClient с мок-зависимостями."""
    from src.api.dependencies import set_session_factory
    set_session_factory(session_factory)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


class TestHealth:
    """Тесты health check."""

    @pytest.mark.asyncio
    async def test_health(self, client):
        response = await client.get("/api/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestAuth:
    """Тесты аутентификации."""

    @pytest.mark.asyncio
    async def test_register(self, client):
        response = await client.post("/api/auth/register", json={
            "email": "test@example.com",
            "password": "SecurePass123!",
        })
        assert response.status_code == 201, f"Register failed: {response.json()}"
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert "user_id" in data

    @pytest.mark.asyncio
    async def test_register_duplicate(self, client):
        payload = {"email": "dup@example.com", "password": "SecurePass123!"}
        r1 = await client.post("/api/auth/register", json=payload)
        assert r1.status_code == 201, f"First register failed: {r1.json()}"

        r2 = await client.post("/api/auth/register", json=payload)
        assert r2.status_code == 409

    @pytest.mark.asyncio
    async def test_login(self, client):
        # Сначала регистрируем
        await client.post("/api/auth/register", json={
            "email": "login@example.com",
            "password": "SecurePass123!",
        })

        # Затем логинимся
        response = await client.post("/api/auth/login", json={
            "email": "login@example.com",
            "password": "SecurePass123!",
        })
        assert response.status_code == 200, f"Login failed: {response.json()}"
        data = response.json()
        assert "access_token" in data

    @pytest.mark.asyncio
    async def test_login_wrong_password(self, client):
        await client.post("/api/auth/register", json={
            "email": "wrong@example.com",
            "password": "CorrectPass123!",
        })

        response = await client.post("/api/auth/login", json={
            "email": "wrong@example.com",
            "password": "WrongPassword!",
        })
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_login_not_found(self, client):
        response = await client.post("/api/auth/login", json={
            "email": "missing@example.com",
            "password": "SomePassword1!",
        })
        assert response.status_code == 401


class TestPrediction:
    """Тесты предсказаний."""

    @pytest.fixture
    def valid_prediction(self):
        return {
            "rooms": 2,
            "area": 50.0,
            "floor": 5,
            "total_floors": 10,
            "year_built": 2005,
            "distance_to_metro_min": 10,
        }

    @pytest.mark.asyncio
    async def test_predict_unauthorized(self, client, valid_prediction):
        """Предсказание без токена возвращает 401."""
        response = await client.post("/api/predict", json=valid_prediction)
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_predict_success(self, client, valid_prediction):
        """Успешное предсказание с токеном."""
        # Мокаем модель, чтобы не нужно было реальной .joblib
        dummy_model = MagicMock(predict=lambda X: [8500000.0] * len(X))

        with patch("src.api.routers.prediction.ModelRepository") as mock_model_repo:
            mock_instance = MagicMock()
            mock_instance.model = dummy_model
            mock_model_repo.return_value = mock_instance

            # Регистрируемся
            reg_resp = await client.post("/api/auth/register", json={
                "email": "pred@example.com",
                "password": "SecurePass123!",
            })
            assert reg_resp.status_code == 201, f"Register failed: {reg_resp.json()}"
            token = reg_resp.json()["access_token"]

            # Делаем предсказание
            headers = {"Authorization": f"Bearer {token}"}
            pred_resp = await client.post("/api/predict", json=valid_prediction, headers=headers)

            assert pred_resp.status_code == 200, f"Predict failed: {pred_resp.json()}"
            data = pred_resp.json()
            assert "id" in data
            assert "predicted_price" in data
            assert data["predicted_price"] == 8500000.0

    @pytest.mark.asyncio
    async def test_history_success(self, client, valid_prediction):
        """История предсказаний с токеном."""
        dummy_model = MagicMock(predict=lambda X: [8500000.0] * len(X))

        with patch("src.api.routers.prediction.ModelRepository") as mock_model_repo:
            mock_instance = MagicMock()
            mock_instance.model = dummy_model
            mock_model_repo.return_value = mock_instance

            # Регистрируемся
            reg_resp = await client.post("/api/auth/register", json={
                "email": "hist@example.com",
                "password": "SecurePass123!",
            })
            assert reg_resp.status_code == 201, f"Register failed: {reg_resp.json()}"
            token = reg_resp.json()["access_token"]

            # Делаем предсказание
            headers = {"Authorization": f"Bearer {token}"}
            pred_resp = await client.post("/api/predict", json=valid_prediction, headers=headers)
            assert pred_resp.status_code == 200, f"Predict failed: {pred_resp.json()}"

            # Получаем историю
            hist_resp = await client.get("/api/predict/history", headers=headers)
            assert hist_resp.status_code == 200, f"History failed: {hist_resp.json()}"
            data = hist_resp.json()
            assert "predictions" in data

    @pytest.mark.asyncio
    async def test_history_unauthorized(self, client):
        """История без токена возвращает 401."""
        response = await client.get("/api/predict/history")
        assert response.status_code == 401
