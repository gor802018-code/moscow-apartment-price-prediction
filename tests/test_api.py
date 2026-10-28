"""Тесты для API Layer (FastAPI + TestClient + JWT)."""

from unittest.mock import MagicMock, PropertyMock, patch

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from src.api.main import app
from src.models.base import Base


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
    async def test_health_ok(self, client):
        with patch("src.api.routers.health.ModelRepository") as repo:
            repo.return_value.load.return_value = object()
            response = await client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["checks"]["database"] == "ok"

    @pytest.mark.asyncio
    async def test_health_degraded_without_model(self, client):
        """Раньше статус считался, но ответ всегда был 200."""
        with patch("src.api.routers.health.ModelRepository") as repo:
            repo.return_value.load.side_effect = FileNotFoundError("no model")
            response = await client.get("/api/v1/health")
        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "degraded"
        assert data["checks"]["model"]["status"] == "missing"


class TestAuth:
    """Тесты аутентификации."""

    @pytest.mark.asyncio
    async def test_register(self, client):
        response = await client.post("/api/v1/auth/register", json={
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
        r1 = await client.post("/api/v1/auth/register", json=payload)
        assert r1.status_code == 201, f"First register failed: {r1.json()}"

        r2 = await client.post("/api/v1/auth/register", json=payload)
        assert r2.status_code == 409

    @pytest.mark.asyncio
    async def test_login(self, client):
        # Сначала регистрируем
        await client.post("/api/v1/auth/register", json={
            "email": "login@example.com",
            "password": "SecurePass123!",
        })

        # Затем логинимся
        response = await client.post("/api/v1/auth/login", json={
            "email": "login@example.com",
            "password": "SecurePass123!",
        })
        assert response.status_code == 200, f"Login failed: {response.json()}"
        data = response.json()
        assert "access_token" in data

    @pytest.mark.asyncio
    async def test_login_wrong_password(self, client):
        await client.post("/api/v1/auth/register", json={
            "email": "wrong@example.com",
            "password": "CorrectPass123!",
        })

        response = await client.post("/api/v1/auth/login", json={
            "email": "wrong@example.com",
            "password": "WrongPassword!",
        })
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_email_is_case_insensitive(self, client):
        payload = {"email": "Case@Example.com", "password": "SecurePass123!"}
        assert (await client.post("/api/v1/auth/register", json=payload)).status_code == 201
        again = {"email": "case@example.com", "password": "SecurePass123!"}
        assert (await client.post("/api/v1/auth/register", json=again)).status_code == 409
        assert (await client.post("/api/v1/auth/login", json=again)).status_code == 200

    @pytest.mark.asyncio
    async def test_login_errors_are_indistinguishable(self, client):
        """Неверный пароль и несуществующий email дают один и тот же ответ (user enumeration)."""
        await client.post("/api/v1/auth/register", json={
            "email": "known@example.com", "password": "CorrectPass123!",
        })
        wrong_pw = await client.post("/api/v1/auth/login", json={
            "email": "known@example.com", "password": "WrongPassword!",
        })
        no_user = await client.post("/api/v1/auth/login", json={
            "email": "nobody@example.com", "password": "WrongPassword!",
        })
        assert wrong_pw.status_code == no_user.status_code == 401
        assert wrong_pw.json() == no_user.json()

    @pytest.mark.asyncio
    async def test_login_not_found(self, client):
        response = await client.post("/api/v1/auth/login", json={
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
        response = await client.post("/api/v1/predict", json=valid_prediction)
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_predict_success(self, client, valid_prediction):
        """Успешное предсказание с токеном."""
        dummy_model = MagicMock(predict=lambda X: [8500000.0] * len(X))

        with patch("src.services.prediction_service.ModelRepository") as mock_model_repo:
            mock_instance = MagicMock()
            mock_instance.model = dummy_model
            mock_model_repo.return_value = mock_instance

            # Регистрируемся
            reg_resp = await client.post("/api/v1/auth/register", json={
                "email": "pred@example.com",
                "password": "SecurePass123!",
            })
            assert reg_resp.status_code == 201, f"Register failed: {reg_resp.json()}"
            token = reg_resp.json()["access_token"]

            # Делаем предсказание
            headers = {"Authorization": f"Bearer {token}"}
            pred_resp = await client.post("/api/v1/predict", json=valid_prediction, headers=headers)

            assert pred_resp.status_code == 201, f"Predict failed: {pred_resp.json()}"
            data = pred_resp.json()
            assert "id" in data
            assert "predicted_price" in data
            assert data["predicted_price"] == 8500000.0

    @pytest.mark.asyncio
    async def test_history_success(self, client, valid_prediction):
        """История предсказаний с токеном."""
        dummy_model = MagicMock(predict=lambda X: [8500000.0] * len(X))

        with patch("src.services.prediction_service.ModelRepository") as mock_model_repo:
            mock_instance = MagicMock()
            mock_instance.model = dummy_model
            mock_model_repo.return_value = mock_instance

            # Регистрируемся
            reg_resp = await client.post("/api/v1/auth/register", json={
                "email": "hist@example.com",
                "password": "SecurePass123!",
            })
            assert reg_resp.status_code == 201, f"Register failed: {reg_resp.json()}"
            token = reg_resp.json()["access_token"]

            # Делаем предсказание
            headers = {"Authorization": f"Bearer {token}"}
            pred_resp = await client.post("/api/v1/predict", json=valid_prediction, headers=headers)
            assert pred_resp.status_code == 201, f"Predict failed: {pred_resp.json()}"

            # Получаем историю
            hist_resp = await client.get("/api/v1/predict/history", headers=headers)
            assert hist_resp.status_code == 200, f"History failed: {hist_resp.json()}"
            data = hist_resp.json()
            assert "predictions" in data

    @pytest.mark.asyncio
    async def test_history_unauthorized(self, client):
        """История без токена возвращает 401."""
        response = await client.get("/api/v1/predict/history")
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_floor_above_total_floors_returns_422(self, client, valid_prediction):
        """Раньше этот запрос приводил к 500."""
        reg = await client.post("/api/v1/auth/register", json={
            "email": "floor@example.com", "password": "SecurePass123!",
        })
        headers = {"Authorization": f"Bearer {reg.json()['access_token']}"}
        response = await client.post(
            "/api/v1/predict", json={**valid_prediction, "floor": 11}, headers=headers
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_predict_without_model_returns_503(self, client, valid_prediction):
        """Модель не обучена — это 503, а не безликий 500."""
        with patch("src.services.prediction_service.ModelRepository") as mock_model_repo:
            instance = MagicMock()
            type(instance).model = PropertyMock(side_effect=FileNotFoundError("no model"))
            mock_model_repo.return_value = instance

            reg = await client.post("/api/v1/auth/register", json={
                "email": "nomodel@example.com", "password": "SecurePass123!",
            })
            headers = {"Authorization": f"Bearer {reg.json()['access_token']}"}
            response = await client.post("/api/v1/predict", json=valid_prediction, headers=headers)

        assert response.status_code == 503
        assert "no model" not in response.text  # внутренности наружу не утекают

    @pytest.mark.asyncio
    async def test_history_is_isolated_between_users(self, client, valid_prediction):
        """Регрессия: user_id всегда был 0, и история у всех пользователей была общей."""
        dummy_model = MagicMock(predict=lambda X: [8500000.0] * len(X))

        with patch("src.services.prediction_service.ModelRepository") as mock_model_repo:
            instance = MagicMock()
            instance.model = dummy_model
            mock_model_repo.return_value = instance

            async def register(email):
                resp = await client.post("/api/v1/auth/register", json={
                    "email": email, "password": "SecurePass123!",
                })
                return {"Authorization": f"Bearer {resp.json()['access_token']}"}

            alice, bob = await register("alice@example.com"), await register("bob@example.com")
            assert (await client.post("/api/v1/predict", json=valid_prediction, headers=alice)).status_code == 201

            alice_history = (await client.get("/api/v1/predict/history", headers=alice)).json()
            bob_history = (await client.get("/api/v1/predict/history", headers=bob)).json()

        assert len(alice_history["predictions"]) == 1
        assert bob_history["predictions"] == []

    @pytest.mark.asyncio
    async def test_token_with_non_numeric_subject_returns_401(self, client):
        from src.core.security import create_access_token

        token = create_access_token("someone@example.com")  # старый формат: subject = email
        response = await client.get(
            "/api/v1/predict/history", headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 401
