"""Тесты для сервисов (Auth & Prediction) с mock-репозиториями."""

from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from src.data.preprocessing import FEATURE_COLUMNS, extract_features
from src.services.auth_service import AuthService
from src.services.prediction_service import PredictionService, ValidationError

FULL_FEATURES = {
    "rooms": 2,
    "area": 50.0,
    "floor": 5,
    "total_floors": 10,
    "year_built": 2005,
    "distance_to_metro_min": 10,
}


@pytest.fixture
def mock_session():
    """Mock сессии БД."""
    session = AsyncMock()
    session.execute = AsyncMock()
    session.commit = AsyncMock()
    session.refresh = AsyncMock()
    return session


@pytest.fixture
def mock_user():
    """Mock пользователя."""
    user = MagicMock()
    user.id = 1
    user.email = "test@example.com"
    user.hashed_password = "$2b$12$..."
    return user


class TestAuthService:
    """Тесты AuthService."""

    @pytest.mark.asyncio
    async def test_register_success(self, mock_session, mock_user):
        """Успешная регистрация."""
        mock_session.execute = AsyncMock(return_value=MagicMock(scalar_one_or_none=AsyncMock(return_value=None)))
        mock_session.add = AsyncMock()
        mock_session.commit = AsyncMock()
        mock_session.refresh = AsyncMock()

        # Mock create_user
        with patch.object(AuthService, '__init__', lambda self, session: None):
            service = AuthService(mock_session)
            service.session = mock_session
            service.user_repo = AsyncMock()
            service.user_repo.get_by_email = AsyncMock(return_value=None)
            service.user_repo.create_user = AsyncMock(return_value=mock_user)

            with patch("src.services.auth_service.hash_password", return_value="hashed123"):
                with patch("src.services.auth_service.create_access_token", return_value="mock_token"):
                    result = await service.register("test@example.com", "password")

            assert "access_token" in result
            assert result["access_token"] == "mock_token"
            assert result["user_id"] == 1

    @pytest.mark.asyncio
    async def test_register_duplicate(self, mock_session, mock_user):
        """Регистрация существующего пользователя."""
        with patch.object(AuthService, '__init__', lambda self, session: None):
            service = AuthService(mock_session)
            service.session = mock_session
            service.user_repo = AsyncMock()
            service.user_repo.get_by_email = AsyncMock(return_value=mock_user)

            with pytest.raises(ValueError, match="already exists"):
                await service.register("test@example.com", "password")

    @pytest.mark.asyncio
    async def test_login_success(self, mock_session, mock_user):
        """Успешный логин."""
        with patch.object(AuthService, '__init__', lambda self, session: None):
            service = AuthService(mock_session)
            service.session = mock_session
            service.user_repo = AsyncMock()
            service.user_repo.get_by_email = AsyncMock(return_value=mock_user)

            with patch("src.services.auth_service.verify_password", return_value=True):
                with patch("src.services.auth_service.create_access_token", return_value="mock_token"):
                    result = await service.login("test@example.com", "password")

            assert result["access_token"] == "mock_token"

    @pytest.mark.asyncio
    async def test_login_user_not_found(self, mock_session):
        """Логин несуществующего пользователя."""
        with patch.object(AuthService, '__init__', lambda self, session: None):
            service = AuthService(mock_session)
            service.session = mock_session
            service.user_repo = AsyncMock()
            service.user_repo.get_by_email = AsyncMock(return_value=None)

            with pytest.raises(ValueError, match="Invalid email or password"):
                await service.login("missing@example.com", "password")

    @pytest.mark.asyncio
    async def test_login_wrong_password(self, mock_session, mock_user):
        """Логин с неверным паролем."""
        with patch.object(AuthService, '__init__', lambda self, session: None):
            service = AuthService(mock_session)
            service.session = mock_session
            service.user_repo = AsyncMock()
            service.user_repo.get_by_email = AsyncMock(return_value=mock_user)

            with patch("src.services.auth_service.verify_password", return_value=False):
                with pytest.raises(ValueError, match="Invalid email or password"):
                    await service.login("test@example.com", "wrong")


class TestPredictionService:
    """Тесты PredictionService."""

    @pytest.mark.asyncio
    async def test_predict(self, mock_session):
        """Предсказание цены."""
        mock_model = MagicMock()
        mock_model.predict = MagicMock(return_value=[8500000.0])

        mock_saved = MagicMock()
        mock_saved.id = 1
        mock_saved.created_at = "2026-01-01T00:00:00"

        with patch.object(PredictionService, '__init__', lambda self, session, model_path=None: None):
            service = PredictionService(mock_session)
            service.session = mock_session
            service.model_repo = MagicMock()
            service.model_repo.model = mock_model
            service.prediction_repo = AsyncMock()
            service.prediction_repo.save_prediction = AsyncMock(return_value=mock_saved)

            result = await service.predict(
                user_id=1,
                input_features=FULL_FEATURES,
            )

            assert result["predicted_price"] == 8500000.0
            assert result["user_id"] == 1
            service.prediction_repo.save_prediction.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_history(self, mock_session):
        """Получение истории предсказаний."""
        mock_pred1 = MagicMock(id=1, input_features={"rooms": 2}, predicted_price=8500000.0, created_at="2026-01-01")
        mock_pred2 = MagicMock(id=2, input_features={"rooms": 3}, predicted_price=12000000.0, created_at="2026-01-02")

        with patch.object(PredictionService, '__init__', lambda self, session, model_path=None: None):
            service = PredictionService(mock_session)
            service.session = mock_session
            service.prediction_repo = AsyncMock()
            service.prediction_repo.get_history = AsyncMock(return_value=[mock_pred1, mock_pred2])

            history = await service.get_history(user_id=1)

            assert len(history) == 2
            assert history[0]["predicted_price"] == 8500000.0

    @pytest.mark.asyncio
    async def test_predict_passes_training_features_to_model(self, mock_session):
        """Модель получает все 9 признаков в порядке обучения (производные не нули)."""
        mock_model = MagicMock()
        mock_model.predict = MagicMock(return_value=[1.0])
        mock_saved = MagicMock(id=1, created_at="2026-01-01T00:00:00")

        with patch.object(PredictionService, "__init__", lambda self, session, model_path=None: None):
            service = PredictionService(mock_session)
            service.session = mock_session
            service.model_repo = MagicMock()
            service.model_repo.model = mock_model
            service.prediction_repo = AsyncMock()
            service.prediction_repo.save_prediction = AsyncMock(return_value=mock_saved)

            await service.predict(user_id=1, input_features=FULL_FEATURES)

        vector = mock_model.predict.call_args[0][0][0]  # predict([vector]) -> первый (единственный) объект
        assert len(vector) == len(FEATURE_COLUMNS)
        expected = extract_features(pd.DataFrame([FULL_FEATURES])).iloc[0][FEATURE_COLUMNS]
        assert vector == [float(v) for v in expected]
        assert vector[FEATURE_COLUMNS.index("area_per_room")] == 25.0

    @pytest.mark.asyncio
    async def test_predict_floor_above_total_floors(self, mock_session):
        with patch.object(PredictionService, "__init__", lambda self, session, model_path=None: None):
            service = PredictionService(mock_session)
            with pytest.raises(ValidationError, match="floor"):
                await service.predict(user_id=1, input_features={**FULL_FEATURES, "floor": 11})

    @pytest.mark.asyncio
    async def test_predict_missing_features(self, mock_session):
        """Раньше недостающие признаки молча заменялись нулями."""
        with patch.object(PredictionService, "__init__", lambda self, session, model_path=None: None):
            service = PredictionService(mock_session)
            with pytest.raises(ValidationError, match="Missing features"):
                await service.predict(user_id=1, input_features={"rooms": 2, "area": 50.0})
